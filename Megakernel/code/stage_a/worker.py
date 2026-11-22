"""Isolated GPU case process; output is atomically published even on a candidate failure."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from .compiler import plan
from .evidence import artifact,diff,errors,features,measure,statistics_for
from .legality import validate_stats
from .util import Rejected,save


def execute(req,rec):
    from .toolchain import require_triton
    rec["triton_version"]=require_triton()
    import torch
    from .workloads import Workload
    from .env import telemetry
    from .numeric import configure_matmul_precision,projection_diagnostics
    torch.cuda.set_device(req["device"])
    rec["matmul_precision"]=configure_matmul_precision(torch)
    cap=req["max_memory_gb"]*(1024**3)
    torch.cuda.set_per_process_memory_fraction(min(1.,cap/torch.cuda.get_device_properties(req["device"]).total_memory),req["device"])
    case=req["case"];folder=Path(req["folder"]);p=case["params"]
    plans={"B1":plan(case,False)}
    fallback=None
    try:plans["B2"]=plan(case,True)
    except Rejected as exc:
        if exc.status not in ("NO_OP","UNSUPPORTED","NOT_APPLICABLE","INVALID_REJECTED"):raise
        fallback=exc
        rec["admission"]=dict(status=exc.status,reason=str(exc))
        plans["B2"]=plans["B1"]

    qualification=None
    if case["workload"]=="W8":
        probe=Workload(case,req["device"],req["seed"],req["assets"],"normal")
        qualification=probe.qualify_qkv_projection()
        rec["qkv_projection_qualification"]=qualification
        del probe
    for mode in ("B1","B2"):
        if qualification is not None:
            plans[mode].proof["qkv_bias_mode"]=qualification["mode"]
        save(folder/(mode+"_plan.json"),plans[mode].json())
        save(folder/mode/"ownership.json",plans[mode].ownership)
        save(folder/mode/"ddg.json",plans[mode].ddg)
    active=plans["B2"]
    if active.stats:
        stats=active.proof["statistics"]
        validate_stats("layer" if stats[0]=="mean" else "softmax" if stats[0]=="max" else "rms",
                       active.proof["original_version_live"],True,len(stats),True)
    rec.update(legality="PASS",site=active.site,plan=active.json())
    dev=req["device"];symbols={}
    # Both dtype variants occur in the registered matrix. Check stability on repeated launches.
    for prof in ["extreme"]+(["stride"] if case["workload"]=="W1" else [])+["normal"]:
        job=Workload(case,dev,req["seed"],req["assets"],prof)
        if qualification is not None:
            job.qkv_bias_mode=qualification["mode"]
            job.qkv_projection_qualification=qualification
        rec["input_contract"]=job.input_contract
        oracle=job.reference(oracle=True)
        rec["weights_loaded"]=job.loaded
        check=errors(job.reference(),oracle,job.w)
        rec["checks"].append(dict(profile=prof,variant="B0",**check))
        if not check["passed"]:
            try:rec["numerical_diagnostics"]=projection_diagnostics(job,False)
            except Exception as error:rec["diagnostic_error"]=repr(error)
            raise Rejected("FAIL_CORRECTNESS","unfused vs independent reference")
        for mode in ("B1","B2"):
            if mode=="B2" and job.w in ("W6","W8") and plans["B2"].ddg.get("data")!=plans["B1"].ddg.get("data"):
                job=Workload(case,dev,req["seed"],req["assets"],prof)
                if qualification:
                    job.qkv_bias_mode=qualification["mode"];job.qkv_projection_qualification=qualification
                oracle=job.reference(oracle=True)
            fn,grid=job.bind(plans[mode])
            rec["phase"]="compile"
            binary=fn();torch.cuda.synchronize()
            rec["phase"]="correctness"
            first=job.output.clone()
            fn();torch.cuda.synchronize()
            check=errors(job.output,oracle,job.w)
            check["repeat_bitwise"]=torch.equal(first,job.output)
            if hasattr(job,"state"):
                check["all_tasks_once"]=bool((job.count==1).all() and (job.state==2).all())
            if job.w=="W8":
                check["kv_cache"]=[errors(tensor,expected,job.w) for tensor,expected in zip((job.kcache,job.vcache),job.expected_cache)]
                check["kv_pass"]=all(row["passed"] for row in check["kv_cache"])
            rec["checks"].append(dict(profile=prof,variant=mode,**check))
            if not all(check.get(k,True) for k in ("passed","all_tasks_once","repeat_bitwise","kv_pass")):
                try:
                    rec["failure_artifacts"]=artifact(binary,folder/("failed_"+mode),grid,dev)
                    rec["numerical_diagnostics"]=projection_diagnostics(job,True)
                except Exception as error:rec["diagnostic_error"]=repr(error)
                raise Rejected("FAIL_CORRECTNESS",f"{mode} failed {prof} accuracy or repeatability")
            if prof=="normal":
                if mode=="B1":symbols["B0"]=job.reference
                symbols[mode]=fn
                rec["metadata"][mode]=artifact(binary,folder/mode,grid,dev)
        del oracle
    diff(folder)
    meta=rec["metadata"]
    if any("cubin" not in x["hashes"] or "ptx" not in x["hashes"] for x in meta.values()):
        raise Rejected("BLOCKED","missing compiled binary or PTX provenance")
    changed=meta["B1"]["hashes"]["cubin"]!=meta["B2"]["hashes"]["cubin"]
    graph_change=active.strategy=="S5" and plans["B1"].ddg["execution"]!=active.ddg["execution"]
    grid_change=meta["B1"]["grid"]!=meta["B2"]["grid"]
    effect=changed or grid_change or graph_change
    if fallback and effect:raise RuntimeError("declined action changed the baseline executable")
    if not fallback and not effect:fallback=Rejected("NO_OP","compiled cubin, grid and task constraints coincide")
    if not fallback and job.w=="W1" and active.strategy=="S1" and p["layout"] in ("lane","warp"):
        key="shuffle" if p["layout"]=="lane" else "shared"
        rec["physical_handoff"]=key if meta["B2"]["ptx_counts"][key] else "UNCONFIRMED"
        if rec["physical_handoff"]=="UNCONFIRMED":fallback=Rejected("UNSUPPORTED","requested handoff not witnessed in PTX: "+key)
    if fallback:
        rec["phase"]="benchmark_fallback"
        rec["gpu_before"]=telemetry(req["selector"])
        # Both labels dispatch the actually qualified untransformed plan.
        times,order=measure({"B0":symbols["B0"],"B1":symbols["B1"],"fallback":symbols["B1"]},
                            req["batches"],req["replays"],req["seed"],dev,req["graph"])
        rec["gpu_after"]=telemetry(req["selector"])
        rec.update(status=fallback.status,legality=fallback.status,reason=str(fallback),
                   fallback=dict(status="MEASURED",requested=case["strategy"],executed="none",raw_us=times,
                   statistics={k:statistics_for(v,req["seed"]) for k,v in times.items()},order=order,
                   timer="CUDA_EVENTS_GRAPH" if req["graph"] else "CUDA_EVENTS_EAGER",binary=meta["B1"]["hashes"]["cubin"]))
        for ck in rec["checks"]:
            if ck["variant"]=="B2":ck["variant"]="declined_candidate" if effect else "fallback"
        return
    rec["effect"]=dict(binary_changed=changed,grid_changed=grid_change,execution_edges_changed=graph_change)
    rec["phase"]="benchmark"
    rec["gpu_before"]=telemetry(req["selector"])
    raw,order=measure(symbols,req["batches"],req["replays"],req["seed"],dev,req["graph"])
    rec.update(raw=raw,order=order,timer="CUDA_EVENTS_GRAPH" if req["graph"] else "CUDA_EVENTS_EAGER",
               gpu_after=telemetry(req["selector"]))
    rec["statistics"]={k:statistics_for(v,req["seed"]) for k,v in raw.items()}
    b0,b1,b2=[rec["statistics"][k]["median"] for k in ("B0","B1","B2")]
    rec.update(incremental_speedup=b1/b2,total_speedup=b0/b2,saving_us=b1-b2)
    rec["paired_speedup_ci95"]=statistics_for([x/y for x,y in zip(raw["B1"],raw["B2"])],req["seed"])["ci95"]
    rec["cost"],rec["sources"]=features(case,active,meta["B2"])
    if hasattr(job,"state"):
        fn,_=job.bind(active,True);fn();torch.cuda.synchronize()
        stamps=job.trace.cpu().tolist()
        if any(last<first for first,last,_ in stamps) or not bool((job.count==1).all()):
            raise Rejected("RUNTIME_ERROR","separate trace inconsistent")
        rec["task_trace_ns"]=stamps
        rec["ready_gap_ns"]=[max(0,stamps[i][0]-max(stamps[j][1] for j in ds)) if ds else None for i,ds in enumerate(active.ddg["execution"])]
        path=[]
        for i,parents in enumerate(active.ddg["execution"]):
            if any(stamps[j][1]>stamps[i][0] for j in parents):
                raise Rejected("RUNTIME_ERROR","instrumented consumer preceded completed predecessor")
            path.append(stamps[i][1]-stamps[i][0]+max((path[j] for j in parents),default=0))
        gaps=[g for g in rec["ready_gap_ns"] if g is not None]
        span=max(v[1] for v in stamps)-min(v[0] for v in stamps)
        busy=sum(v[1]-v[0] for v in stamps)
        crossed=sum(stamps[i][2]!=stamps[j][2] for i,parents in enumerate(active.ddg["data"]) for j in parents)
        additional=dict(instrumented_cross_worker_edges=crossed,instrumented_critical_path_ns=max(path),ready_gap_median_ns=statistics.median(gaps) if gaps else "NOT_APPLICABLE",
                        instrumented_task_span_ns=span,
                        instrumented_worker_free_fraction=1-busy/(max(span,1)*p.get("workers",1)))
        rec["cost"].update(additional)
        rec["sources"].update({k:"DERIVED" if isinstance(v,(float,int)) else v for k,v in additional.items()})
    rec["optional"]={}
    if job.w in ("W5","W8"):
        try:
            alt=lambda:job.reference(sdpa=True)
            check=errors(alt(),job.reference(True),job.w)
            rec["optional"]["torch_sdpa"]=dict(check=check,status="FAIL_CORRECTNESS",triton_version=rec["triton_version"])
            if check["passed"]:
                times,_=measure({"sdpa":alt},req["batches"],req["replays"],req["seed"],dev,req["graph"])
                rec["optional"]["torch_sdpa"].update(status="MEASURED",samples_us=times["sdpa"])
        except Exception as ex:
            if any(word in str(ex).lower() for word in ("illegal memory","device-side","device lost")):raise
            rec["optional"]["torch_sdpa"]=dict(status="SKIPPED_OPTIONAL",reason=repr(ex))
    rec["status"]="PASS"


def main(reqpath,out):
    from .offline import activate
    activate()
    req=json.loads(Path(reqpath).read_text(encoding="utf8"))
    rec=dict(**req["case"],input_seed=req["seed"],status="BLOCKED",checks=[],metadata={},phase="plan",legality="UNAVAILABLE")
    try:execute(req,rec)
    except Rejected as e:
        rec.update(status=e.status,reason=str(e))
        if e.status=="INVALID_REJECTED":rec["legality"]="INVALID_REJECTED"
    except Exception as e:
        msg=repr(e);text=msg.lower()
        status="OOM" if "out of memory" in text else "BLOCKED" if "blocked:" in text else "COMPILE_ERROR" if rec["phase"]=="compile" else "RUNTIME_ERROR"
        rec.update(status=status,reason=msg,traceback=traceback.format_exc(),fatal_device=any(x in text for x in ("illegal memory","device-side","unspecified launch failure","device lost","illegal instruction")))
    save(out,rec)

if __name__=="__main__":main(*sys.argv[1:])
