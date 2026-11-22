"""One isolated device process per plan. Every prefix passes real numerical admission."""
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from stage_a.compiler import plan as single
from stage_a.evidence import artifact, errors, features, measure, statistics_for
from stage_a.env import telemetry
from stage_a.numeric import configure_matmul_precision
from stage_a.util import Rejected, digest,save
from stage_a.workloads import Workload
from .importer import read
from .planner import decode,derive,key
from .runtime import JointWorkload

def check_tensor(torch,job,oracle,launch):
    def poison_writes():
        job.output.fill_(float("nan"))
        if job.w=="W8":
            job.kcache[:,job.KL-job.L:].fill_(float("nan"))
            job.vcache[:,job.KL-job.L:].fill_(float("nan"))
    poison_writes();launch();first=job.output.clone()
    poison_writes();launch();torch.cuda.synchronize()
    result=errors(job.output,oracle,job.w)
    result["repeat_bitwise"]=torch.equal(first,job.output)
    passed=result["passed"] and result["repeat_bitwise"]
    if hasattr(job,"state"):
        result["all_tasks_once"]=bool((job.state==2).all() and (job.count==1).all())
        passed &= result["all_tasks_once"]
    if job.w=="W8":
        result["cache"]=[errors(x,y,job.w) for x,y in zip((job.kcache,job.vcache),job.expected_cache)]
        passed &= all(x["passed"] for x in result["cache"])
    result["admitted"]=bool(passed)
    return result

def run(req,rec):
    from stage_a.toolchain import require_triton
    rec["triton_version"]=require_triton()
    import torch
    case=req["case"];device="cuda:0";path=decode(req["sequence"]);folder=Path(req["folder"])
    torch.cuda.set_device(device);configure_matmul_precision(torch)
    prop=torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(min(1.,req["memory_gb"]*1024**3/prop.total_memory),device)
    legacy=req.get("legacy",False)
    prefixes=[path[:i] for i in range(len(path)+1)]
    exes=[derive(case,p) for p in prefixes]
    hashes=[key(case,x) for x in exes]
    mode=None;symbols={};objects={};binaries={};seen=set();reps=req["batches"]
    for profile in ["extreme"]+(["stride"] if case["workload"]=="W1" else [])+["normal"]:
        for idx,(prefix,exe,h) in enumerate(zip(prefixes,exes,hashes)):
            # The imported single baseline calls the unchanged Stage A workload and plan.
            use_legacy=legacy
            job=(Workload if use_legacy else JointWorkload)(case,device,req["input_seed"],req["assets"],profile)
            if job.w=="W8":
                if mode is None:
                    probe=Workload(case,device,req["input_seed"],req["assets"],"normal")
                    mode=probe.qualify_qkv_projection();del probe
                job.qkv_bias_mode=mode["mode"];job.qkv_projection_qualification=mode
                if not use_legacy and exe.head_group!=1:
                    job.edges=exe.ddg["data"];job.dispatch_data(len(job.edges))
            oracle=job.reference(oracle=True)
            if idx==0:
                ck=errors(job.reference(),oracle,job.w);rec["checks"].append(dict(profile=profile,variant="B0",**ck))
                if not ck["passed"]:raise Rejected("FAIL_CORRECTNESS","PyTorch comparator failed the independent oracle")
            if use_legacy:
                original=single(case,idx>0)
                if mode:original.proof["qkv_bias_mode"]=mode["mode"]
                launch,grid=job.bind(original)
            else:launch,grid=job.bind(exe)
            # Use exactly the Stage A launch arguments, without a profiling wrapper in timed code.
            args=inspect.getclosurevars(launch).nonlocals
            rec["phase"]="compile"
            rec["compile_attempts"]+=1
            start=time.perf_counter()
            try:
                binary=args["fn"].warmup(*args["args"],grid=grid,num_warps=args["warps"],num_stages=2,enable_fp_fusion=False)
            finally:rec["compile_api_seconds"]+=time.perf_counter()-start
            if binary.hash not in seen:
                seen.add(binary.hash);rec["compiled_specializations"]=len(seen)
                rec["compile_seconds"]+=time.perf_counter()-start
            shared_limit=getattr(prop,"shared_memory_per_block_optin",None) or 101376
            rec["resource_checks"].append(dict(profile=profile,step=idx,plan=h,grid=list(grid),shared=binary.metadata.shared,
                                                limit=shared_limit,num_warps=args["warps"],artifact_hash=binary.hash))
            if binary.metadata.shared>shared_limit:
                raise Rejected("RESOURCE_EXCEEDED","compiled shared memory exceeds SM120 per-block limit")
            rec["phase"]="correctness"
            ck=check_tensor(torch,job,oracle,launch)
            rec["checks"].append(dict(profile=profile,variant=h,**ck))
            if not ck["admitted"]:raise Rejected("FAIL_CORRECTNESS","prefix failed "+profile+" at "+h)
            if profile=="normal":
                ir=folder/("prefix_"+str(idx))
                save(ir/"plan.json",exe.json());save(ir/"ownership.json",exe.ownership);save(ir/"ddg.json",exe.ddg)
                meta=artifact(binary,ir,grid,device)
                meta["origin"]="REGENERATED" if legacy else "STAGE_B_COMPILED"
                if not all(meta["hashes"].get(k) for k in ("ttir","ttgir","llir","ptx","cubin")):
                    raise Rejected("BLOCKED","incomplete physical IR / binary provenance")
                if job.w=="W1" and exe.local and case["params"]["layout"]!="thread":
                    kind="shuffle" if case["params"]["layout"]=="lane" else "shared"
                    if not meta["ptx_counts"][kind]:raise Rejected("UNSUPPORTED","missing "+kind+" in PTX")
                cert=digest([meta["hashes"]["cubin"],grid,exe.ddg.get("execution"),exe.ddg.get("data")])
                binaries[h]=cert
                rec["prefix_artifacts"].append(dict(plan=h,step=idx,meta=meta,executable=cert))
                if idx and cert in list(binaries.values())[:-1]:
                    raise Rejected("NO_OP","machine program and dispatch match a strict prefix")
                if idx==0:
                    objects["base"]=job;symbols.update(B0=job.reference,B1=launch)
                if idx==len(path) and idx:
                    objects["subject"]=job;symbols["candidate"]=launch
                if idx:
                    import difflib
                    prior=folder/("prefix_"+str(idx-1))/"kernel.ttgir"
                    (ir/"change.diff").write_text("".join(difflib.unified_diff(prior.read_text().splitlines(True),(ir/"kernel.ttgir").read_text().splitlines(True))),encoding="utf8")
            del oracle
    rec["phase"]="benchmark";rec["gpu_before"]=telemetry(req["selector"])
    timings,order=measure(symbols,reps,req["replays"],req["input_seed"],device,req["graph"])
    rec["gpu_after"]=telemetry(req["selector"]);rec["raw_us"]=timings;rec["order"]=order
    rec["timer"]="CUDA_EVENTS_GRAPH" if req["graph"] else "CUDA_EVENTS_EAGER"
    rec["statistics"]={m:statistics_for(t,req["input_seed"]) for m,t in timings.items()}
    subject="candidate" if path else "B1"
    rec["latency_us"]=rec["statistics"][subject]["median"]
    rec["paired_speedup_ci95"]=statistics_for([t/u for t,u in zip(timings["B1"],timings[subject])],req["input_seed"])["ci95"]
    meta=rec["prefix_artifacts"][-1]["meta"];exe=exes[-1]
    # Use the original cost extractors without adding single-strategy gains or times.
    detached={}
    for strategy in sorted(set(a.strategy for a in path) or {"none"}):
        proof={"ordered_partition":exe.proof["S4"]["partition"]} if exe.cut else {}
        data=SimpleNamespace(ownership=exe.ownership,ddg=exe.ddg,proof=proof)
        vals,src=features({**case,"strategy":strategy},data,meta)
        if "data" in exe.ddg:
            vals["workers"]=exe.workers;src["workers"]="STATIC_ESTIMATE"
        if strategy=="S1" and case["workload"]=="W7":
            vals["local_handoff_ratio"]=((exe.local&1)>0)*.5+((exe.local&2)>0)*.5
        detached[strategy]=dict(values=vals,sources=src)
    rec["cost_by_strategy"]=detached
    rec["qualification"]=mode;rec["weights_loaded"]=job.loaded;rec["contract"]=job.input_contract
    if hasattr(job,"state"):
        if legacy:
            tracer,_=job.bind(original,True)
        else:tracer,_=job.bind(exe,True)
        tracer();torch.cuda.synchronize()
        stamps=job.trace.cpu().tolist()
        if not bool((job.count==1).all() and (job.state==2).all()) or any(end<start for start,end,_ in stamps):
            raise Rejected("RUNTIME_ERROR","instrumented task completeness")
        parents=exe.ddg["execution"]
        if any(stamps[j][1]>stamps[i][0] for i,ds in enumerate(parents) for j in ds):
            raise Rejected("RUNTIME_ERROR","publication order violated")
        longest=[]
        for i,ds in enumerate(parents):
            longest.append(stamps[i][1]-stamps[i][0]+max((longest[j] for j in ds),default=0))
        from statistics import median
        gaps=[stamps[i][0]-max(stamps[j][1] for j in ds) for i,ds in enumerate(parents) if ds]
        rec.update(trace_ns=stamps,instrumented_critical_path_ns=max(longest),
                   instrumented_cross_worker_edges=sum(stamps[i][2]!=stamps[j][2] for i,ds in enumerate(parents) for j in ds),
                   ready_gap_median_ns=median(gaps) if gaps else "NOT_APPLICABLE")
    rec["status"]="PASS";rec["phase"]="complete"

def main(request,output):
    from stage_a.offline import activate
    activate();req=read(request)
    start=time.perf_counter()
    rec=dict(status="BLOCKED",phase="admission",checks=[],resource_checks=[],prefix_artifacts=[],compile_seconds=0.,compile_api_seconds=0.,
             compiled_specializations=0,compile_attempts=0,sequence=req["sequence"],input_seed=req["input_seed"],identity=req["identity"])
    try:run(req,rec)
    except Rejected as err:rec.update(status=err.status,reason=str(err))
    except Exception as err:
        msg=str(err).lower()
        status="OOM" if "out of memory" in msg else "RESOURCE_EXCEEDED" if "out of resource" in msg else "COMPILE_ERROR" if rec["phase"]=="compile" else "RUNTIME_ERROR"
        rec.update(status=status,reason=repr(err),traceback=traceback.format_exc(),
                   fatal=any(x in msg for x in ("illegal memory","device-side","device lost","illegal instruction","unspecified launch failure")))
    rec["evaluation_seconds"]=time.perf_counter()-start
    save(output,rec)

if __name__=="__main__":main(*sys.argv[1:])
