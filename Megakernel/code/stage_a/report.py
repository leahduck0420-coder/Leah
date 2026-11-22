import collections
import json
import math
from pathlib import Path
import statistics
from .util import file_hash,save,table
from .toolchain import matches_triton

DIRS=("00_environment","01_correctness","02_performance","03_cost","04_ir","05_analysis","06_reports","07_dataset","08_logs")
FIGS=["F1_s1_handoff","F2_s2_merge","F3_s3_reconstruction","F4_s4_slice","F5_s5_scheduling","F6_shape_heatmap","F7_cost_correlation","F8_correctness_stability"]
TRAINING_COLUMNS=("case_id","candidate_id","workload","strategy","status","reason","legality","correctness","dtype",
                  "params","shape_and_tile","baseline","program_count","logical_owner_grid","owner_coverage_elements",
                  "owner_multiplicity_padded","cross_program_dependencies","local_handoff_ratio","consumer_count",
                  "flops_main_matmul","nominal_main_arithmetic_ops","logical_io_lower_bound_bytes","baseline_intermediate_write_bytes",
                  "estimated_operand_bytes","reconstruction_count","reconstruction_arithmetic_ops","reduction_slices",
                  "workers","dag_depth","execution_constraint_edges","instrumented_cross_worker_edges","instrumented_critical_path_ns","ready_gap_median_ns","registers","smem","spill",
                  "theoretical_occupancy","gpu_latency_us","baseline_latency_us","incremental_speedup","total_speedup",
                  "absolute_saving_us","batches","std_us","paired_speedup_ci95","environment_id","cost_sources")


def figures(folder,rows,reason):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize,TwoSlopeNorm
    passed=[r for r in rows if valid(r)]
    available=[]
    for i,name in enumerate(FIGS):
        fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained")
        data=[r for r in passed if i>=5 or r["strategy"]=="S"+str(i+1)]
        if not data:
            ax.set_axis_off();ax.text(.5,.5,"No eligible GPU measurements\n"+reason[:120],ha="center",va="center",wrap=True,transform=ax.transAxes)
        elif i==5:
            # Actual shape+dtype rows; paginate horizontally by workload via labeled case index.
            def shape(r):
                p=r["params"];return r["workload"]+" "+",".join(str(p[k]) for k in ("m","n","k","seq","tasks","batch","d") if k in p)
            groups=sorted({shape(r) for r in data});values=np.full((5,len(groups)),np.nan)
            for j,s in enumerate(("S1","S2","S3","S4","S5")):
                for k,g in enumerate(groups):
                    v=[r["incremental_speedup"] for r in data if r["strategy"]==s and shape(r)==g]
                    if v:values[j,k]=statistics.median(v)
            color=ax.imshow(values,aspect="auto",interpolation="nearest",cmap="RdBu",norm=TwoSlopeNorm(vcenter=1,vmin=min(.99,np.nanmin(values)),vmax=max(1.01,np.nanmax(values))))
            ix=list(range(0,len(groups),max(1,len(groups)//18)));ax.set_xticks(ix,[groups[k] for k in ix],rotation=75,fontsize=7)
            ax.set_yticks(range(5),["S1","S2","S3","S4","S5"]);fig.colorbar(color,ax=ax,label="Median incremental speedup")
        elif i==7:
            counts=collections.Counter(r["status"] for r in rows)
            ax.bar(list(counts),list(counts.values()));ax.tick_params(axis="x",rotation=22);ax.set_ylabel("Configurations")
            bx=ax.twinx();cv=[r["statistics"]["B2"]["std"]/r["statistics"]["B2"]["mean"] for r in data]
            bx.plot([list(counts).index("PASS")],[statistics.median(cv)],"ro");bx.set_ylabel("Median batch CV (red point)")
        else:
            specs=[("layout","B2 latency (us)"),("merge","Incremental speedup"),("reconstruction_count","Incremental speedup"),
                   ("cut","B2 latency (us)"),("workers","B2 latency (us)"),None,("registers","B2 latency (us)")]
            key,ylabel=specs[i]
            scale=None
            if i in (1,4):
                col=[r["cost"].get("registers") if i==1 else r["params"].get("tasks",len(r["plan"]["ddg"].get("data",[]))) for r in data]
                numbers=[v for v in col if isinstance(v,(int,float))]
                if numbers:scale=Normalize(min(numbers),max(numbers) if max(numbers)>min(numbers) else min(numbers)+1)
            for label in sorted({r["workload"] for r in data}):
                x=[];y=[];colors=[]
                for r in data:
                    if r["workload"]!=label:continue
                    par=r["params"]; cost=r["cost"]
                    val=par.get(key,cost.get(key,None))
                    if i==0 and val is None:val="same_program"
                    if val in ("UNAVAILABLE","NOT_APPLICABLE",None):continue
                    x.append(val);y.append(r["incremental_speedup"] if i in (1,2) else r["statistics"]["B2"]["median"])
                    colors.append(cost.get("registers") if i==1 else par.get("tasks",len(r["plan"]["ddg"].get("data",[]))) if i==4 else None)
                if x:
                    shading={"c":colors,"cmap":"viridis","norm":scale} if scale is not None and all(isinstance(c,(int,float)) for c in colors) else {}
                    ax.scatter(x,y,s=19,alpha=.76,label=label,**shading)
            if scale is not None:fig.colorbar(plt.cm.ScalarMappable(norm=scale,cmap="viridis"),ax=ax,label="Registers/thread" if i==1 else "Tasks")
            ax.set_xlabel(key);ax.set_ylabel(ylabel)
            if ax.collections:ax.legend(fontsize=8)
            if i in (1,2):ax.axhline(1,color=".45",lw=1)
        ax.set_title(name.replace("_"," "));fig.savefig(folder/(name+".png"),dpi=160);plt.close(fig)
        available.append(dict(name=name,actual_points=len(data)))
    return available


def valid(r):
    if not matches_triton(r.get("triton_version")):return False
    if r.get("status")!="PASS" or r.get("legality")!="PASS" or not r.get("effect"):return False
    variants=("B0","B1","B2")
    if any(len(r.get("raw",{}).get(k,[]))<10 or any(not math.isfinite(x) or x<=0 for x in r["raw"][k]) for k in variants):return False
    for mode in variants[1:]:
        if not all(r.get("metadata",{}).get(mode,{}).get("hashes",{}).get(ext) for ext in ("cubin","ptx")):return False
    checks=r.get("checks",[])
    profiles={"normal","extreme"}|({"stride"} if r.get("workload")=="W1" else set())
    if not {(v,p) for v in variants for p in profiles}<={(x.get("variant"),x.get("profile")) for x in checks}:return False
    for x in checks:
        if not x["passed"]:return False
        if x["variant"]=="B0":continue
        if not x.get("repeat_bitwise"):return False
        if r.get("workload") in ("W6","W8") and not x.get("all_tasks_once"):return False
        if r.get("workload")=="W8" and not x.get("kv_pass"):return False
    return True


def generate(root,env,assets,protocol,rows,cpu,micro):
    e,c,p,k,ir,a,d,g,log=[root/x for x in DIRS]
    for f in (e,c,p,k,ir,a,d,g,log):f.mkdir(parents=True,exist_ok=True)
    corr=[];raw=[];summ=[];dataset=[];meta=[];legal=[];cost=[];orders=[]
    for r in rows:
        base=dict(case_id=r["id"],candidate_id=r["id"]+"-B2",workload=r["workload"],strategy=r["strategy"])
        summary={**base,"status":r["status"],"reason":r.get("reason",""),"params":r["params"]}
        legal.append({**base,"legality":r.get("legality","UNAVAILABLE"),"execution":r["status"],"proof":r.get("plan",{}).get("proof")})
        corr.extend({**base,**x} for x in r.get("checks",[]))
        dataset_row={**summary,"anchor_id":r.get("anchor_id"),"input_seed":r.get("input_seed"),"executed_strategy":r["strategy"] if valid(r) else "none" if r.get("fallback") else "UNAVAILABLE","legality":r.get("legality","UNAVAILABLE"),"dtype":r["params"]["dtype"],"shape_and_tile":r["params"],
                     "baseline":"B1","correctness":("PASS" if valid(r) else "BLOCKED" if r["status"]=="PASS" else r["status"]),
                     "environment_id":protocol["environment_id"],"weights":assets["weights"] if r["workload"] in ("W7","W8") else "NOT_APPLICABLE",
                     "loaded_real_weights":r.get("weights_loaded",False),"cost_sources":r.get("sources",{})}
        for mode,obj in r.get("metadata",{}).items():meta.append({**base,"variant":mode,**obj})
        if valid(r):
            for mode,z in r["statistics"].items():summ.append({**base,"variant":mode,**z})
            for mode,times in r["raw"].items():raw.extend({**base,"variant":mode,"batch":idx,"latency_us":x,"method":r["timer"]} for idx,x in enumerate(times))
            f=r["cost"];cost.append({**base,**f,"sources":r["sources"]})
            dataset_row.update(**f,baseline_latency_us=r["statistics"]["B1"]["median"],gpu_latency_us=r["statistics"]["B2"]["median"],
                               incremental_speedup=r["incremental_speedup"],total_speedup=r["total_speedup"],absolute_saving_us=r["saving_us"],
                               batches=len(r["raw"]["B2"]),std_us=r["statistics"]["B2"]["std"],paired_speedup_ci95=r["paired_speedup_ci95"])
            orders.append({**base,"before":r["gpu_before"],"after":r["gpu_after"],"order":r["order"]})
        dataset.append(dataset_row)
    def lines(path,records):path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records),encoding="utf8")
    table(c/"correctness_cases.csv",corr,["case_id","variant","passed"]);table(c/"numerical_errors.csv",corr,["case_id","max_abs_error","max_rel_error","RMS_error"])
    lines(c/"legality_results.jsonl",legal+cpu)
    counts=dict(collections.Counter(r["status"] for r in rows))
    save(c/"correctness_summary.json",dict(status_counts=counts,cpu=cpu))
    table(p/"raw_timings.csv",raw,["case_id","variant","batch","latency_us"])
    table(p/"baseline_timings.csv",[x for x in raw if x["variant"]!="B2"],["case_id","variant","latency_us"])
    table(p/"optimized_timings.csv",[x for x in raw if x["variant"]=="B2"],["case_id","variant","latency_us"])
    table(p/"benchmark_summary.csv",summ,["case_id","variant","median","ci95"])
    table(p/"speedups.csv",[{k:v for k,v in x.items() if k in ("case_id","workload","strategy","status","incremental_speedup","total_speedup","absolute_saving_us","paired_speedup_ci95")} for x in dataset],["case_id","status","incremental_speedup"])
    lines(k/"compilation_metadata.jsonl",meta);table(k/"resource_usage.csv",meta,["case_id","registers","smem","spill"])
    table(k/"cost_features.csv",cost,["case_id","sources"])
    table(k/"ownership_features.csv",[{key:x[key] for key in ("case_id","program_count","logical_owner_grid","owner_coverage_elements","local_handoff_ratio")} for x in cost],["case_id"])
    table(k/"dag_features.csv",[{key:x[key] for key in ("case_id","workers","cross_program_dependencies","dag_depth")} for x in cost],["case_id"])
    lines(k/"timing_context.jsonl",orders)
    opt=[{"case_id":r["id"],"baseline":n,**v} for r in rows for n,v in r.get("optional",{}).items()]
    lines(p/"optional_baselines.jsonl",opt)
    schema=list(dict.fromkeys([*TRAINING_COLUMNS,*(key for r in dataset for key in r)]))
    table(g/"stage_a_dataset.csv",[{key:r.get(key,"UNAVAILABLE") for key in schema} for r in dataset],schema)
    save(g/"stage_a_dataset_schema.json",dict(columns=schema,missing="UNAVAILABLE",inapplicable="NOT_APPLICABLE",latency_unit="microsecond",sources={"latencies":"MEASURED","ratios":"DERIVED","static_features":"see cost_sources"}))
    save(g/"dataset_manifest.json",dict(rows=len(dataset),valid=sum(valid(x) for x in rows),sha256=file_hash(g/"stage_a_dataset.csv"),protocol=protocol))
    why="; ".join(env["errors"]) or "Inspect per-case failures, rejections and optional capabilities."
    try:fig=figures(a,rows,why);plot_error=None
    except Exception as ex:fig=[];plot_error=repr(ex)
    ok=[x for x in rows if valid(x)];by=collections.Counter((x["workload"],x["strategy"]) for x in ok)
    from .matrix import SUPPORT
    missing=[w+"/"+s for w,ss in SUPPORT.items() for s in ss if not by[w,s]]
    from .all_pairs import planned_pairs
    pairs=planned_pairs(rows)
    table(d/"workload_strategy_8x5.csv",pairs)
    cases=[]
    from .compiler import uses_joint_backend
    for r in rows:
        applied=valid(r)
        fallback=r.get("fallback",{}).get("status")=="MEASURED"
        timing=r["statistics"] if applied else r["fallback"]["statistics"] if fallback else {}
        cases.append(dict(workload=r["workload"],strategy=r["strategy"],anchor_id=r.get("anchor_id"),case_id=r["id"],
            input_seed=r.get("input_seed"),params=r["params"],status=r["status"],reason=r.get("reason",""),
            lowering="joint_single" if uses_joint_backend(r) else "stage_a_native",
            executed=r["strategy"] if applied else "none" if fallback else "UNAVAILABLE",
            B0_us=timing.get("B0",{}).get("median"),B1_us=timing.get("B1",{}).get("median"),
            applied_us=timing.get("B2",{}).get("median"),fallback_us=timing.get("fallback",{}).get("median"),
            B0_over_B2=r.get("total_speedup") if applied else None,B1_over_B2=r.get("incremental_speedup") if applied else None))
    table(d/"workload_strategy_cases.csv",cases,["case_id","anchor_id","workload","strategy","status"])
    fb_rows=[dict(case_id=r["id"],anchor_id=r["anchor_id"],workload=r["workload"],requested=r["strategy"],executed="none",
                  admission=r["status"],reason=r["reason"],variant=k,batch=i,latency_us=t)
             for r in rows if r.get("fallback",{}).get("status")=="MEASURED"
             for k,ts in r["fallback"]["raw_us"].items() for i,t in enumerate(ts)]
    table(p/"fallback_timings.csv",fb_rows,["case_id","anchor_id","requested","executed","admission","variant","batch","latency_us"])
    from .compiler import plan
    from .util import Rejected
    applicable=set()
    for r in rows:
        try:plan(r,True)
        except Rejected:continue
        applicable.add((r["workload"],r["strategy"]))
    missing=[w+"/"+s for w,s in sorted(applicable) if not by[w,s]]
    integrity=all(valid(x) for x in rows if x["status"]=="PASS")
    failed=[x for x in rows if x["status"] in ("FAIL_CORRECTNESS","RUNTIME_ERROR","COMPILE_ERROR","OOM","TIMEOUT","BLOCKED")]
    micro_failed=any(row["status"] in ("FAILED","TIMEOUT","BLOCKED") for row in micro)
    gate=dict(env_ok=not env["errors"],qwen_assets=assets["qwen_status"]=="READY",cpu_checks=bool(cpu) and all(x["passed"] for x in cpu),
              all_workload_strategy_pairs=not missing,all_40_requested=all(q["requested"] for q in pairs),
              all_requested_accounted=all(q["requested"]==q["measured_applied"]+q["measured_fallback"] for q in pairs),valid_raw_and_artifacts=bool(ok) and integrity,
              core_failures_absent=not failed and not micro_failed,plots=len(fig)==8 and all(x["actual_points"] for x in fig),
              microbench=not micro_failed and all(any(x["family"]==h and x.get("status") in ("MEASURED","COMPILER_REPORTED") for x in micro) for h in ["H"+str(j) for j in range(1,7)]),
              formal_scope=protocol["mode"]=="full" and not protocol["filtered"])
    counts_s=collections.Counter(r["strategy"] for r in ok)
    lower=dict(S1=50,S2=80,S3=60,S4=60,S5=60)
    gate.update(venv_selected=bool(env.get("venv")) and str(env.get("prefix"))==str(env.get("venv")),
                torch_detected=env.get("torch") not in (None,"NOT_AVAILABLE"),
                triton_api=matches_triton(env.get("triton")) and all(env.get("language_api",{}).values()) and bool(env.get("language_api")),
                sm120_kernel=env.get("sm120_probe") is True,all_workloads_registered={r["workload"] for r in rows}==set(SUPPORT),
                strategy_implementation=bool(ok) and all(any(r["strategy"]==s and r["effect"] for r in ok) for s in lower),
                negative_controls=all(any(x["strategy"]==s and x["expected_reject"] and x["passed"] for x in cpu) for s in lower),
                baseline_pairs=all(len(r["raw"]["B0"])==len(r["raw"]["B1"])==len(r["raw"]["B2"])==protocol["batches"] for r in ok),
                case_identity=len(set(r["id"] for r in rows))==len(rows),
                cost_and_resources=bool(ok) and all(all(r["metadata"][m].get(k) is not None for m in ("B1","B2") for k in ("registers","smem")) and bool(r["cost"]) for r in ok),
                qualified_range=all(counts_s[s]>=n for s,n in lower.items()),
                qwen_provenance=all(assets["weights"]!="REAL_WEIGHTS" or r.get("weights_loaded") for r in ok if r["workload"] in ("W7","W8")),
                failure_reasons=all(bool(r.get("reason")) for r in rows if r["status"]!="PASS"),
                driver_and_clocks=bool(ok) and all(env.get(k) and not env[k].startswith("UNAVAILABLE") for k in ("driver_supported_cuda","driver_version"))
                    and all(not r[key].startswith("UNAVAILABLE") for r in ok for key in ("gpu_before","gpu_after")),
                verified_handoffs=all(any(r["family"]=="H3" and r["name"]==name and r["status"]=="MEASURED" for r in micro) for name in ("thread","lane","warp")),
                no_counter_profiler="ncu" in env.get("disabled_executables",[]) and "nv-nsight-cu-cli" in env.get("disabled_executables",[]))
    verdict="PASS" if all(gate.values()) else "FAIL" if env["errors"] or not integrity or any(x["status"]=="FAIL_CORRECTNESS" for x in rows) else "PARTIAL"
    result=dict(verdict=verdict,gates=gate,status_counts=counts,missing_pairs=missing,environment_errors=env["errors"],plot_error=plot_error,figures=fig,
                coverage={w:{s:by[w,s] for s in ("S1","S2","S3","S4","S5")} for w in SUPPORT},planned=len(rows),eligible_measured=len(ok))
    result["all_pairs"]=pairs
    result["measured_fallback"]=sum(x["measured_fallback"] for x in pairs)
    save(d/"acceptance_report.json",result)
    summary=["# Stage A","",verdict,"",f"{len(ok)}/{len(rows)} configurations have paired, qualified GPU measurements.","",str(counts),"",why,"","See acceptance_report.json for all gates and uncovered workload/strategy pairs."]
    associations=[]
    for strat in ("S1","S2","S3","S4","S5"):
        for workload in SUPPORT:
            group=[r for r in ok if r["strategy"]==strat and r["workload"]==workload]
            for key in ("registers","smem","reconstruction_count","reduction_slices","workers","dag_depth","logical_io_lower_bound_bytes"):
                pairs=[(v,r["statistics"]["B2"]["median"]) for r in group if isinstance(v:=r["cost"].get(key),(int,float))]
                if len(pairs)<3:continue
                xs,ys=zip(*pairs);dx=[v-statistics.mean(xs) for v in xs];dy=[v-statistics.mean(ys) for v in ys]
                norm=math.sqrt(sum(x*x for x in dx)*sum(y*y for y in dy))
                if norm:associations.append(dict(workload=workload,strategy=strat,feature=key,n=len(pairs),pearson_r=sum(x*y for x,y in zip(dx,dy))/norm))
    table(a/"feature_correlations.csv",associations,["workload","strategy","feature","n","pearson_r"])
    for s in ("S1","S2","S3","S4","S5"):
        rs=[x for x in ok if x["strategy"]==s]
        body=[f"# {s}","",f"Eligible measured configurations: {len(rs)}"]
        if rs:
            gains=[x["incremental_speedup"] for x in rs];worst=min(rs,key=lambda r:r["incremental_speedup"])
            faster=sum(r["paired_speedup_ci95"][0]>1 for r in rs);slower=sum(r["paired_speedup_ci95"][1]<1 for r in rs)
            body+=["",f"Geometric mean B1/B2 {statistics.geometric_mean(gains):.4f}. Regression rate {sum(x<1 for x in gains)/len(gains):.2%}.",
                   f"Paired within-run intervals above one: {faster}; below one: {slower}; crossing one: {len(rs)-faster-slower}.",
                   f"Worst case {worst['id']} at {worst['incremental_speedup']:.4f}x.","","|Parameter stratum|Count|Median B1/B2|Regressions|","|---|---:|---:|---:|"]
            key={"S1":"layout","S2":"merge","S3":"consumers","S4":"cut","S5":"workers"}[s]
            strata=collections.defaultdict(list)
            for x in rs:strata[(x["workload"],str(x["params"].get(key,"default")))].append(x["incremental_speedup"])
            for (w,v),vals in sorted(strata.items()):body.append(f"|{w} {key}={v}|{len(vals)}|{statistics.median(vals):.4f}|{sum(z<1 for z in vals)}|")
        (d/(s.lower()+"_report.md")).write_text("\n".join(body),encoding="utf8")
    (d/"stage_a_report.md").write_text("\n".join(summary),encoding="utf8")
    (d/"hardware_cost_report.md").write_text("# A0 and A3\n\n"+str(collections.Counter(x["status"] for x in micro))+"\n\nSee microbench.json for measurements. Occupancy is theoretical. Missing counters are UNAVAILABLE.",encoding="utf8")
    (d/"failures_and_limitations.md").write_text("# Limitations\n\n"+why+"\n\n"+"\n".join(f"- {r['id']} {r['status']} {r.get('reason','')}" for r in rows if not valid(r)),encoding="utf8")
    return result
