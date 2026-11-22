import collections,json,math
from statistics import median
from stage_a.util import file_hash,save,table
from .report import Dirs,COLUMNS,UN,NA,dump_lines

FIGS=("B_F1_pairwise_speedup","B_F2_order_sensitivity","B_F3_multistep_speedup","B_F4_interaction_heatmap",
      "B_F5_search_depth_width","B_F6_search_vs_oracle","B_F7_legality_resource","B_F8_qwen_subgraph")
def group(rows,key,value):
    result=collections.defaultdict(list)
    for r in rows:
        v=value(r)
        if isinstance(v,(int,float)) and math.isfinite(v):result[key(r)].append(v)
    return result

def figures(store,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names=[]
    for i,stem in enumerate(FIGS):
        fig,ax=plt.subplots(figsize=(9.2,4.9),layout="constrained")
        rows=[];points=0
        if i in (0,2,5,7):
            if i==0:
                rows=store.pairs;lookup=lambda r: "+".join(x["strategy"] for x in r["sequence"]);val="combined_speedup";ylabel="B1 / combination"
            elif i==2:
                rows=store.multi;lookup=lambda r:str(r["depth"])+" actions";val="best_double_improvement";ylabel="Best double / plan"
            elif i==5:
                rows=store.decisions;lookup=lambda r:r["method"];val="regret";ylabel="Regret vs finite measured oracle"
            else:
                rows=[r for r in store.data if r["workload"] in ("W7","W8")];lookup=lambda r:r["workload"]+" d"+str(r["sequence_depth"]);val="total_speedup";ylabel="PyTorch B0 / plan"
            # Collapse commuting histories to a single vote within each group.
            unique={(r.get("region"),lookup(r),r.get("plan",r.get("candidate_id",str(idx)))):r for idx,r in enumerate(rows)}
            groups=group(unique.values(),lookup,lambda r:r.get(val))
            if groups:
                keys=sorted(groups);ax.boxplot([groups[k] for k in keys]);ax.set_xticks(range(1,len(keys)+1),keys,rotation=35)
                ax.axhline(0 if i==5 else 1,color=".5",ls="--");points=sum(map(len,groups.values()))
            ax.set_ylabel(ylabel)
        elif i==1:
            for category in ("SAME_PLAN","MEASURED_DISTINCT"):
                pairs=[r for r in store.orders if r["status"]==category];points+=len(pairs)
                if pairs:ax.scatter([r["forward_us"] for r in pairs],[r["reverse_us"] for r in pairs],label=category,s=14,alpha=.55)
            if points:
                ax.set_xscale("log");ax.set_yscale("log");ax.legend()
                a,b=ax.get_xlim();ax.plot([a,b],[a,b],"k--",lw=.7)
            ax.set_xlabel("Forward latency (us)");ax.set_ylabel("Reverse latency (us)")
        elif i==3:
            names_s=["S"+str(j) for j in range(1,6)];heat=np.full((5,5),np.nan)
            groups=group(store.pairs,lambda r:tuple(x["strategy"] for x in r["sequence"]),lambda r:r["interaction_score"])
            for (a,b),xs in groups.items():heat[names_s.index(a),names_s.index(b)]=median(xs);points+=len(xs)
            if points:
                bound=max(.01,float(np.nanmax(np.abs(heat))))
                im=ax.imshow(heat,cmap="RdBu",vmin=-bound,vmax=bound);fig.colorbar(im,ax=ax,label="Median log interaction")
                ax.set_xticks(range(5),names_s);ax.set_yticks(range(5),names_s)
                for i0,j in zip(*np.where(np.isfinite(heat))):ax.text(j,i0,f"{heat[i0,j]:.2f}",ha="center",va="center")
            ax.set_xlabel("Second action");ax.set_ylabel("First action")
        elif i==4:
            ordered=group([r for r in store.decisions if r["method"]=="beam" and not r.get("ablation")],
                          lambda r:(r["budget"],r["width"],r["max_depth"]),lambda r:r["regret"])
            for budget,width in sorted({(b,w) for b,w,_ in ordered}):
                line=sorted((d,median(v)) for (b,w,d),v in ordered.items() if (b,w)==(budget,width))
                ax.plot(*zip(*line),marker="o",label=f"budget {budget}, width {width}");points+=len(line)
            if points:ax.legend(fontsize=7)
            ax.set_xlabel("Maximum actions");ax.set_ylabel("Median finite-pool regret")
        else:
            groups=group(store.data,lambda r:r["workload"],lambda r:r.get("combined_speedup"))
            for w in groups:
                data=[r for r in store.data if r["workload"]==w and isinstance(r.get("shared_memory_delta"),int) and r["status"]=="PASS"]
                if data:ax.scatter([r["shared_memory_delta"]/1024 for r in data],[r["combined_speedup"] for r in data],s=12,label=w,alpha=.45);points+=len(data)
            if points:ax.legend(fontsize=7);ax.set_yscale("log")
            ax.set_xlabel("Compiled shared-memory change (KiB/CTA)");ax.set_ylabel("B1 / qualified plan")
        if not points:
            ax.set_axis_off();ax.text(.5,.5,"No eligible measurements. See failures_and_limitations.md",transform=ax.transAxes,ha="center",wrap=True)
        ax.set_title(stem.replace("_"," "));fig.savefig(folder/(stem+".png"),dpi=145);plt.close(fig)
        names.append(dict(figure=stem,points=points))
    return names

def finish(s,env,assets,quarantine):
    e,l,p,m,se,k,ir,a,r,d,log=[s.out/name for name in Dirs]
    for n,folder in [("sequence_legality",l),("proof_revalidation",l),("transformation_sequences",m),("plan_differences",m)]:
        with (folder/(n+".jsonl")).open("wb") as out:
            for f in sorted((folder/"parts").glob("*_"+n+".jsonl")):out.write(f.read_bytes())
    table(l/"compatibility_matrix.csv",s.compat,["region","pair","status","reason"]);table(l/"rejected_candidates.csv",s.rejected,["region","status","reason"])
    table(p/"pairwise_results.csv",s.pairs,["region","sequence","status","latency_us","controls"])
    table(p/"order_sensitivity.csv",s.orders,["region","actions","forward","reverse","status"])
    table(p/"interaction_scores.csv",[{k:x[k] for k in ("region","sequence","plan","baseline_us","single_i_us","single_j_us","latency_us","interaction_score")} for x in s.pairs],["region","interaction_score"])
    targets=[]
    for reg,w in s.regions.items():
        for label,rule,work in [("T1",["S1","S2","S3"],("W3","W7")),("T2",["S3","S4","S5"],("W5","W8")),("T3",["S2","S4","S5"],("W8",))]:
            rows=[x for x in s.multi if x["region"]==reg and sorted(y["strategy"] for y in x["sequence"])==rule]
            separate=label=="T3"
            status="NOT_APPLICABLE" if separate or w not in work or not rows else "MEASURED" if any(x["status"]=="PASS" for x in rows) else "BLOCKED_PREFIX"
            reason="PV slices live in one task; no independently published slice-reduction DAG" if separate else "" if rows else "no submitting three-action path at these sites"
            targets.append(dict(region=reg,workload=w,target=label,status=status,related_intra_task_histories=len(rows),reason=reason))
    table(m/"target_coverage.csv",targets,["region","workload","target","status","reason"])
    table(m/"multistep_results.csv",s.multi,["region","sequence","status","depth","latency_us"])
    table(p/"stage_a_single_remeasurements.csv",s.references,["region","source_case","status","latency_us"])
    sr=[{k:v for k,v in x.items() if k!="trace"} for x in s.decisions]
    for method in ("greedy","beam"):table(se/(method+"_results.csv"),[x for x in sr if x["method"]==method],["region","method","selected_latency","regret"])
    table(se/"oracle_results.csv",s.oracles,["region","scope","plans","measured","oracle_latency"])
    table(se/"regret.csv",sr,["region","method","regret"]);table(se/"search_budget_report.csv",sr,["region","method","budget","evaluated_candidates","compile_time","search_time"])
    dump_lines(se/"search_trace.jsonl",s.decisions)
    table(se/"ablations.csv",s.ablations,["region","ablation","best_latency"])
    table(k/"combined_cost_features.csv",s.cost,["region","plan","latency_us"])
    for name,column in (("resource","resource_delta"),("ownership","ownership_delta"),("dag","ddg_delta")):
        table(k/(name+"_deltas.csv"),[dict(region=x["region"],plan=x["candidate_id"],delta=x[column]) for x in s.data],["region","plan","delta"])
    schema=list(dict.fromkeys(s.audit["stage_a_dataset_schema"]+COLUMNS+[key for row in s.data for key in row]))
    table(d/"stage_b_dataset.csv",[{key:x.get(key,UN) for key in schema} for x in s.data],schema)
    save(d/"stage_b_dataset_schema.json",dict(columns=schema,missing=UN,inapplicable=NA,time_unit="us; search and compilation use seconds",
               label="gpu_latency_us; only PASS + MEASURED",group_holdout=["stage_a_run_id","workload","functional shape"],feature_provenance="cost_sources and data_source"))
    save(d/"dataset_manifest.json",dict(rows=len(s.data),sha256=file_hash(d/"stage_b_dataset.csv"),protocol=s.proto))
    try:plots=figures(s,a);plot_error=None
    except Exception as exc:plots=[];plot_error=repr(exc)
    good=[x for x in s.data if x["status"]=="PASS"];pair=[x for x in s.pairs if x["status"]=="PASS"]
    triple=[x for x in s.multi if x["status"]=="PASS" and x["depth"]==3]
    pair_types={tuple(sorted(v["strategy"] for v in x["sequence"])) for x in pair}
    trip_types={tuple(sorted(v["strategy"] for v in x["sequence"])) for x in triple}
    opts=s.proto["options"]
    fairness=all(x["evaluated_candidates"]<=x["budget"] and len(x["trace"])==x["evaluated_candidates"] for x in s.decisions)
    fair_groups=collections.defaultdict(set)
    for x in s.decisions:
        if x["method"] in ("beam","greedy") and "ablation" not in x:fair_groups[x["region"],x["max_depth"],x["budget"]].add(x["method"])
    comparisons=sum(x["status"]=="MEASURED_DISTINCT" for x in s.orders);aliases=sum(x["status"]=="SAME_PLAN" for x in s.orders)
    gates=dict(import_trusted=s.audit["trusted"]>0 and not s.audit["issues"],environment=not env["errors"] and not quarantine,
        qwen_assets=assets.get("qwen_status")=="READY",all_eight_workloads=s.tested_groups=={"W"+str(i) for i in range(1,9)},
        isolated_rules={v for x in good for v in x["strategy_sequence"]}=={"S1","S2","S3","S4","S5"},
        revalidation=bool(good) and all(x["proof_revalidation_count"]>=x["sequence_depth"] for x in good),
        six_target_pairs=all(x in pair_types for x in [("S1","S2"),("S2","S3"),("S3","S4"),("S4","S5"),("S1","S3"),("S2","S5")]),
        three_actions=all(x in trip_types for x in [("S1","S2","S3"),("S3","S4","S5")]),
        distinct_order_measurements=comparisons>0 and aliases>0,searches=all(any(x["method"]==method and x["selected"] for x in s.decisions) for method in ("none","single","fixed","greedy","beam")),
        finite_oracles=bool(s.oracles) and all(x["measured"]>0 for x in s.oracles),
        budget=bool(fair_groups) and fairness and all(len(v)==2 for v in fair_groups.values()),
        ablations={"fixed_order","fixed_param","guided","dag"}<={x["ablation"] for x in s.ablations},
        qwen_subgraphs={"W7","W8"}<=s.tested_groups,raw_and_cost=bool(good) and all(isinstance(x.get("register_delta"),int) for x in good),
        regenerated_singles=bool(s.references) and all(x["status"]=="PASS" for x in s.references),
        plots=len(plots)==8 and all(x["points"] for x in plots),dataset=len(s.data)==s.node_count and bool(s.data),
        formal=opts["mode"]=="full" and not opts.get("only_workload") and not opts.get("only_combination") and opts["max_depth"]>=3,
        core_execution=not any(x.get("status") in ("FAIL_CORRECTNESS","COMPILE_ERROR","RUNTIME_ERROR","TIMEOUT","BLOCKED","OOM") for x in s.failures))
    verdict="PASS" if all(gates.values()) else "FAIL" if not gates["environment"] or not gates["import_trusted"] or any(x.get("status")=="FAIL_CORRECTNESS" for x in s.failures) else "PARTIAL"
    smoke=bool(pair and triple and fair_groups and len(plots)==8) and gates["environment"] and gates["core_execution"]
    result=dict(verdict=verdict,smoke_complete=smoke,gates=gates,failed_gates=[k for k,v in gates.items() if not v],statuses=dict(s.statuses),
                qualified=len(good),order_distinct=comparisons,order_aliases=aliases,figures=plots,plot_error=plot_error,quarantine=quarantine,
                stages={"B0":gates["import_trusted"] and gates["regenerated_singles"],"B1":bool(s.compat) and gates["revalidation"],
                        "B2":gates["six_target_pairs"],"B3":gates["three_actions"],"B4":gates["searches"] and gates["budget"],"B5":gates["ablations"] and gates["qwen_subgraphs"]})
    save(r/"acceptance_report.json",result)
    failures=[f"{x.get('region')} {x.get('plan','')} {x['status']} {x.get('reason','')}" for x in s.failures]
    ranked=sorted((x for x in good if x["sequence_depth"]>=2),key=lambda x:x["combined_speedup"])
    extrema=[f"{tag} {x['region']} {x['candidate_id'][:12]} {x['strategy_sequence']} | B1 {x['baseline_latency_us']:.4g} us | plan {x['combined_latency']:.4g} us | {x['combined_speedup']:.4g}x" for tag,x in ([("Slowest relative to B1",ranked[0]),("Fastest relative to B1",ranked[-1])] if ranked else [])]
    policy_summary=[]
    for method in sorted({x["method"] for x in s.decisions}):
        vals=[x for x in s.decisions if x["method"]==method and not x.get("ablation") and isinstance(x["regret"],(int,float))]
        if vals:policy_summary.append(f"{method}: {len(vals)} decisions across protocol cells, median regret {median(x['regret'] for x in vals):.2%}, median paid evaluations {median(x['evaluated_candidates'] for x in vals):g}.")
    for name,body in {
        "stage_b_report":[f"Verdict **{verdict}**. Qualified {len(good)} of {len(s.data)} unique canonical plans.",f"Open gates: {result['failed_gates']}.",f"Stages {result['stages']}.",f"Joint plans slower than paired B1: {sum(x['combined_speedup']<1 for x in ranked)} / {len(ranked)}.",*extrema],
        "compatibility_report":[f"{len(s.compat)} region/pair applicability rows. Static rejects {len(s.rejected)}.",f"Measured distinct order comparisons {comparisons}. Deduplicated comparisons {aliases}.",
                                "W5 has no published scheduler tasks. W8 groups real head tasks and slices PV inside each group. Cross-CTA split-reduction tasks: NOT_APPLICABLE."],
        "pairwise_report":[f"{len(pair)} qualified ordered pair histories (including equivalent orders).",
            str(collections.Counter("positive" if x["interaction_score"]>0 else "negative" if x["interaction_score"]<0 else "zero" for x in pair if isinstance(x["interaction_score"],(float,int)))),
            "See interaction_scores.csv for same-input measured constituent times. A derived site without an isolated counterpart has no interaction score."],
        "multistep_report":[f"{len(triple)} qualified three-action histories.",f"Strategy triplets {sorted(trip_types)}.","Detailed ordered prefix artifacts in transformation_sequences and the region plan directories."],
        "search_report":[f"{len(s.decisions)} independent policy decisions. Oracle scope: each predeclared regional pool.",f"Budget accounting: {fairness}. Oracle results are released after selection.","Use charged_search_seconds for cache-independent comparisons and search_time for observed wall time.",*policy_summary],
        "ablation_report":["ablations.csv and regret.csv give depth, fixed order, parameter, structural pruning, greedy/beam and DAG-priority contrasts. Compare within region and matched budgets.",
                           "The final legality and numerical gates remain active in the unguided arm."],
        "failures_and_limitations":[str(quarantine or ""),str(plot_error or ""),*failures]}.items():
        (r/(name+".md")).write_text("# "+name.replace("_"," ")+"\n\n"+"\n\n".join(body),encoding="utf8")
    return result
