import csv
import itertools
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median
from stage_a.util import save,table,file_hash,digest
from .search import finite_oracle

Dirs=("00_environment","01_legality","02_pairwise","03_multistep","04_search","05_cost","06_ir","07_analysis","08_reports","09_dataset","10_logs")
NA="NOT_APPLICABLE"
UN="UNAVAILABLE"
COLUMNS=("stage_a_run_id stage_a_case_id strategy_sequence transformation_locations strategy_parameters sequence_depth "
         "initial_plan_hash intermediate_plan_hashes final_plan_hash legality_status proof_revalidation_count ownership_delta ddg_delta "
         "resource_delta register_delta shared_memory_delta estimated_memory_delta search_method search_budget evaluated_candidates "
         "compiled_candidates unique_plans search_time compile_time selected_latency oracle_latency regret single_strategy_latency "
         "combined_latency combined_speedup best_single_improvement interaction_score correctness_status measurement_status data_source").split()
ALL_PAIRS=list(itertools.combinations(("S1","S2","S3","S4","S5"),2))
def dump_lines(path,rows):
    with path.open("w",encoding="utf8") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")

def safe(record):
    return record and record.get("status")=="PASS" and isinstance(record.get("latency_us"),(int,float)) and 0<record["latency_us"]<float("inf")

def difference(a,b):
    return {k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}

def interaction_us(baseline,left,right,joint):
    times=(baseline,left,right,joint)
    if not all(isinstance(t,(int,float)) and math.isfinite(t) and t>0 for t in times):
        return UN
    return math.log(left/baseline)+math.log(right/joint)

def packed(a):return [x.json() for x in a]

class Results:
    def __init__(self,out,audit,protocol):
        self.out=out;self.audit=audit;self.proto=protocol
        self.data=[];self.cost=[];self.pairs=[];self.multi=[];self.orders=[];self.compat=[];self.decisions=[]
        self.oracles=[];self.ablations=[];self.failures=[];self.rejected=[];self.node_count=0
        self.regions={};self.tested_groups=set();self.statuses=Counter();self.references=[]

    def unavailable(self,c,only):
        self.failures.append(dict(region=c["id"],status="NOT_APPLICABLE",reason="no shared executable workload sites: "+"+".join(only)))

    def structure(self,pool):
        region=pool.case["id"];out=self.out
        self.regions[region]=pool.case["workload"]
        base=out/Dirs[6]/region/"plans";base.mkdir(parents=True,exist_ok=True)
        for h,plan in pool.plans.items():save(base/(h+".json"),plan.json())
        self.node_count+=len(pool.nodes)
        admitted=[];proof=[];diff=[]
        for seq,parent,child in pool.paths:
            before,after=pool.plans[parent],pool.plans[child]
            rec=dict(region=region,sequence=packed(seq),before=parent,after=child,status="STATIC_LEGAL",depth=len(seq),
                     input_ir=f"06_ir/{region}/plans/{parent}.json",output_ir=f"06_ir/{region}/plans/{child}.json")
            admitted.append(rec)
            proof.append(dict(**rec,revalidated=[dict(action=a.json(),proof=after.proof[a.strategy]) for a in seq],
                              runtime_admission="see prefix_artifacts and numerical checks under 10_logs/records"))
            diff.append(dict(region=region,before=parent,after=child,ownership=difference(before.ownership,after.ownership),
                             dependency_fields=list(difference(before.ddg,after.ddg)),lifetime_changed=before.buffers!=after.buffers))
        for path,parent,a,status,why in pool.rejects:
            row=dict(region=region,sequence=packed(path+(a,)),before=parent,status=status,reason=why)
            admitted.append(row);self.rejected.append(row)
        for d,name,rows in [(1,"sequence_legality",admitted),(1,"proof_revalidation",proof),(3,"plan_differences",diff),(3,"transformation_sequences",admitted)]:
            folder=out/Dirs[d]/"parts";folder.mkdir(exist_ok=True)
            dump_lines(folder/(region+"_"+name+".jsonl"),rows)
        structural=pool.pairs()
        for a,b in ALL_PAIRS:
            seq=[s for s,_,_ in pool.paths if len(s)==2 and {x.strategy for x in s}=={a,b}]
            self.compat.append(dict(region=region,workload=pool.case["workload"],pair=a+"+"+b,
                  forward=sum(s[0].strategy==a for s in seq),reverse=sum(s[0].strategy==b for s in seq),
                  status="STATIC_LEGAL" if (a,b) in structural else "NOT_APPLICABLE",
                  reason="" if seq else "no legal two-step path in the scoped sites, parameters and active extent"))

    def blocked(self,pool,root):
        r={h:(root if h==pool.root else dict(status="BLOCKED",reason="root gate failed: "+str(root.get("reason")))) for h in pool.nodes}
        self.region(pool,r,[],[]);self.flush_partial()

    def region(self,pool,records,refs,policies):
        c=pool.case;region=c["id"];w=c["workload"];root=records[pool.root]
        start=dict(region=region,workload=w,stage_a_run_id=c["stage_a_run_id"],stage_a_case_id=region)
        for seed,r in refs:
            self.references.append(dict(**start,source_case=seed["id"],strategy=seed["strategy"],params=seed["params"],
                                        status=r["status"],reason=r.get("reason",""),latency_us=r.get("latency_us",UN),
                                        historical_us=seed["latency"],origin="REGENERATED",input_seed=c["input_seed"]))
        sing=[r["latency_us"] for _,r in refs if safe(r)]
        bestref=min(sing) if sing else None
        unique_single={h for seq,_,h in pool.paths if len(seq)==1}
        single_pool=[records[h]["latency_us"] for h in unique_single if safe(records[h])]
        double_pool=[records[h]["latency_us"] for h in {child for seq,_,child in pool.paths if len(seq)==2} if safe(records[h])]
        oracle=finite_oracle(pool,records);ot=records[oracle]["latency_us"] if oracle else None
        self.oracles.append(dict(**start,scope="predeclared finite verified pool",plans=len(records),measured=sum(bool(safe(r)) for r in records.values()),
                                 selected=oracle,oracle_latency=ot or UN,pool_digest=digest(sorted(records))))
        selected=defaultdict(list)
        for result in policies:
            h=result["selected"];t=result["selected_latency"]
            rung={**start,**{k:v for k,v in result.items() if k!="trace"},"trace":result["trace"],
                  "oracle_latency":ot or UN,"regret":(t-ot)/ot if t and ot else UN,
                  "speedup":root["latency_us"]/t if safe(root) and t else UN,
                  "measurement_success_rate":result["valid_evaluations"]/result["evaluated_candidates"] if result["evaluated_candidates"] else UN,
                  "legality_rate":(result["candidate_tests"]-result["rejected_checks"])/result["candidate_tests"] if result["candidate_tests"] else NA,
                  "finite_pool_digest":digest(sorted(records))}
            self.decisions.append(rung)
            if h:selected[h].append(rung)
        byhash={}
        for h,seq in pool.nodes.items():
            r=records[h];plan=pool.plans[h];before=pool.plans[pool.root];good=safe(r)
            self.statuses[r["status"]]+=1
            latency=r["latency_us"] if good else None
            us_base=r["statistics"]["B1"]["median"] if good else None
            modes=[a.strategy for a in seq]
            chain=[pool.hash_path(seq[:i]) for i in range(len(seq)+1)]
            row=dict(**start,case_id=region,candidate_id=h,params=c["params"],shape_and_tile=c["params"],dtype=c["params"]["dtype"],strategy=modes or ["none"],
                     status=r["status"],reason=r.get("reason",""),baseline="current matched Megakernel B1",
                     strategy_sequence=modes,transformation_locations=[a.location for a in seq],strategy_parameters=[a.parameter for a in seq],
                     sequence_depth=len(seq),initial_plan_hash=pool.root,intermediate_plan_hashes=chain[1:-1],final_plan_hash=h,
                     legality_status="STATIC_LEGAL",proof_revalidation_count=sum(range(1,len(seq)+1)),
                     ownership_delta=difference(before.ownership,plan.ownership),ddg_delta=list(difference(before.ddg,plan.ddg)),
                     buffer_delta=[before.buffers,plan.buffers] if before.buffers!=plan.buffers else NA,
                     correctness_status="PASS" if good else r["status"],measurement_status="MEASURED" if good else r["status"],
                     matched_stage_a_singles=len(refs),qualified_stage_a_singles=len(sing),single_strategy_latency=bestref or UN,best_pool_single_us=min(single_pool) if single_pool else UN,
                     combined_latency=latency or UN,combined_speedup=us_base/latency if good else UN,
                     best_single_improvement=bestref/latency if bestref and good else UN,
                     best_double_improvement=min(double_pool)/latency if double_pool and good and len(seq)>=3 else NA,
                     absolute_saving_us=us_base-latency if good else UN,interaction_score=NA,oracle_latency=ot or UN,
                     data_source="MEASURED" if good else UN,
                     compile_time=r.get("compile_seconds",UN),compile_api_seconds=r.get("compile_api_seconds",UN),
                     unique_plans=len(records),search_method=[x["method"] for x in selected[h]],selection_records=[{k:s[k] for k in
                     ("method","budget","width","max_depth","selected_latency","regret")} for s in selected[h]],
                     resource_delta=UN,estimated_memory_delta=UN,environment_id=self.proto["signature"],input_seed=c["input_seed"],
                     model_weights=self.proto["assets"]["weights"] if w in ("W7","W8") else NA)
            row["correctness"]=row["correctness_status"];row["legality"]=row["legality_status"]
            if selected[h]:
                for dest,src in (("search_budget","budget"),("evaluated_candidates","evaluated_candidates"),("compiled_candidates","compiled_candidates"),
                                  ("search_time","search_time"),("selected_latency","selected_latency"),("regret","regret")):
                    row[dest]=[s[src] for s in selected[h]]
            if good:
                metas=r["prefix_artifacts"];b=metas[0]["meta"];t=metas[-1]["meta"]
                delta={k:t[k]-b[k] if isinstance(t.get(k),(int,float)) and isinstance(b.get(k),(int,float)) else UN
                       for k in ("registers","smem","spill","ctas","theoretical_occupancy")}
                row.update(resource_delta=delta,register_delta=delta["registers"],shared_memory_delta=delta["smem"],
                           register_delta_source="DERIVED",gpu_latency_us=latency,baseline_latency_us=us_base,
                           incremental_speedup=us_base/latency,total_speedup=r["statistics"]["B0"]["median"]/latency,
                           paired_speedup_ci95=r["paired_speedup_ci95"],batches=len(r["raw_us"]["B0"]),std_us=r["statistics"].get("candidate",r["statistics"]["B1"])["std"],
                           cost_sources={},cost_by_strategy=r["cost_by_strategy"])
                for k in r["cost_by_strategy"].values():
                    for f,v in k["values"].items():
                        if v not in (NA,UN) or f not in row:
                            row[f]=v;row["cost_sources"][f]=k["sources"][f]
                for f in ("instrumented_critical_path_ns","instrumented_cross_worker_edges","ready_gap_median_ns"):
                    if f in r:
                        row[f]=r[f];row["cost_sources"][f]="DERIVED" if isinstance(r[f],(int,float)) else r[f]
                if "S3" in modes and isinstance(row.get("stats_bytes"),int) and isinstance(row.get("baseline_intermediate_write_bytes"),int):
                    published=0 if w=="W3" and plan.local else row["stats_bytes"]
                    row.update(published_statistics_bytes=published,estimated_memory_delta=published-row["baseline_intermediate_write_bytes"],
                               estimated_memory_delta_scope="S3 logical statistics vs replaced intermediate writes only")
                    row["cost_sources"]["published_statistics_bytes"]="STATIC_ESTIMATE"
                self.cost.append(dict(**start,plan=h,**delta,latency_us=latency,source="DERIVED from compiler / CUDA API",cost_by_strategy=r["cost_by_strategy"]))
                self.tested_groups.add(w)
            else:self.failures.append(dict(**start,plan=h,status=r["status"],reason=r.get("reason","")))
            byhash[h]=row;self.data.append(row)
        groups=defaultdict(list)
        for seq,parent,h in pool.paths:
            if len(seq)<2:continue
            ckeys=[pool.hash_path(seq[:i]) for i in range(len(seq)+1)]
            qualified=all(safe(records[k]) for k in ckeys)
            row=dict(**start,sequence=packed(seq),plan=h,prefix_hashes=ckeys,depth=len(seq),
                     status="PASS" if qualified else "BLOCKED_PREFIX",latency_us=records[h].get("latency_us",UN) if qualified else UN,
                     evidence=[f"06_ir/{region}/{k}/prefix_{len(pool.nodes[k])}" for k in ckeys],
                     combined_speedup=byhash[h]["combined_speedup"] if qualified else UN,best_single_improvement=byhash[h]["best_single_improvement"] if qualified else UN)
            if len(seq)==2:
                left=ckeys[1]
                try:right=pool.hash_path((seq[1],))
                except ValueError:right=None
                compatible=qualified and right in records and safe(records[right])
                groupkey=tuple(sorted(seq))
                controls=dict(C0=pool.root,C1=left,C2=right,C3=h)
                row.update(controls=controls,single_i_us=records[left].get("latency_us",UN),
                    single_j_us=records.get(right,{}).get("latency_us",UN),baseline_us=root.get("latency_us",UN),
                    absolute_saving_us=byhash[h]["absolute_saving_us"],interaction_score=interaction_us(root["latency_us"],records[left]["latency_us"],records[right]["latency_us"],records[h]["latency_us"]) if compatible else UN)
                groups[groupkey].append(row);self.pairs.append(row)
                if compatible and pool.nodes[h]==seq:byhash[h]["interaction_score"]=row["interaction_score"]
            else:
                row.update(best_double_improvement=min(double_pool)/records[h]["latency_us"] if double_pool and qualified else UN,
                           canonical_depth=len(pool.nodes[h]),equivalent_shorter_plan=len(pool.nodes[h])<len(seq))
                self.multi.append(row)
        for actions,variants in groups.items():
            for row in variants:
                reverse=next((v for v in variants if v["sequence"]==row["sequence"][::-1]),None)
                row["controls"]["C4"]=reverse["plan"] if reverse else NA
                fully=reverse and reverse["status"]==row["status"]=="PASS"
                self.orders.append(dict(**start,actions=row["sequence"],forward=row["plan"],reverse=reverse["plan"] if reverse else NA,
                          status="SAME_PLAN" if fully and row["plan"]==reverse["plan"] else "MEASURED_DISTINCT" if fully else "REVERSE_UNAVAILABLE",
                          forward_us=row["latency_us"],reverse_us=reverse["latency_us"] if reverse else UN,
                          order_ratio=row["latency_us"]/reverse["latency_us"] if fully else UN))
        for d in (1,2,3,4):
            vals=[r["latency_us"] for h,r in records.items() if len(pool.nodes[h])==d and safe(r)]
            self.ablations.append(dict(**start,ablation="depth",depth=d,count=len(vals),best_latency=min(vals) if vals else UN))
        for r in self.decisions:
            if r["region"]!=region:continue
            self.ablations.append(dict(**start,ablation=r.get("ablation","policy"),method=r["method"],width=r["width"],depth=r["max_depth"],
                                       budget=r["budget"],regret=r["regret"],best_latency=r["selected_latency"],checks=r["candidate_tests"]))

    def flush_partial(self):
        save(self.out/"10_logs/regions.json",dict(regions=self.regions,states=dict(self.statuses),plans=self.node_count))

    def finish(self,env,assets,quarantine):
        from .outputs import finish
        return finish(self,env,assets,quarantine)
