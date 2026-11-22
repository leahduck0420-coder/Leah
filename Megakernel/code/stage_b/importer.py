"""Read-only Stage A import. Only fully qualified records can seed the GPU experiments."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from stage_a.all_pairs import PROTOCOL,fingerprint
from stage_a.matrix import matrix as make_matrix
from stage_a.compiler import plan
from stage_a.report import valid
from stage_a.toolchain import TRITON_VERSION, matches_triton
from stage_a.util import digest, file_hash, save, table

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "all_pairs":"06_reports/workload_strategy_8x5.csv","requests":"06_reports/workload_strategy_cases.csv","fallback":"02_performance/fallback_timings.csv",
    "dataset": "07_dataset/stage_a_dataset.csv", "schema": "07_dataset/stage_a_dataset_schema.json",
    "manifest": "07_dataset/dataset_manifest.json", "protocol": "00_environment/protocol.json",
    "environment": "00_environment/environment.json", "matrix": "00_environment/matrix.json",
    "accept": "06_reports/acceptance_report.json", "limitations": "06_reports/failures_and_limitations.md",
    "checks": "01_correctness/correctness_cases.csv", "errors": "01_correctness/numerical_errors.csv",
    "legality": "01_correctness/legality_results.jsonl", "raw": "02_performance/raw_timings.csv",
    "baseline": "02_performance/baseline_timings.csv", "optimized": "02_performance/optimized_timings.csv",
    "speedups": "02_performance/speedups.csv", "cost": "03_cost/cost_features.csv",
    "ownership": "03_cost/ownership_features.csv", "usage": "03_cost/resource_usage.csv",
    "dag": "03_cost/dag_features.csv", "metadata": "03_cost/compilation_metadata.jsonl"}
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def records(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def discover(path):
    path=Path(path).resolve()
    if (path/"07_dataset/dataset_manifest.json").is_file():return path
    candidates=list(path.rglob("dataset_manifest.json"))
    if len(candidates)!=1:raise ValueError("Select a single Stage A run. Manifests: "+str(candidates))
    return candidates[0].parent.parent

def resolve_files(path):
    override=read(path/"stage_b_import_paths.json") if (path/"stage_b_import_paths.json").is_file() else {}
    out={}
    for key,rel in REQUIRED.items():
        item=path/override.get(key,rel)
        if not item.is_file():
            candidates=list(path.rglob(Path(rel).name))
            if len(candidates)!=1:raise ValueError(f"Missing or ambiguous {key}. Supply stage_b_import_paths.json. Found {candidates}")
            item=candidates[0]
        out[key]=item
    return out

def contract(case):
    p=case["params"]
    knobs={"tile","bm","bn","bk","warps","merge","cut","workers","schedule"}
    # W1's permutation and W5's normalizer are part of the comparison contract.
    return digest([case["workload"],{k:v for k,v in p.items() if k not in knobs}])

def audit(path, out=None, *, mode="full", export_rows=False):
    path=discover(path); files=resolve_files(path)
    manifest,protocol,env,verdict=[read(files[k]) for k in ("manifest","protocol","environment","accept")]
    if protocol.get("required_triton")!=TRITON_VERSION or not matches_triton(env.get("triton")):
        raise ValueError("Stage B requires fresh Stage A measurements collected with Triton "+TRITON_VERSION)
    schema=read(files["schema"])["columns"]
    data=records(files["dataset"])
    if not data or not set(schema)<=set(data[0]) or file_hash(files["dataset"])!=manifest["sha256"]:
        raise ValueError("Stage A dataset digest/schema mismatch")
    source=fingerprint()
    issues=[]
    if protocol!=manifest["protocol"] or protocol.get("mode")!=mode or protocol.get("design")!=PROTOCOL or protocol.get("filtered"):issues.append("input does not match the declared unfiltered "+mode+" protocol")
    if protocol.get("code")!=source:issues.append("source fingerprint mismatch; core reproduction unverified")
    if len(data)!=manifest["rows"] or len({v["case_id"] for v in data})!=len(data):issues.append("dataset identity/count mismatch")
    matrix={c["id"]:c for c in read(files["matrix"])}
    expected={c["id"]:c for c in make_matrix(protocol["seed"],mode=="smoke")}
    if matrix!=expected or set(matrix)!={r["case_id"] for r in data}:
        raise ValueError("Stage A input is incomplete or does not match this all-five design and seed")
    if not (path/"06_reports/workload_strategy_8x5.csv").is_file():raise ValueError("missing Stage A 8x5 coverage table")
    run_id=protocol.get("run_id") or digest(protocol)[:16]
    raw=defaultdict(lambda:defaultdict(dict))
    for row in records(files["raw"]):
        arr=raw[row["case_id"]][row["variant"]];batch=int(row["batch"])
        if batch in arr:raise ValueError("duplicate timing batch")
        arr[batch]=float(row["latency_us"])
    meta=defaultdict(dict)
    for line in files["metadata"].read_text(encoding="utf8").splitlines():
        obj=json.loads(line);meta[obj["case_id"]][obj["variant"]]=obj
    checkrows=defaultdict(list)
    for c in records(files["checks"]):checkrows[c["case_id"]].append(c)
    indexed={key:{r["case_id"] for r in records(files[key])} for key in ("errors","baseline","optimized","speedups","cost","ownership","usage","dag")}
    legal={q['case_id']:q for q in (json.loads(line) for line in files['legality'].read_text(encoding='utf8').splitlines()) if 'case_id' in q}
    index=[];seeds=[];blocks=[];n=protocol.get("batches",0)
    for row in data:
        ident=row["case_id"];status=row["status"];case=matrix.get(ident);why=[]
        if case is None or case["params"]!=json.loads(row["params"]):
            raise ValueError("Inconsistent Stage A matrix row: "+ident)
        original=path/"08_logs/cases"/(ident+".json")
        rec=read(original) if original.is_file() else {}
        provenance="RECORDED" if rec else "RECONSTRUCTED_FROM_TABLES"
        proof=None;lat=None;speed=None;pressure=None
        if status=="PASS":
            try:
                if not matches_triton(rec.get("triton_version")) or rec.get("anchor_id")!=case["anchor_id"] or rec.get("input_seed")!=int(digest([protocol["seed"],case["anchor_id"]])[:15],16):
                    raise ValueError("missing fresh paired input attestation")
                proof=plan(case,True)
                obs=rec.get("plan",{});actual=proof.json()
                # JSON normalizes tuples and runtime W8 adds qualified bias mode to proof.
                for k in ("ownership","ddg","strategy","merge","cut"):
                    if rec and digest(obs[k])!=digest(actual[k]):raise ValueError("recomputed plan differs: "+k)
                arrays={v:[raw[ident][v][i] for i in range(n)] for v in ("B0","B1","B2")}
                if n<10 or any(len(raw[ident][v])!=n or not all(math.isfinite(x) and x>0 for x in a) for v,a in arrays.items()):raise ValueError("incomplete batches")
                if rec and (not valid(rec) or rec['status']!=status or rec.get("signature")!=digest(protocol) or arrays!=rec['raw']):raise ValueError("case attestation inconsistent")
                m=meta[ident]
                for v in ("B1","B2"):
                    if any(not m[v]["hashes"].get(ext) for ext in ("ttgir","ptx","cubin")) or m[v]["registers"] is None:raise ValueError("incomplete compiler evidence")
                    if rec and m[v]["hashes"]!=rec["metadata"][v]["hashes"]:raise ValueError("binary digest tables disagree")
                    for ext,h in m[v]["hashes"].items():
                        f=path/"04_ir"/ident/v/("kernel."+ext)
                        if not f.is_file() or file_hash(f)!=h:raise ValueError("missing or mismatched Stage A IR")
                changed=m["B1"]["hashes"]["cubin"]!=m["B2"]["hashes"]["cubin"] or m["B1"]["grid"]!=m["B2"]["grid"]
                if not changed and not (case["strategy"]=="S5" and plan(case,False).ddg["execution"]!=proof.ddg["execution"]):raise ValueError("no verified change")
                profiles={"normal","extreme"}|({"stride"} if case["workload"]=="W1" else set())
                ck=checkrows[ident]
                if not {(v,p) for v in arrays for p in profiles}<={(x["variant"],x["profile"]) for x in ck}:raise ValueError("missing numerical gates")
                for x in ck:
                    flags=["passed"]+([] if x["variant"]=="B0" else ["repeat_bitwise"]+(["all_tasks_once"] if case["workload"] in ("W6","W8") else [])+(["kv_pass"] if case["workload"]=="W8" else []))
                    if any(x.get(k,"").lower()!="true" for k in flags):raise ValueError("correctness gate disagrees")
                if legal[ident]["legality"]!="PASS" or not all(ident in ids for ids in indexed.values()):raise ValueError("missing joined evidence")
                lat=median(arrays["B2"]);speed=median(arrays["B1"])/lat
                if not math.isclose(lat,float(row["gpu_latency_us"])) or not math.isclose(speed,float(row["incremental_speedup"])):raise ValueError("aggregates disagree")
                pressure=m["B2"]["smem"]+m["B2"].get("spill",0)*32
            except (KeyError,ValueError,TypeError,IndexError) as error:why.append(str(error))
        trusted=status=="PASS" and not why and not issues
        entry=dict(stage_a_run_id=run_id,stage_a_case_id=ident,workload=case["workload"],strategy=case["strategy"],params=case["params"],status=status,
                   trusted=trusted,reason=" | ".join(why) or row.get("reason",""),historical_us=lat,incremental=speed,provenance=provenance)
        index.append(entry)
        if trusted:seeds.append(dict(**case,latency=lat,incremental=speed,pressure=pressure,contract=contract(case),
                                     input_seed=rec.get("input_seed",int(digest([protocol["seed"],case["anchor_id"]])[:15],16)),stage_a_run_id=run_id))
        else:blocks.append(entry)
    result=dict(run_id=run_id,source_sha256=source,environment_id=protocol["environment_id"],environment=env,protocol=protocol,
                git_commit=env.get("git_commit","UNAVAILABLE"),stage_a_verdict=verdict["verdict"],issues=issues,trusted=len(seeds),total=len(data),
                statuses={s:dict(Counter(x["status"] for x in index if x["strategy"]==s)) for s in ('S1','S2','S3','S4','S5')},
                coverage=dict(Counter(x["workload"] for x in seeds)),ir_directory_present=(path/"04_ir").is_dir(),
                analysis_files={f.name:file_hash(f) for f in (path/"05_analysis").glob("*") if f.is_file()},
                stage_a_dataset_schema=schema,files={k:file_hash(v) for k,v in files.items()})
    if out:write_import(out,result,index,blocks)
    if export_rows:return result,seeds,index,blocks
    return result,seeds

def write_import(out,result,index,blocks):
    out=Path(out);save(out/"stage_a_import_audit.json",result);save(out/"stage_a_blockers.json",blocks)
    table(out/"stage_a_candidate_index.csv",index)
    (out/"stage_a_import_report.md").write_text(f"# Stage A import\n\nRun {result['run_id']}. Verdict {result['stage_a_verdict']}. {result['trusted']}/{result['total']} qualified.\n\n"+str(result["statuses"])+"\n\n"+str(result["issues"])+"\n\nMissing historical IR is never recovered as historical timing. See source and IR manifest.",encoding="utf8")

def select_seeds(rows,smoke=False):
    selected=[]
    for w,subset in sorted(group(rows,lambda x:x["workload"]).items()):
        ordered=sorted(subset,key=lambda r:(r["incremental"],r["id"]))
        # Prefer a successful non-trivial reduction shape for the short pipeline test.
        center=[r for r in ordered if r["params"].get("seq",64)>=64 and r["params"].get("m",32)>=2 and r["params"].get("consumer")!="scale"]
        choices=[(center or ordered)[len(center or ordered)//2]] if smoke else [
            ordered[-1],ordered[len(ordered)//2],ordered[0],max(ordered,key=lambda x:x["pressure"])]
        if not smoke:
            for field in ("dtype","phase","dag","op","layout","consumer"):
                for _,pool in group([r for r in ordered if field in r["params"]],lambda x:x["params"][field]).items():
                    if not any(c["params"].get(field)==pool[0]["params"][field] for c in choices):choices.append(pool[len(pool)//2])
        selected.extend({r["id"]:r for r in choices}.values())
    return selected

def group(values,key):
    groups=defaultdict(list)
    for value in values:groups[key(value)].append(value)
    return groups
