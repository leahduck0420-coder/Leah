import argparse,itertools,json,os,sys,tempfile,time
from pathlib import Path
from stage_a.assets import assets
from stage_a.env import inspect
from stage_a.offline import activate
from stage_a.util import digest,file_hash,save
from .importer import ROOT,audit,discover,select_seeds,write_import
from .planner import original_action
from .search import Pool,select,finite_oracle
from .runner import Service
from .report import Dirs,Results

def parser():
    p=argparse.ArgumentParser(description="Stage B offline single-device joint search")
    p.add_argument("--mode",choices=("full","smoke","preflight","audit"),default="full")
    p.add_argument("--device",choices=["cuda:0"],default="cuda:0")
    p.add_argument("--venv-path",type=Path);p.add_argument("--stage-a-results",type=Path,default=ROOT/"stage_b_inputs/stage_a_results")
    p.add_argument("--offline-assets-dir",type=Path)
    p.add_argument("--model-dir",type=Path);p.add_argument("--real-weights",action="store_true")
    p.add_argument("--only-workload",choices=["W"+str(i) for i in range(1,9)])
    p.add_argument("--only-combination",help="e.g. S1+S2 or S3+S4+S5")
    p.add_argument("--max-depth",type=int,choices=(2,3,4),default=4);p.add_argument("--beam-width",type=int,choices=(4,8,16),default=16)
    p.add_argument("--search-budget",type=int,choices=(20,50,100),default=100)
    p.add_argument("--max-memory-gb",type=float,default=24.);p.add_argument("--timeout-seconds",type=int,default=900)
    p.add_argument("--output-dir",type=Path);p.add_argument("--seed",type=int,default=20261119);p.add_argument("--resume",action="store_true")
    return p

def asset_directory(explicit,source):
    if explicit:return Path(explicit).resolve()
    if os.environ.get("SEAM_OFFLINE_ASSETS"):return Path(os.environ["SEAM_OFFLINE_ASSETS"]).resolve()
    for parent in [source,*list(source.parents)[:3]]:
        folder=parent/"offline_assets"
        if folder.is_dir():return folder
    return ROOT/"offline_assets"

def main(argv=None):
    opt=parser().parse_args(argv);activate()
    selected=opt.venv_path or os.getenv("SEAM_SELECTED_VENV") or os.getenv("SEAM_VENV")
    if not selected or Path(selected).resolve()!=Path(sys.prefix).resolve() or not (Path(selected)/"pyvenv.cfg").is_file():
        raise SystemExit("Use the SEAM .venv via run_stage_b.sh --venv-path")
    if not 0<opt.max_memory_gb<float("inf") or opt.timeout_seconds<1:raise SystemExit("invalid memory/time limit")
    if not sys.platform.startswith("linux"):raise SystemExit("SEAM Linux required")
    import fcntl
    lock=open(Path(tempfile.gettempdir())/"seam-stage-a-benchmark.lock","a+")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("Stage A or B already owns the GPU benchmark lock")
    only=tuple(x.strip().upper() for x in opt.only_combination.replace(",","+").split("+")) if opt.only_combination else ()
    if any(x not in ("S1","S2","S3","S4","S5") for x in only):raise SystemExit("invalid strategy name")
    try:
        src=discover(opt.stage_a_results);input_mode="smoke" if opt.mode=="smoke" else "full";aud,rows,index,blocks=audit(src,mode=input_mode,export_rows=True)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        reason="Stage A import failed: "+str(exc)
        dest=(opt.output_dir or ROOT/"results"/("blocked-import-"+str(time.time_ns()))).resolve()
        inputs=opt.stage_a_results.resolve()
        if dest==inputs or inputs in dest.parents or dest in inputs.parents or dest.exists() and any(dest.iterdir()):
            raise SystemExit(reason)
        save(dest/"00_environment/stage_a_import_audit.json",{"status":"BLOCKED","reason":reason})
        save(dest/"08_reports/acceptance_report.json",{"verdict":"FAIL","qualified":0,"failed_gates":["stage_a_import"],"reason":reason})
        print(reason,file=sys.stderr);return 2
    env=inspect(selected,opt.device) if opt.mode!="audit" else {"errors":["AUDIT_ONLY"]}
    if opt.mode!="audit":
        for package in ("numpy","matplotlib"):
            if env.get(package)=="NOT_AVAILABLE":env["errors"].append("required reporting dependency missing: "+package)
        try:
            from triton.runtime.jit import JITFunction
            if not callable(getattr(JITFunction,"warmup",None)):env["errors"].append("JITFunction.warmup unavailable")
        except ImportError as exc:env["errors"].append(repr(exc))
    opt.offline_assets_dir=asset_directory(opt.offline_assets_dir,src)
    try:
        asset=assets(opt.offline_assets_dir,opt.model_dir,opt.real_weights)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        asset={"qwen_status":"BLOCKED","weights":"REAL_WEIGHTS" if opt.real_weights else "QWEN_SHAPE_SYNTHETIC_WEIGHTS","reason":repr(exc)}
    if opt.mode!="audit":
        original=aud["protocol"]
        for key,field in (("config","config_sha256"),("weight_mode","weights"),("weights","weight_files")):
            if original.get(key)!=asset.get(field):
                env["errors"].append("Stage A/B model asset mismatch: "+field)
    smoke=opt.mode=="smoke"
    depth=min(opt.max_depth,3) if smoke else opt.max_depth
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items() if k not in ("resume","output_dir","timeout_seconds")}
    inputs={k:env.get(k) for k in ("python","version","venv","gpu","gpu_selector","capability","torch","triton","libtriton_sha256",
                                      "driver_version","nvcc","torch_cuda_runtime","cuda_graph")}
    protocol=dict(options=options,environment=inputs,stage_a=aud["files"],stage_a_run=aud["run_id"],
                  assets={k:asset.get(k) for k in ("config_sha256","weights","weight_files")},batches=10 if smoke else 12,replays=2 if smoke else 3,
                  stage_b_code=digest({f.relative_to(ROOT).as_posix():file_hash(f) for f in (ROOT/"stage_b").rglob("*.py")}),
                  stage_a_code=aud["source_sha256"],trusted=digest(sorted((r["id"],r["input_seed"]) for r in rows)))
    protocol["signature"]=digest(protocol)
    out=(opt.output_dir or ROOT/"results"/(opt.mode+"-"+protocol["signature"][:12]+("" if opt.resume else "-"+str(time.time_ns())))).resolve()
    if out==src or src in out.parents or out in src.parents:raise SystemExit("Keep Stage B output separate from Stage A inputs")
    prior=out/"00_environment/protocol.json"
    if out.exists() and any(out.iterdir()) and (not opt.resume or not prior.is_file() or json.loads(prior.read_text())!=protocol):
        raise SystemExit("Output not empty or resume signature differs")
    for d in Dirs:(out/d).mkdir(parents=True,exist_ok=True)
    for name in ("runner.log","error.log"):(out/"10_logs"/name).touch(exist_ok=True)
    e=out/Dirs[0];save(prior,protocol);save(e/"environment.json",env);save(e/"versions.json",inputs);save(e/"assets.json",asset)
    write_import(e,aud,index,blocks)
    compared=("venv","python","torch","triton","torch_cuda_runtime","libtriton_sha256","triton_path",
              "gpu","gpu_selector","capability","driver_version","nvcc","cuda_graph","matmul_precision")
    if opt.mode!="audit":
        mismatches=[k for k in compared if env.get(k) is None or env.get(k)!=aud["environment"].get(k)]
        if mismatches:env["errors"].append("fresh Stage A environment mismatch: "+", ".join(mismatches))
        save(e/"environment.json",env)
    save(e/"comparison_environment.json",{k:{"historical":aud["environment"].get(k),"current":env.get(k),"matches":env.get(k) is not None and env.get(k)==aud["environment"].get(k)} for k in compared})
    print("Results",out,"source",aud["run_id"],"trusted",len(rows),"device",json.dumps(inputs),flush=True)
    if opt.mode in ("audit","preflight"):
        print("Preflight",json.dumps({"environment":env["errors"],"historical":aud["issues"],"Qwen":asset["qwen_status"]}),flush=True)
        return int(not rows or bool(aud["issues"]) or (opt.mode=="preflight" and (bool(env["errors"]) or asset["qwen_status"]!="READY")))
    chosen=[c for c in select_seeds(rows,smoke) if not opt.only_workload or c["workload"]==opt.only_workload]
    store=Results(out,aud,protocol)
    service=Service(out,opt,protocol,asset,env)
    if env["errors"] or aud["issues"]:service.fatal="; ".join(env["errors"]+aud["issues"])
    for n,seed in enumerate(chosen):
        base_seed=seed
        if opt.seed!=aud["protocol"]["seed"]:base_seed={**seed,"input_seed":int(digest([opt.seed,seed["id"]])[:15],16)}
        pool=Pool(base_seed,depth,only)
        if only and not set(only)<=set(a.strategy for a in pool.actions):
            store.unavailable(base_seed,only);continue
        print(n+1,len(chosen),seed["id"],"finite plans",len(pool.nodes),flush=True)
        store.structure(pool)
        def evaluate(h):return service.get(base_seed,base_seed,pool.nodes[h],h)
        if seed["workload"] in ("W7","W8") and asset.get("qwen_status")!="READY":
            store.blocked(pool,dict(status="BLOCKED",reason="Qwen offline resources: "+str(asset.get("reason",asset.get("missing_shards")))));continue
        base=evaluate(pool.root)
        if base["status"]!="PASS":
            store.blocked(pool,base);continue
        matched=[r for r in rows if r["contract"]==seed["contract"]]
        refs=[]
        for c in sorted(matched,key=lambda r:r["id"]):
            ref=service.get(base_seed,c,(original_action(c),),c["id"],True);refs.append((c,ref))
        policies=[]
        for method in ("none","single","fixed"):
            policies.append(select(pool,method,evaluate,len(pool.nodes) if method=="single" else opt.search_budget,
                                   1,1 if method=="single" else depth))
        budgets=[b for b in (20,50,100) if b<=opt.search_budget] if not smoke else [20]
        widths=[w for w in (4,8,16) if w<=opt.beam_width] if not smoke else [opt.beam_width]
        for b,d in itertools.product(budgets,range(2,depth+1)):
            policies.append(select(pool,"greedy",evaluate,b,1,d))
            for width in widths:policies.append(select(pool,"beam",evaluate,b,width,d))
        for change in (dict(guided=False),dict(dag=False),dict(fixed_param=True),dict(fixed_order=True)):
            result=select(pool,"beam",evaluate,budgets[-1],widths[-1],depth,**change)
            result["ablation"]=next(iter(change));policies.append(result)
        # The oracle is filled only after every search policy has finished choosing.
        records={h:evaluate(h) for h in pool.nodes}
        store.region(pool,records,refs,policies)
        store.flush_partial()
    verdict=store.finish(env,asset,service.fatal)
    if os.getenv("SEAM_AB_INVOCATION"):save(out/"pipeline_completion.json",dict(invocation=os.environ["SEAM_AB_INVOCATION"],verdict=verdict["verdict"]))
    print("Acceptance",verdict["verdict"],out/"08_reports/acceptance_report.json")
    if opt.mode=="smoke":return int(not verdict["smoke_complete"])
    return int(verdict["verdict"]!="PASS")

if __name__=="__main__":sys.exit(main())
