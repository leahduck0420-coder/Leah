import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from .assets import ROOT,assets
from .env import inspect
from .toolchain import TRITON_VERSION, matches_triton
from .all_pairs import PROTOCOL,fingerprint
from .legality import regression
from .matrix import matrix,SUPPORT,TARGET
from .report import DIRS,generate
from .util import save,digest,file_hash
from .workloads import INPUT_CONTRACT_VERSION,QKV_PROJECTION_CONTRACT_VERSION


def parser():
    p=argparse.ArgumentParser(description="Offline single-GPU Stage A")
    p.add_argument("--mode",choices=("full","smoke","preflight","assets","micro"),default="full")
    p.add_argument("--device",default="cuda:0");p.add_argument("--venv-path",type=Path)
    p.add_argument("--resume",action="store_true");p.add_argument("--output-dir",type=Path)
    p.add_argument("--only-strategy",choices=list(TARGET));p.add_argument("--only-workload",choices=list(SUPPORT))
    p.add_argument("--seed",type=int,default=20261119);p.add_argument("--max-memory-gb",type=float,default=24)
    p.add_argument("--offline-assets-dir",type=Path,default=ROOT/"offline_assets")
    p.add_argument("--model-dir",type=Path);p.add_argument("--real-weights",action="store_true")
    p.add_argument("--timeout-seconds",type=int,default=600)
    p.add_argument("--micro-timeout-seconds",type=float,default=120)
    from .micro import items
    p.add_argument("--micro-item",choices=[item["id"] for item in items()])
    return p


def cached(case,loc,signature,ir):
    if not loc.is_file():return None
    try:
        r=json.loads(loc.read_text(encoding="utf8"))
        if not matches_triton(r.get("triton_version")) or r["id"]!=case["id"] or r.get("signature")!=signature:return None
        if r["status"] not in ("PASS","NO_OP","INVALID_REJECTED","UNSUPPORTED","NOT_APPLICABLE"):return None
        if r["status"]!="PASS":
            fb=r.get("fallback",{})
            from math import isfinite
            from statistics import median
            if fb.get("status")!="MEASURED" or fb.get("binary")!=r["metadata"]["B1"]["hashes"]["cubin"]:return None
            for k in ("B0","B1","fallback"):
                times=fb["raw_us"][k]
                if len(times)<10 or not all(isfinite(t) and t>0 for t in times) or median(times)!=fb["statistics"][k]["median"]:return None
        if r["status"]=="PASS" or r.get("fallback"):

            from .report import valid
            if r["status"]=="PASS" and not valid(r):return None
            for mode in ("B1","B2"):
                for ext,h in r["metadata"][mode]["hashes"].items():
                    if file_hash(ir/mode/("kernel."+ext))!=h:return None
        return r
    except (ValueError,KeyError,OSError,TypeError):return None


def signature_for(opt,env,asset):
    return dict(required_triton=TRITON_VERSION,mode=opt.mode,device=opt.device,seed=opt.seed,filtered=bool(opt.only_strategy or opt.only_workload),
                input_contract_version=INPUT_CONTRACT_VERSION,
                qkv_projection_contract_version=QKV_PROJECTION_CONTRACT_VERSION,
                matmul_precision=env.get("matmul_precision"),
                only_strategy=opt.only_strategy,only_workload=opt.only_workload,max_memory_gb=opt.max_memory_gb,
                environment_id=digest({k:env.get(k) for k in ("venv","version","torch","triton","gpu","gpu_selector","libtriton_sha256",
                    "torch_cuda_runtime","driver_supported_cuda","driver_version","nvcc")}),config=asset.get("config_sha256"),
                micro_timeout_seconds=opt.micro_timeout_seconds,micro_item=opt.micro_item,
                weights=asset.get("weight_files"),weight_mode=asset["weights"],graph=env.get("cuda_graph",False),
                design=PROTOCOL,code=fingerprint(),batches=12 if opt.mode=="full" else 10,
                replays=3 if opt.mode=="full" else 2)


def main(args=None):
    opt=parser().parse_args(args)
    import math
    if not math.isfinite(opt.micro_timeout_seconds) or opt.micro_timeout_seconds<=0:
        raise SystemExit("--micro-timeout-seconds must be finite and positive")
    if opt.micro_item and opt.mode!="micro":
        raise SystemExit("--micro-item requires --mode micro")
    from .offline import activate
    activate()
    if opt.max_memory_gb<=0 or opt.timeout_seconds<1:raise SystemExit("invalid memory limit or timeout")
    venv=opt.venv_path or os.getenv("SEAM_SELECTED_VENV") or os.getenv("SEAM_VENV")
    if not venv or Path(sys.prefix).resolve()!=Path(venv).resolve() or not (Path(venv)/"pyvenv.cfg").is_file():
        raise SystemExit("Use run_stage_a.sh with the existing SEAM venv (interpreter prefix mismatch)")
    for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","PIP_NO_INDEX"):os.environ[key]="1"
    if platform.system()!="Linux":raise SystemExit("formal measurements require Linux SEAM")
    import fcntl
    import tempfile
    # Shared host lock serializes benchmark jobs, including processes using alternate device aliases.
    lock=open(Path(tempfile.gettempdir())/"seam-stage-a-benchmark.lock","a+")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("Another Stage A job owns the GPU benchmark lock")
    try: asset=assets(opt.offline_assets_dir,opt.model_dir,opt.real_weights)
    except (OSError,ValueError,KeyError) as ex:asset=dict(qwen_status="BLOCKED",weights="REAL_WEIGHTS" if opt.real_weights else "QWEN_SHAPE_SYNTHETIC_WEIGHTS",reason=repr(ex))
    if opt.mode=="assets":
        print(json.dumps(asset,indent=2));return int(asset["qwen_status"]!="READY")
    env=inspect(venv,opt.device)
    from .offline import DISABLED_EXECUTABLES
    env["disabled_executables"]=sorted(DISABLED_EXECUTABLES)
    print(json.dumps(env,indent=2),flush=True)
    protocol=signature_for(opt,env,asset);sig=digest(protocol)
    root=opt.output_dir or ROOT/"results"/(opt.mode+"-"+sig[:12] if opt.resume else "run-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if root.exists() and any(root.iterdir()):
        prior=root/"00_environment/protocol.json"
        if not opt.resume or not prior.is_file() or json.loads(prior.read_text())!=protocol:
            raise SystemExit("Output directory is not empty or resume protocol differs")
    for directory in DIRS:(root/directory).mkdir(parents=True,exist_ok=True)
    e=root/DIRS[0];logs=root/DIRS[-1];casesdir=logs/"cases";casesdir.mkdir(exist_ok=True)
    print("Results:",root.resolve(),flush=True)
    save(e/"environment.json",env);save(e/"versions.json",{k:env.get(k) for k in ("python","version","torch","triton","required_triton","triton_path","libtriton_sha256","torch_cuda_runtime","cuda_toolkit_release","nvcc","driver_supported_cuda","driver_state")})
    save(e/"offline_assets_status.json",asset);save(e/"protocol.json",protocol)
    cpu=regression()
    save(e/"preflight.json",dict(errors=env["errors"],cpu=cpu,workloads={w:list(TARGET) for w in SUPPORT},anchor_strata=SUPPORT,asset_status=asset["qwen_status"]))
    if opt.mode=="preflight":return int(bool(env["errors"]) or not all(x["passed"] for x in cpu))
    planned=[c for c in ([] if opt.mode=="micro" else matrix(opt.seed,opt.mode=="smoke")) if
             (not opt.only_strategy or c["strategy"]==opt.only_strategy) and (not opt.only_workload or c["workload"]==opt.only_workload)]
    if not planned and opt.mode!="micro":raise SystemExit("No registered workload/strategy pair for the chosen filters")
    save(e/"matrix.json",planned)
    hw=env["errors"].copy();micro=[]
    if not all(x["passed"] for x in cpu):hw.append("CPU legality verification failed")
    if not hw:
        try:
            from .micro import run
            micro=run(opt.device,opt.seed,protocol["graph"],opt.max_memory_gb,
                      output_dir=root,timeout_seconds=opt.micro_timeout_seconds,only_item=opt.micro_item)
            if any(row.get("fatal_device") for row in micro):
                hw.append("A0 worker failed or timed out; inspect 08_logs/micro before retrying")
        except Exception as ex:
            micro.append(dict(family="A0",name="microbench",status="BLOCKED",reason=repr(ex)))
            if any(x in str(ex).lower() for x in ("illegal memory","device-side","device lost")):hw.append("A0 device error: "+repr(ex))
    save(e/"microbench.json",micro)
    if opt.mode=="micro":
        failed=[row for row in micro if row["status"] not in ("MEASURED","COMPILER_REPORTED","SKIPPED_OPTIONAL")]
        status="PASS" if micro and not hw and not failed else "FAIL"
        save(e/"microbench_status.json",dict(status=status,errors=hw,failed=failed,selected_item=opt.micro_item))
        print("Microbench:",status,e/"microbench.json",flush=True)
        return int(status!="PASS")
    rows=[]; optional_done=set();proton_attempted=False
    with (logs/"runner.log").open("a",encoding="utf8") as log,(logs/"error.log").open("a",encoding="utf8") as err:
        for i,case in enumerate(planned):
            target=casesdir/(case["id"]+".json");ir=root/"04_ir"/case["id"]
            rec=cached(case,target,sig,ir) if opt.resume else None
            if rec is None:
                if hw:rec=dict(**case,status="BLOCKED",reason="; ".join(hw))
                elif case["workload"] in ("W7","W8") and asset["qwen_status"]!="READY":rec=dict(**case,status="BLOCKED",reason="Qwen input unavailable")
                else:
                    case_seed=int(digest([opt.seed,case["anchor_id"]])[:15],16)
                    req=dict(case=case,folder=str(ir.resolve()),seed=case_seed,device=opt.device,assets=asset,selector=env["gpu_selector"],
                             max_memory_gb=opt.max_memory_gb,graph=protocol["graph"],batches=protocol["batches"],replays=protocol["replays"])
                    req["probe_proton"]=not proton_attempted
                    request=logs/"current_request.json";save(request,req)
                    attempt=casesdir/(case["id"]+".attempt.json")
                    attempt.unlink(missing_ok=True)
                    try:
                        code=subprocess.run([sys.executable,"-B","-m","stage_a.worker",str(request.resolve()),str(attempt.resolve())],
                                            cwd=ROOT,stdout=log,stderr=err,timeout=opt.timeout_seconds).returncode
                        if code or not attempt.exists():
                            rec=dict(**case,status="RUNTIME_ERROR",reason=f"worker exited {code}",fatal_device=code<0)
                        else:rec=json.loads(attempt.read_text(encoding="utf8"))
                    except subprocess.TimeoutExpired:rec=dict(**case,status="TIMEOUT",reason="case process terminated at wall clock limit",fatal_device=True)
                    attempt.unlink(missing_ok=True)
                    key=(case["workload"],case["params"].get("op"),case["params"]["dtype"])
                    compatible_control=(case["workload"]!="W5" or case["params"]["seq"]%128==0)
                    if rec["status"]=="PASS" and compatible_control and key not in optional_done and case["workload"] in ("W2","W3","W4","W5","W7"):
                        optional_done.add(key)
                        try:
                            alt=subprocess.run([sys.executable,"-B","-m","stage_a.optional",str(request.resolve()),str(attempt.resolve())],
                                               cwd=ROOT,stdout=log,stderr=err,timeout=opt.timeout_seconds)
                            extra=json.loads(attempt.read_text()) if attempt.is_file() and alt.returncode==0 else {"local_controls":{"status":"SKIPPED_OPTIONAL","reason":f"exit {alt.returncode}"}}
                            rec.setdefault("optional",{}).update(extra)
                            if any(item.get("fatal_device") for item in extra.values()):rec["fatal_device"]=True
                        except subprocess.TimeoutExpired:
                            rec.setdefault("optional",{})["local_controls"]={"status":"TIMEOUT","reason":"optional worker timed out"}
                        attempt.unlink(missing_ok=True)
                rec.setdefault("input_seed",int(digest([opt.seed,case["anchor_id"]])[:15],16));rec["signature"]=sig;save(target,rec)
            if set(rec.get("optional",{}))-{"torch_sdpa"}:
                optional_done.add((case["workload"],case["params"].get("op"),case["params"]["dtype"]))
            if "proton" in rec.get("optional",{}):proton_attempted=True
            rows.append(rec)
            if rec.get("fatal_device") and not hw:hw.append("device quarantined after "+case["id"]+"; diagnose and resume")
            msg=f"{i+1}/{len(planned)} {case['id']} {rec['status']}"
            print(msg,flush=True);print(msg,file=log,flush=True)
            if "reason" in rec:print(msg,rec["reason"],file=err,flush=True)
            save(logs/"progress.json",dict(last=case["id"],completed=i+1,total=len(planned),signature=sig))
    verdict=generate(root,env,asset,protocol,rows,cpu,micro)
    if os.getenv("SEAM_AB_INVOCATION"):save(root/"pipeline_completion.json",dict(invocation=os.environ["SEAM_AB_INVOCATION"],verdict=verdict["verdict"]))
    print("Acceptance:",verdict["verdict"],root/"06_reports/acceptance_report.json")
    if opt.mode=="smoke" and not (env["errors"] or verdict["missing_pairs"] or not verdict["gates"]["core_failures_absent"] or not verdict["gates"]["plots"] or not verdict["gates"]["all_requested_accounted"]):return 0
    return 0 if verdict["verdict"]=="PASS" else 1

if __name__=="__main__":sys.exit(main())
