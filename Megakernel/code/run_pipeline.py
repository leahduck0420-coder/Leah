"""Run Stage A and its Stage B consumer sequentially in the selected SEAM interpreter."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from stage_a.all_pairs import PROTOCOL, fingerprint
from stage_a.util import digest, file_hash, save
from stage_a.toolchain import require_triton
from stage_b.importer import REQUIRED, read

ROOT=Path(__file__).resolve().parent

def options(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("smoke","full"),default="full")
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--venv-path",type=Path)
    p.add_argument("--seed",type=int,default=20261119)
    p.add_argument("--max-memory-gb",type=float,default=24.)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--offline-assets-dir",type=Path,default=ROOT/"offline_assets")
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--real-weights",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--retry-stage-a",action="store_true",help="on resume retry failed A cases before re-import")
    return p.parse_args(argv)

def handoff(a):
    """Authenticate the completed producer and identify precisely the evidence Stage B will read."""
    files={k:file_hash(a/path) for k,path in REQUIRED.items()}
    manifest=read(a/REQUIRED["manifest"])
    if manifest["sha256"]!=files["dataset"]:
        raise ValueError("Stage A dataset hash disagrees with its manifest")
    cases=sorted((a/"08_logs/cases").glob("*.json"))
    matrix=read(a/REQUIRED["matrix"])
    if {f.stem for f in cases}!={c["id"] for c in matrix} or len(cases)!=len(matrix):
        raise ValueError("Stage A is incomplete; resume the same run")
    if manifest["rows"]!=len(matrix):
        raise ValueError("Stage A has unreported requests")
    files["individual_records"]=digest({f.name:file_hash(f) for f in cases})
    return digest(files)

def stages(root,opt,signature):
    a=root/"stage_a"
    manifest_path=root/"pipeline.json"
    state=read(manifest_path) if manifest_path.is_file() else {}
    if state and state["signature"]!=signature:raise SystemExit("Different pipeline inputs; choose a fresh output directory")
    def persist(**updates):
        state.update(signature=signature,design=PROTOCOL,mode=opt.mode,stage_a_path="stage_a",**updates)
        save(manifest_path,state)
    common=["--mode",opt.mode,"--device","cuda:0","--venv-path",str(Path(sys.prefix).resolve()),
            "--seed",str(opt.seed),"--max-memory-gb",str(opt.max_memory_gb),"--offline-assets-dir",str(opt.offline_assets_dir.resolve())]
    if opt.real_weights:common.append("--real-weights")
    if opt.model_dir:common+=["--model-dir",str(opt.model_dir.resolve())]
    def execute(stage,path,extra):
        if path.exists() and not path.is_dir():raise ValueError("Invalid stage directory")
        token=uuid.uuid4().hex
        proc=subprocess.run([sys.executable,"-B","-m","stage_"+stage+".cli",*common,"--output-dir",str(path),
                             *(["--resume"] if opt.resume else []),*extra],cwd=ROOT,
                             env={**os.environ,"SEAM_AB_INVOCATION":token})
        stamp=path/"pipeline_completion.json"
        if not stamp.is_file() or read(stamp).get("invocation")!=token:return 2
        return proc.returncode
    def blocked(reason):
        # Keep an earlier B run intact if a later A attempt is interrupted.
        previous=state.get("stage_b_path")
        b=root/"stage_b"/"blocked"
        save(b/"08_reports/acceptance_report.json",dict(verdict="BLOCKED",reason=reason))
        persist(stage_b_path=b.relative_to(root).as_posix(),stage_b_verdict="BLOCKED",overall="INCOMPLETE",reason=reason)
        if previous and previous!=state["stage_b_path"]:persist(previous_stage_b_path=previous)
        print(reason,flush=True)
        return 2
    use_completed=opt.resume and not opt.retry_stage_a and state.get("a_complete")
    if use_completed:
        try:
            if handoff(a)!=state.get("a_evidence"):return blocked("Stored Stage A evidence changed; inspect files or use --retry-stage-a")
        except (OSError,ValueError,KeyError) as exc:return blocked(str(exc))
    else:
        persist(overall="RUNNING_A",a_complete=False)
        rc=execute("a",a,[])
        if rc not in (0,1) or not (a/"06_reports/acceptance_report.json").is_file():
            return blocked("Stage A interrupted or did not produce acceptance; exit "+str(rc))
    try:
        evidence=handoff(a);verdict=read(a/"06_reports/acceptance_report.json")
        protocol=read(a/REQUIRED["protocol"])
        if protocol.get("design")!=PROTOCOL or protocol.get("code")!=fingerprint() or protocol["mode"]!=opt.mode:
            raise ValueError("Stage A source, matrix or protocol does not match")
    except (OSError,ValueError,KeyError) as exc:return blocked(str(exc))
    persist(a_complete=True,a_evidence=evidence,stage_a_verdict=verdict["verdict"],overall="IMPORTING_A")
    if verdict["environment_errors"] or not verdict["eligible_measured"] or not verdict["gates"]["all_40_requested"]:
        return blocked("Stage A environment or input coverage prevents the Stage B run")
    if any(read(f).get("fatal_device") for f in (a/"08_logs/cases").glob("*.json")):
        return blocked("Stage A quarantined the device. Diagnose and resume with --retry-stage-a")
    if any(x.get("fatal_device") for x in read(a/"00_environment/microbench.json")):
        return blocked("Stage A microbenchmark quarantined the device. Diagnose and resume with --retry-stage-a")
    b=root/"stage_b"/evidence[:16]
    persist(stage_b_path=b.relative_to(root).as_posix(),stage_b_verdict="RUNNING",overall="RUNNING_B",reason="")
    rc=execute("b",b,["--stage-a-results",str(a)])
    if not (b/"08_reports/acceptance_report.json").is_file():
        return blocked("Stage B did not finish; inspect its logs and resume. Exit "+str(rc))
    vr=read(b/"08_reports/acceptance_report.json")
    if rc not in (0,1):return blocked("Stage B exited abnormally: "+str(rc))
    smoke_a=not verdict["environment_errors"] and not verdict["missing_pairs"] and verdict["gates"]["core_failures_absent"] and verdict["gates"]["all_requested_accounted"] and verdict["gates"]["plots"]
    if opt.mode=="smoke":overall="SMOKE_COMPLETE" if smoke_a and rc==0 and vr.get("smoke_complete") else "SMOKE_INCOMPLETE"
    else:overall="PASS" if rc==0 and vr["verdict"]==verdict["verdict"]=="PASS" else "FAIL" if "FAIL" in (verdict["verdict"],vr["verdict"]) else "PARTIAL"
    persist(overall=overall,stage_b_verdict=vr["verdict"],stage_b_exit=rc)
    print("Combined",overall,"\nStage A",a,"\nStage B",b,flush=True)
    return int(overall not in ("SMOKE_COMPLETE","PASS"))

def main(argv=None):
    o=options(argv)
    if sys.platform!="linux":raise SystemExit("Benchmarks require SEAM Linux")
    selected=os.environ.get("SEAM_SELECTED_VENV")
    if not selected or Path(selected).resolve()!=Path(sys.prefix).resolve() or not (Path(selected)/"pyvenv.cfg").is_file():
        raise SystemExit("Use run_all.sh with the existing SEAM .venv")
    try:
        active_triton=require_triton()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    from triton._C import libtriton
    import fcntl
    if o.retry_stage_a and not o.resume:raise SystemExit("--retry-stage-a requires --resume")
    payload={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(o).items() if k not in ("venv_path","output_dir","resume","retry_stage_a")}
    signature=digest(dict(options=payload,venv=str(Path(sys.prefix).resolve()),code=fingerprint(),triton=active_triton,libtriton=file_hash(libtriton.__file__)))
    tag=signature[:12] if o.resume else datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    root=(o.output_dir or ROOT/"results"/(o.mode+"-"+tag)).resolve()
    if root.exists() and any(root.iterdir()) and (not o.resume or not (root/"pipeline.json").is_file()):
        raise SystemExit("Use a new output directory or resume a recognized pipeline run")
    root.mkdir(parents=True,exist_ok=True)
    with (root/".pipeline.lock").open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("This pipeline is already active")
        return stages(root,o,signature)
if __name__=="__main__":
    sys.exit(main())
