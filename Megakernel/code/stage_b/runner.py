"""Serial single-device evaluation service. Policies obtain measurements through a paid query."""
import json,math,os,subprocess,sys,tempfile,time
from statistics import median
from pathlib import Path
from stage_a.util import digest,file_hash,save
from .importer import read,ROOT
from .planner import original_action
from stage_a.toolchain import matches_triton

def admissible_cache(record,identity,folder,batches):
    if not isinstance(record,dict):return False
    if record.get("identity")!=identity or record.get("status") not in ("PASS","INVALID_REJECTED","NO_OP","RESOURCE_EXCEEDED","UNSUPPORTED"):return False
    if not matches_triton(record.get("triton_version")):return False
    if record["status"]!="PASS":return True
    try:
        raw=record["raw_us"];sequence=record["sequence"]
        expected={"B0","B1","candidate"} if sequence else {"B0","B1"}
        if record.get("phase")!="complete" or expected!=set(raw):return False
        times=record["statistics"]
        for variant,values in raw.items():
            if len(values)!=batches or not all(math.isfinite(x) and x>0 for x in values):return False
            if times[variant]["median"]!=median(values) or times[variant]["batches"]!=batches:return False
        if record["latency_us"]!=times["candidate" if sequence else "B1"]["median"]:return False
        if len(record["order"])!=batches or any(sorted(o)!=sorted(expected) for o in record["order"]):return False
        artifacts=record["prefix_artifacts"]
        if [x["step"] for x in artifacts]!=list(range(len(sequence)+1)):return False
        profiles={"normal","extreme"}|({"stride"} if record["source_case"].startswith("W1-") else set())
        checks={(c["variant"],c["profile"]):c for c in record["checks"]}
        for v in ["B0"]+[x["plan"] for x in artifacts]:
            for profile in profiles:
                c=checks[v,profile]
                if not c["passed"] or (v!="B0" and not c["admitted"]):return False
        for x in artifacts:
            base=folder/("prefix_"+str(x["step"]))
            hashes=x["meta"]["hashes"]
            if not {"ttir","ttgir","llir","ptx","cubin"}<=hashes.keys():return False
            if any(file_hash(base/("kernel."+ext))!=h for ext,h in hashes.items()):return False
            if any(not (base/f).is_file() for f in ("plan.json","ownership.json","ddg.json")):return False
        return bool(record["cost_by_strategy"])
    except (KeyError,OSError,TypeError,AttributeError,ValueError):return False

class Service:
    def __init__(self,out,options,protocol,asset,env):
        self.out=out;self.opt=options;self.protocol=protocol;self.asset=asset;self.env=env
        self.fatal=None;self.cache={};self.new_calls=0

    def get(self,seed,case,seq,h,legacy=False):
        seq_json=[a.json() for a in seq]
        name=("stage_a_"+case["id"]) if legacy else h
        identity=digest([self.protocol["signature"],seed["id"],case["id"],name,seed["input_seed"],seq_json])
        if identity in self.cache:return self.cache[identity]
        log=self.out/"10_logs";folder=self.out/"06_ir"/seed["id"]/name
        dest=log/"records"/seed["id"]/(name+".json")
        seal=dest.with_suffix(".sha256")
        if self.opt.resume and dest.is_file() and seal.is_file() and file_hash(dest)==seal.read_text().strip():
            try:r=read(dest)
            except (OSError,ValueError):r={}
            if admissible_cache(r,identity,folder,self.protocol["batches"]):
                self.cache[identity]=r;return r
        req=dict(case=case,sequence=seq_json,input_seed=seed["input_seed"],legacy=legacy,folder=str(folder),
                 selector=self.env.get("gpu_selector"),memory_gb=self.opt.max_memory_gb,assets=self.asset,graph=self.env.get("cuda_graph",False),
                 batches=self.protocol["batches"],replays=self.protocol["replays"],identity=identity)
        rec=dict(status="BLOCKED",reason=self.fatal or "preflight failed",sequence=seq_json,identity=identity)
        if not self.fatal:
            folder.mkdir(parents=True,exist_ok=True)
            current=log/"request.json";save(current,req)
            attempt=log/"attempt.json";attempt.unlink(missing_ok=True)
            t=time.perf_counter()
            try:
                # Each process gets a fresh Triton disk cache, including when resuming failed measurements.
                with tempfile.TemporaryDirectory(prefix="stage-b-jit-") as cache,(log/"runner.log").open("a",encoding="utf8") as stdout,(log/"error.log").open("a",encoding="utf8") as stderr:
                    proc=subprocess.run([sys.executable,"-B","-m","stage_b.worker",str(current),str(attempt)],
                          cwd=ROOT,env={**os.environ,"TRITON_CACHE_DIR":cache},timeout=self.opt.timeout_seconds,stdout=stdout,stderr=stderr)
                if proc.returncode or not attempt.is_file():raise RuntimeError("isolated worker exited "+str(proc.returncode))
                rec=read(attempt)
                if rec.get("identity")!=identity:raise ValueError("stale worker output")
            except subprocess.TimeoutExpired:
                rec.update(status="TIMEOUT",reason="worker wall-clock timeout",fatal=True)
            except (OSError,ValueError,RuntimeError) as e:rec.update(status="RUNTIME_ERROR",reason=repr(e),fatal=True)
            rec["process_seconds"]=time.perf_counter()-t;self.new_calls+=1
            if rec.get("fatal"):self.fatal=rec.get("reason","device quarantined")
        rec.update(source_case=case["id"],region=seed["id"],plan_id=name)
        save(dest,rec)
        temp=seal.with_suffix(".sha256.tmp");temp.write_text(file_hash(dest),encoding="ascii");temp.replace(seal)
        self.cache[identity]=rec
        with (log/"runner.log").open("a",encoding="utf8") as f:print(seed["id"],name,rec["status"],file=f)
        if rec["status"]!="PASS":
            with (log/"error.log").open("a",encoding="utf8") as f:print(seed["id"],name,rec["status"],rec.get("reason"),file=f)
        save(log/"progress.json",dict(last_case=seed["id"],last_plan=name,status=rec["status"],new_evaluations=self.new_calls,quarantine=self.fatal))
        return rec
