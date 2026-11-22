"""Balanced deterministic stratification, fixed public workload/strategy matrix."""
import itertools
import random
from .util import digest

TARGET = dict(S1=64, S2=96, S3=84, S4=76, S5=72)
SUPPORT = dict(W1=["S1"], W2=["S1","S2"], W3=["S1","S2","S3"], W4=["S3","S4"],
               W5=["S3","S4"], W6=["S5"], W7=["S1","S2","S3"], W8=["S3","S4","S5"])
DT = ["float16","bfloat16"]


def axes(w, strategy):
    if w == "W1":
        return dict(n=[4096,65536,1048576,16777216,4099], tile=[128,256,512,1024],
                    layout=["thread","lane","warp","cross_program"], consumers=[1,2,4], dtype=DT)
    if w == "W2":
        return dict(m=[16,128,512,1024,65], n=[512,2048,4096,515], k=[512,2048,4096],
                    bm=[32,64,128], bn=[32,64,128], bk=[32,64], merge=[1,2,4,8] if strategy=="S2" else [1],
                    warps=[4,8,16], dtype=DT)
    if w == "W3":
        return dict(m=[1,32,256,1024,33], k=[256,1024,4096,8192,259], op=["rms","layer"],
                    consumer=["scale","linear"], consumers=[1,2,4,8], merge=[1,2,4] if strategy=="S2" else [1],dtype=DT)
    if w == "W4":
        return dict(m=[1,32,256,33], k=[128,512,1024,2048,4096,517],
                    consumer=["weighted"] if strategy=="S4" else ["scale","weighted"], consumers=[1,2,4],
                    cut=[32,64,128,256,512] if strategy=="S4" else [32],dtype=DT)
    if w == "W5":
        return dict(batch=[1,4], seq=[128,512,1024,2048,133], d=[64,128], causal=[False,True],
                    cut=[32,64,128] if strategy=="S4" else [32],normalizer=["online","row"],dtype=DT)
    if w == "W6":
        return dict(tasks=[16,64,256], workers=[1,2,4,8,16], depth=[2,4,8,16],
                    dag=["chain","fork_join","fan_in","mixed"], schedule=["fixed","topological"],dtype=DT)
    if w == "W7":
        return dict(m=[1,8,16,32,65,128], batch=[1,2], k=[896], n=[4864], bn=[32,64],
                    merge=[1,2,4] if strategy=="S2" else [1],dtype=DT)
    return dict(batch=[1,2],seq=[16,64,128,257],phase=["prefill","decode"],d=[64],
                workers=[2,4,8] if strategy=="S5" else [1],cut=[32,64,128] if strategy=="S4" else [32],dtype=DT)


def anchors(seed, smoke=False):
    rng = random.Random(seed)
    result=[]
    for s,target in TARGET.items():
        workloads=[w for w in SUPPORT if s in SUPPORT[w]]
        for i,w in enumerate(workloads):
            spec=axes(w,s)
            pool=list(itertools.product(*spec.values()))
            rng.shuffle(pool)
            count=1 if smoke else target//len(workloads)+(i<target%len(workloads))
            if smoke:
                row=[xs[0] for xs in spec.values()]
                if "merge" in spec and s=="S2": row[list(spec).index("merge")]=2
                if w=="W3" and s=="S2": row[list(spec).index("m")]=32
                if w=="W7" and s=="S2": row[list(spec).index("m")]=32
                if w=="W8" and s=="S4": row[list(spec).index("seq")]=64
                if w=="W6": row[list(spec).index("dag")]="fork_join"
                pool=[tuple(row)]
            # Greedy spread over marginal parameter levels. Never duplicate a configuration.
            seen={k:{} for k in spec}
            valid_count=0;controls=set();unique=set()
            while pool and valid_count<count:
                n=min(96,len(pool))
                j=min(range(n),key=lambda j:sum(seen[k].get(v,0) for k,v in zip(spec,pool[j])))
                vals=pool.pop(j)
                p=dict(zip(spec,vals))
                if w=="W6" and p["dag"] in ("chain","fan_in"):
                    p["depth"]=p["tasks"] if p["dag"]=="chain" else 2
                fingerprint=digest(p)
                if fingerprint in unique:continue
                unique.add(fingerprint)
                case={"workload":w,"strategy":s,"params":p}
                from .compiler import native_plan as plan
                from .util import Rejected
                try:plan(case,True)
                except Rejected as e:
                    if e.status in controls:continue
                    controls.add(e.status)
                else:valid_count+=1
                for k,v in p.items(): seen[k][v]=seen[k].get(v,0)+1
                case["id"]=w+"-"+s+"-"+digest(case)[:14]
                result.append(case)
    return result

STRATEGIES=tuple(TARGET)

def expanded_case(anchor, strategy):
    w=anchor["workload"]
    params=dict(anchor["params"])
    original=strategy==anchor["strategy"]
    if not original:
        # Change only the new pass's knob. All functional inputs and the seed stay paired.
        if strategy=="S2":
            choices=(2,4,8) if w=="W2" else (2,4)
            params["merge"]=choices[int(anchor["id"][-4:],16)%len(choices)]
        if strategy=="S4":
            if w=="W6":choices=(2,4)
            elif w=="W2":choices=(2*params["bk"],4*params["bk"])
            elif w in ("W3","W4","W7"):choices=(32,64,128,256)
            else:choices=(32,64,128)
            params["cut"]=choices[int(anchor["id"][-4:],16)%len(choices)]
    core=dict(workload=w,strategy=strategy,params=params)
    ident=anchor["id"] if original else w+"-"+strategy+"-"+digest([anchor["id"],core])[:14]
    return dict(**core,id=ident,anchor_id=anchor["id"],anchor_strategy=anchor["strategy"])

def matrix(seed,smoke=False):
    return [expanded_case(a,s) for a in anchors(seed,smoke) for s in STRATEGIES]
