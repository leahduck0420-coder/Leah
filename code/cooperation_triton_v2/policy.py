"""Conservative eligibility, local empirical cost model, cached dispatch.

All training/preparation is explicit and outside steady-state dispatch.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path
from .specs import CONFIGS, config_id


def physical(x):
    return max(16,1<<(max(1,x)-1).bit_length())


def regions(m,n,bm,bn):
    rows=[(bm,m//bm)]+([(m%bm,1)] if m%bm else [])
    cols=[(bn,n//bn)]+([(n%bn,1)] if n%bn else [])
    return [(vm,vn,a*b) for vm,a in rows for vn,b in cols if a*b]


def geometry(m,n,k,cfg):
    p,bm,bn,bk=cfg
    if tuple(cfg) not in CONFIGS or min(m,n,k)<1:
        raise ValueError('unsupported configuration')
    tc=simt=useful_simt=masked=ctas=layout=0
    ck=k//bk*bk; rk=k-ck
    for vm,vn,count in regions(m,n,bm,bn):
        if p=='padding':
            t=bm*bn*math.ceil(k/bk)*bk; f=u=0
        elif p=='small_mma':
            t=physical(vm)*physical(vn)*(ck+(physical(rk) if rk else 0));f=u=0
        elif p=='hybrid_k':
            lanes=physical(vm)*physical(vn)
            t=lanes*ck;f=lanes*rk;u=vm*vn*rk
            layout+=count*lanes*bool(rk)
        else:
            cm,cn=vm//16*16,vn//16*16
            core=cm*cn; fringe=vm*vn-core
            lanes=physical(cm)*physical(cn) if core else 0
            edge=max(32,1<<(fringe-1).bit_length()) if fringe else 0
            t=lanes*ck;f=lanes*rk+edge*k;u=core*rk+fringe*k
            layout+=count*lanes*bool(rk)
        tc+=count*t;simt+=count*f;useful_simt+=count*u
        masked+=count*(t+f-vm*vn*k);ctas+=count
    return dict(tc_slots=tc,simt_slots=simt,useful_simt_macs=useful_simt,
                useful_macs=m*n*k,masked_slots=masked,ctas=ctas,layout_elements=layout)


def eligibility(case,cfg,unsafe=False):
    p,bm,bn,bk=cfg
    if tuple(cfg) not in CONFIGS:
        return False,'unsupported configuration'
    if unsafe or p in ('padding','small_mma'):
        return True,'eligible'
    # No unbounded pure-SIMT K loops for narrow or short-K problems.
    if case.k<bk:
        return False,'no TC K core'
    if p=='hybrid_k':
        if not 1<=case.k%bk<=4:
            return False,'hybrid_k restricted to K remainder 1..4'
        return True,'K-only SIMT; spatial edge remains TC'
    if min(case.groups)<16 or case.n<16:
        return False,'no TC spatial core in at least one group'
    if case.k%bk>4:
        return False,'large K tail'
    gs=[geometry(m,case.n,case.k,cfg) for m in case.groups]
    fraction=sum(g['useful_simt_macs'] for g in gs)/sum(g['useful_macs'] for g in gs)
    if fraction==0:
        return False,'no SIMT work; redundant aligned hybrid'
    if fraction>.02:
        return False,'SIMT MAC fraction exceeds 2% guard'
    return True,'eligible bounded hybrid'


def candidates(case,unsafe=False):
    allowed=[];skipped={}
    for cfg in CONFIGS:
        ok,reason=eligibility(case,cfg,unsafe)
        if ok:allowed.append(cfg)
        else:skipped[config_id(cfg)]=reason
    return tuple(allowed),skipped


def fingerprint(env):
    h=hashlib.sha256()
    for name in ('kernels.py','runtime.py','policy.py','specs.py','benchmark.py','timing.py'):
        h.update((Path(__file__).parent/name).read_bytes())
    return hashlib.sha256(json.dumps({'environment':env,'source':h.hexdigest()},sort_keys=True).encode()).hexdigest()


def coordinates(case,cfg):
    _,bm,bn,bk=cfg
    # Per-configuration coordinates; distinguishes grouped workload and remainder.
    return [math.log2(max(case.groups)),math.log2(case.n),math.log2(case.k),
            math.log2(sum(case.groups)),math.log2(len(case.groups)),
            max(m%bm/bm for m in case.groups),case.n%bn/bn,case.k%bk/bk]


def fit(samples,identity):
    """Local measured cost table; no common linear coefficient forcing one Tile.

    Keep all eligible (Case, config, latency_us) calibration observations.
    """
    by={}
    for case,cfg,latency in samples:
        key=f'{case.op}:{case.ep}:{config_id(cfg)}'
        by.setdefault(key,[]).append({'x':coordinates(case,cfg),'us':latency,'case_key':case.key})
    return dict(schema=2,identity=identity,kind='bounded_log_knn',strata=by,
                training_keys=sorted({c.key for c,_,_ in samples}),max_octave_distance=1.0)


def predict(case,cfg,model):
    points=model['strata'].get(f'{case.op}:{case.ep}:{config_id(cfg)}',[])
    q=coordinates(case,cfg)
    near=[]
    for point in points:
        x=point['x']
        # An OOD candidate is not extrapolated, even if its estimated cost looks low.
        if max(abs(a-b) for a,b in zip(q[:5],x[:5]))>model['max_octave_distance']:
            continue
        d=sum((a-b)**2 for a,b in zip(q[:5],x[:5]))+0.25*sum((a-b)**2 for a,b in zip(q[5:],x[5:]))
        near.append((d,point['us']))
    if not near:return None
    near=sorted(near)[:3]
    weights=[1/(d+0.05) for d,_ in near]
    return math.exp(sum(w*math.log(t) for w,(_,t) in zip(weights,near))/sum(weights))


class Selector:
    def __init__(self,identity,model=None):
        if model and (model.get('schema')!=2 or model.get('identity')!=identity):
            raise ValueError('v2 model/source/environment fingerprint mismatch')
        self.model=model

    def choose_model(self,case,allowed,fallback):
        if not self.model:return fallback,'no_model_baseline_config'
        values={c:predict(case,c,self.model) for c in allowed}
        # Only compare candidates in measured support; retain the real padding fallback.
        values={c:t for c,t in values.items() if t is not None}
        if fallback not in values:return fallback,'out_of_domain_baseline_config'
        best=min(values,key=values.get)
        if values[best]>.97*values[fallback]:best=fallback
        return best,'local_model'


class Dispatcher:
    """Immutable tuple lookup only; caches callables rather than config strings."""
    def __init__(self):
        self.entries={}

    def register(self,case,fn):
        self.entries[case.dispatch_key]=fn

    def __call__(self,key):
        return self.entries[key]()


def select_discovery(discoveries,allowed,margin=.03):
    if set(discoveries)!=set(map(config_id,allowed)):
        raise ValueError('incomplete eligible candidate set')
    center={k:statistics.median(v) for k,v in discoveries.items()}
    best=min(allowed,key=lambda c:center[config_id(c)])
    pad=min((c for c in allowed if c[0]=='padding'),key=lambda c:center[config_id(c)])
    if center[config_id(best)]>(1-margin)*center[config_id(pad)]:best=pad
    return best,pad


def guard_decision(validation,margin=.03):
    """Use a separate validation phase, never holdout, for conservative fallback."""
    names=('inductor_default','triton_padding_bound')
    base=min(names,key=lambda n:statistics.median(validation[n]))
    opt=validation['optimized_dispatch']
    qualifies=all(statistics.median(opt)<(1-margin)*statistics.median(validation[n])
                  and max(opt)<min(validation[n]) for n in names)
    return ('optimized_dispatch' if qualifies else base),qualifies
