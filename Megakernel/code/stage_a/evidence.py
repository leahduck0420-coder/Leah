import ctypes
import difflib
import hashlib
import math
import re
import statistics
import random
from math import prod
from .util import save


def occupancy(binary,device):
    try:
        import torch
        driver=ctypes.CDLL("libcuda.so.1")
        handle=int(binary.function)
        blocks=ctypes.c_int()
        threads=binary.metadata.num_warps*32
        status=driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(blocks),ctypes.c_void_p(handle),
                                ctypes.c_int(threads),ctypes.c_size_t(binary.metadata.shared))
        capacity=ctypes.c_int()
        dev_id=ctypes.c_int()
        if driver.cuCtxGetDevice(ctypes.byref(dev_id)):raise RuntimeError("no CUDA context")
        if driver.cuDeviceGetAttribute(ctypes.byref(capacity),39,dev_id):raise RuntimeError("thread capacity unavailable")
        if status or not capacity.value:raise RuntimeError("occupancy query failed")
        return blocks.value*threads/capacity.value
    except (AttributeError,OSError,TypeError,ValueError,RuntimeError):return "UNAVAILABLE"


def artifact(binary,folder,grid,device):
    folder.mkdir(parents=True,exist_ok=True)
    keys=("ttir","ttgir","llir","ptx","cubin")
    hashes={}
    for k in keys:
        item=binary.asm.get(k)
        if item is None:continue
        data=item if isinstance(item,bytes) else str(item).encode()
        (folder/("kernel."+k)).write_bytes(data)
        hashes[k]=hashlib.sha256(data).hexdigest()
    ptx=binary.asm.get("ptx","")
    return dict(hashes=hashes,grid=list(grid),ctas=prod(grid),registers=getattr(binary,"n_regs",None),
                spill=getattr(binary,"n_spills",None),smem=getattr(binary.metadata,"shared",None),
                num_warps=binary.metadata.num_warps,num_stages=getattr(binary.metadata,"num_stages",None),
                theoretical_occupancy=occupancy(binary,device),
                ptx_counts={k:len(re.findall(rx,ptx)) for k,rx in {"shuffle":r"\bshfl\.","barrier":r"\bbar\.",
                "shared":r"\b(?:ld|st)\.shared","local":r"\b(?:ld|st)\.local","global":r"\b(?:ld|st)\.(?:volatile\.)?global"}.items()})


def diff(folder):
    left=folder/"B1/kernel.ttgir";right=folder/"B2/kernel.ttgir"
    if left.is_file() and right.is_file():
        (folder/"before_after.diff").write_text("".join(difflib.unified_diff(left.read_text().splitlines(True),
                                             right.read_text().splitlines(True),"B1.ttgir","B2.ttgir")),encoding="utf8")


def errors(actual,expected,workload):
    import torch
    a,b=actual.float(),expected.float()
    finite=torch.isfinite(a)&torch.isfinite(b)
    d=(a-b).abs()
    bf=actual.dtype==torch.bfloat16
    # Match the explicit low-precision storage boundaries. Separate near-zero absolute floor.
    rtol,atol={"W1":(.009,.0006),"W2":(.009,.002),"W3":(.016,.002),"W4":(.014,.0003),
               "W5":(.016,.0004),"W6":(.009,.0005),"W7":(.020,.0003),"W8":(.020,.0006)}[workload]
    if not bf:rtol*=.28;atol*=.35
    bad=(~finite)|(d>atol+rtol*b.abs())
    def number(v):
        n=float(v.item());return n if math.isfinite(n) else None
    return dict(passed=not bool(bad.any().item()),max_abs_error=number(d.max()),
                max_rel_error=number((d/b.abs().clamp_min(1e-12)).max()),
                RMS_error=number(d.square().mean().sqrt()),nan=int(a.isnan().sum()),inf=int(a.isinf().sum()),
                reference_nan=int(b.isnan().sum()),reference_inf=int(b.isinf().sum()),
                exceeded=int(bad.sum()),rtol=rtol,atol=atol)


def quantile(seq,p):
    a=sorted(seq);pos=(len(a)-1)*p;i=int(pos)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(pos-i)


def statistics_for(values,seed):
    r=random.Random(seed)
    boots=[statistics.median(r.choices(values,k=len(values))) for _ in range(1000)]
    return dict(median=statistics.median(values),p10=quantile(values,.1),p90=quantile(values,.9),
                mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else 0.,
                ci95=[quantile(boots,.025),quantile(boots,.975)],batches=len(values))


def measure(functions,batches,replays,seed,device,graph=True):
    import torch
    # Graph captures the full operation including required resets. All modes use the same timer.
    runners={}
    for name,fn in functions.items():
        for _ in range(3):fn()
        torch.cuda.synchronize(device)
        if graph:
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(replays):fn()
            runners[name]=(g.replay,1)
        else:runners[name]=(fn,replays)
    for fn,loops in runners.values():
        for _ in range(3*loops):fn()
    torch.cuda.synchronize(device)
    rng=random.Random(seed); result={n:[] for n in runners};order=[]
    for _ in range(batches):
        modes=list(runners);rng.shuffle(modes);order.append(modes)
        for n in modes:
            fn,loops=runners[n]
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(loops):fn()
            b.record();b.synchronize()
            us=a.elapsed_time(b)*1000/replays
            if not math.isfinite(us) or us<=0:raise RuntimeError("invalid CUDA event interval")
            result[n].append(us)
    return result,order


def features(case,plan,meta):
    p=case["params"];w=case["workload"];m=p.get("m",1);k=p.get("k",1);n=p.get("n",1);c=p.get("consumers",1)
    b=p.get("batch",1);l=p.get("seq",1);s=case["strategy"]
    na="NOT_APPLICABLE";un="UNAVAILABLE"
    if w=="W7":m*=b
    # Operand traffic for the transformed intermediate/hand-off only; excludes weights and cache effects.
    handoff_size={"W1":4*n,"W2":4*m*n,"W3":2*m*k,"W4":2*m*k,"W5":4*b*l*l,
                 "W7":4*m*n,"W8":4*b*(l if p.get("phase")=="prefill" else 1)*14*l}.get(w)
    if w=="W7" and s=="S3":handoff_size=2*m*k
    if w=="W7" and s=="S2":handoff_size=2*m*k
    stat_rows=m if w in ("W3","W4","W7") else b*(l if p.get("phase","prefill")=="prefill" else 1)*(14 if w=="W8" else 1)
    stat_width=2 if w in ("W4","W5","W8") or p.get("op")=="layer" else 1
    norm_scalars=m*k*c if w=="W3" else m*k*ceildiv(n,p["bn"]) if w=="W7" else m*k*c if w=="W4" else stat_rows*l
    recon=norm_scalars if s=="S3" else na
    if w=="W1":flops=0
    elif w=="W2":flops=2*m*n*k
    elif w=="W7":flops=6*m*n*k
    elif w=="W5":flops=4*b*l*l*p["d"]
    elif w=="W8":
        q=l if p["phase"]=="prefill" else 1
        flops=2*b*q*896*(896+256+896)+4*b*14*q*l*64
    else:flops=0
    # Operand-volume estimates at the logical tensor boundary (one read per operand use
    # in this definition). Cache, transaction alignment, spill and padding are excluded.
    qlen=l if p.get("phase","prefill")=="prefill" else 1
    if w=="W1":
        compulsory=2*n*(2+2*c); arithmetic=n*(1+5*c);special=n*c
    elif w=="W2":
        compulsory=2*(m*k+k*n+n+m*n);arithmetic=flops+6*m*n;special=m*n
    elif w=="W3":
        dest=c*m*(1 if p["consumer"]=="linear" else k)
        compulsory=2*(m*k+k+c*k+dest)
        arithmetic=m*k*(5 if p["op"]=="layer" else 3)+c*m*k*(2 if p["consumer"]=="linear" else 1);special=m
    elif w=="W4":
        dest=c*m*(1 if p["consumer"]=="weighted" else k)
        compulsory=2*(m*k+c*k+dest);arithmetic=m*k*(3+c*(2 if p["consumer"]=="weighted" else 1));special=m*k
    elif w=="W5":
        compulsory=8*b*l*p["d"];arithmetic=flops+3*b*l*l;special=b*l*l
    elif w=="W6":
        edges=plan.ddg.get("original_data",plan.ddg["data"]);width=32
        compulsory=2*width*(2*len(edges)+sum(map(len,edges)))
        arithmetic=width*sum(2*len(v)+2*(1+(i%7 if p["dag"]=="mixed" else 0)) for i,v in enumerate(edges));special=0
    elif w=="W7":
        compulsory=2*(2*m*k+3*n*k+k);arithmetic=flops+5*m*n+3*m*k;special=m*n+m
    else:
        compulsory=2*(2*b*qlen*896+896*(896+256+896)+1152+2*b*l*128)
        arithmetic=flops+3*b*14*qlen*l;special=b*14*qlen*l
    dag=plan.ddg
    data=dict(program_count=meta["ctas"],logical_owner_grid=plan.ownership["grid"],
              owner_coverage_elements=prod(plan.ownership["shape"]),owner_multiplicity_padded=plan.ownership["copies"],
              local_handoff_ratio=1 if s=="S1" else na,
              cross_program_dependencies=sum(map(len,dag["data"])) if "data" in dag else 0 if s=="S1" else un,
              execution_constraint_edges=dag.get("edge_count",na),
              consumer_count=c if w in ("W1","W3","W4") else un,flops_main_matmul=flops,
              logical_io_lower_bound_bytes=compulsory,nominal_main_arithmetic_ops=arithmetic,special_function_ops=special,
              baseline_intermediate_write_bytes=handoff_size if handoff_size is not None else na,
              stats_bytes=4*stat_rows*stat_width if s=="S3" else na,
              estimated_operand_bytes=norm_scalars*(4 if w in ("W5","W8") else 2) if s=="S3" else na,
              reconstruction_count=recon,reconstruction_arithmetic_ops=(norm_scalars*(3 if p.get("op")=="layer" else 2)) if s=="S3" else na,
              reduction_slices=len(plan.proof["ordered_partition"]) if "ordered_partition" in plan.proof else na,
              workers=p.get("workers",na),dag_depth=dag.get("depth",na),instrumented_cross_worker_edges=un if "data" in dag else na,
              instrumented_critical_path_ns=un if "data" in dag else na,
              ready_gap_median_ns=un if "data" in dag else na,registers=meta["registers"],smem=meta["smem"],
              spill=meta["spill"],theoretical_occupancy=meta["theoretical_occupancy"],
              l2_hit_rate=un,actual_dram_bytes=un,achieved_occupancy=un,warp_stalls=un)
    data={k:un if v is None else v for k,v in data.items()}
    sources={k:(v if v in (na,un) else "COMPILER_REPORTED" if k in ("smem","spill","registers") else
                   "DERIVED" if k=="theoretical_occupancy" else "STATIC_ESTIMATE") for k,v in data.items()}
    return data,sources


def ceildiv(a,b):return (a+b-1)//b
