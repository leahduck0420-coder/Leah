"""Single-pass driver. The lowering consumes the returned fields directly."""
from dataclasses import dataclass, asdict
from math import ceil
from .ownership import OwnershipMap, LinearSystem, bit_width
from .util import Rejected, digest


def power(n):
    return 1 << (n-1).bit_length()


def tile_map(n, tile, dtype):
    grid=ceil(n/tile)
    bits=bit_width(n)
    cols=tuple(1 << (bit_width(tile)+i) for i in range(bit_width(grid)))
    return OwnershipMap((grid,), (n,), dtype, cols, min(tile-1,(1<<bits)-1))


def tiled_tensor(shape,tiles,dtype):
    """Concatenate active axis projections without folding row/column tail masks."""
    columns=[]; mask=0;shift=0;grid=[]
    for n,tile in zip(shape,tiles):
        axis=tile_map(n,tile,dtype)
        columns.extend(col<<shift for col in axis.columns)
        mask|=axis.local_mask<<shift
        shift+=bit_width(n);grid.append(axis.grid[0])
    return OwnershipMap(tuple(grid),tuple(shape),dtype,tuple(columns),mask)


def handoff(write,read,local_mask,valid_version=True):
    if not valid_version or write.grid!=read.grid or write.shape!=read.shape:
        raise Rejected("INVALID_REJECTED","incompatible instance, domain or lifetime")
    local=[1<<i for i in range(write.tensor_bits) if write.local_mask>>i&1]
    requested=[1<<i for i in range(read.tensor_bits) if local_mask>>i&1]
    system=LinearSystem(local)
    result=[system.reduce(x) for x in [*(a^b for a,b in zip(write.columns,read.columns)),*requested]]
    if any(residual for residual,_ in result):
        raise Rejected("INVALID_REJECTED","demand is outside same-program local image")
    return [w for _,w in result]


def slices(length,width):
    if min(length,width)<1: raise Rejected("INVALID_REJECTED","invalid reduction domain")
    return [(i,min(i+width,length)) for i in range(0,length,width)]


def check_slices(parts,length):
    pos=0
    for a,b in parts:
        if a!=pos or b<=a: raise Rejected("INVALID_REJECTED","missing, duplicated or reordered contribution")
        pos=b
    if pos!=length: raise Rejected("INVALID_REJECTED","incomplete reduction")


def graph(p,w):
    if w=="W8":
        b=p["batch"]
        # Per batch: projection+RoPE, 14 independently ready query heads, output projection.
        return [[] if j==0 else [16*i] if j<15 else list(range(16*i+1,16*i+15))
                for i in range(b) for j in range(16)]
    n=p["tasks"]; depth=min(p["depth"],n)
    if p["dag"]=="chain": return [[] if i==0 else [i-1] for i in range(n)]
    if p["dag"]=="fan_in": return [[] for _ in range(n-1)]+[list(range(n-1))]
    levels=[list(range(j*n//depth,(j+1)*n//depth)) for j in range(depth)]
    deps=[]
    for level,items in enumerate(levels):
        prev=levels[level-1] if level else []
        for j,task in enumerate(items):
            if not prev: ds=[]
            elif p["dag"]=="fork_join": ds=prev if level%2==0 else [prev[j%len(prev)]]
            else: ds=sorted({prev[j%len(prev)],prev[(3*j+1)%len(prev)]})
            deps.append(ds)
    return deps


def levels(deps):
    depth=[]
    for i,row in enumerate(deps):
        if any(p<0 or p>=i for p in row) or len(set(row))!=len(row):
            raise Rejected("INVALID_REJECTED","cyclic, duplicated or non-topological predecessor")
        depth.append(1+max((depth[p] for p in row),default=0))
    return depth


def order_constraints(deps,mode):
    if mode=="ready": return deps
    if mode=="fixed": return [sorted(set(row+([i-1] if i else []))) for i,row in enumerate(deps)]
    depth=levels(deps)
    return [sorted(set(row+[j for j in range(i) if depth[j]==depth[i]-1])) for i,row in enumerate(deps)]


@dataclass(frozen=True)
class Plan:
    strategy:str
    site:str
    tile:int
    merge:int
    local:bool
    stats:bool
    cut:int
    scheduling:str
    proof:dict
    ownership:dict
    ddg:dict

    def json(self):
        return asdict(self)


def native_plan(case,enabled):
    w,p=case["workload"],case["params"]
    s=case["strategy"] if enabled else "none"
    from .matrix import SUPPORT
    if s!="none" and s not in SUPPORT[w]: raise Rejected("UNSUPPORTED","no lowering for this rule at this site")
    site={"W1":"bias_gelu","W2":"dot_epilogue","W3":"norm_consumer","W4":"softmax_consumer",
          "W5":"attention_prob_pv","W6":"task_ready","W7":"norm_gate_down","W8":"qkv_head_out"}[w]
    if w=="W7":
        site={"S1":"gate_up_to_swiglu","S2":"mlp_row_programs","S3":"rms_to_gate_up"}.get(case["strategy"],site)
    if w=="W8":
        site="qkv_head_out_task_graph" if case["strategy"]=="S5" else "per_head_prob_to_pv"
    extent={"W1":p.get("n"),"W2":p.get("n"),"W3":p.get("m"),"W4":p.get("k"),
            "W5":p.get("seq"),"W6":p.get("tasks"),"W7":p.get("m",1)*p.get("batch",1),"W8":p.get("seq")}[w]
    base=16 if w=="W7" else p.get("tile",p.get("bn",1)) if w in ("W1","W2") else 1
    merge=p.get("merge",1) if s=="S2" else 1
    if merge not in (1,2,4,8): raise Rejected("INVALID_REJECTED","non-binary merge")
    if s=="S2" and (merge==1 or extent<=base): raise Rejected("NO_OP","no occupied program bit migrated")
    old=tile_map(extent,base,p["dtype"]); new=tile_map(extent,base*merge,p["dtype"])
    proof={"domain":[0,extent],"mask":"coordinate < extent","input_version":"immutable v0","source_tile":base}
    if s=="S1":
        if p.get("layout")=="cross_program": raise Rejected("INVALID_REJECTED","consumer crosses CTA boundary")
        proof["local_coordinate_witness"]=handoff(old,old,old.local_mask)
        proof["local_xor_offset"]={"lane":1,"warp":32}.get(p.get("layout"),0)
    if s=="S2":
        retained=min(extent,base*merge)
        for x in {0,retained-1,extent-1,max(0,base-1),min(base,extent-1)}:
            if old.owners((x,))!=((x//base,),) or new.owners((x,))!=((x//(base*merge),),):
                raise Rejected("INVALID_REJECTED","active-domain ownership changed unexpectedly")
        proof.update(migrated_bit_count=bit_width(merge),contribution_order="unchanged",
                     index_identity="old_tile=merged_tile*factor+member; disjoint tail-masked union")
    if s=="S3":
        rule=p.get("op", "rms" if w=="W7" else "softmax")
        proof["statistics"]={"rms":["inv_rms"],"layer":["mean","inv_std"],"softmax":["max","sum_exp"]}[rule]
        proof["original_version_live"]=True
    cut=p.get("cut",32) if s=="S4" else 0
    if cut:
        parts=slices(extent,cut);check_slices(parts,extent)
        if len(parts)==1: raise Rejected("NO_OP","single reduction slice")
        proof["ordered_partition"]=parts
    schedule="ready" if s=="S5" else p.get("schedule","fixed")
    if w in ("W6","W8"):
        actual=graph(p,w); deps=order_constraints(actual,schedule)
        if s=="S5" and deps==order_constraints(actual,p.get("schedule","fixed")):
            raise Rejected("NO_OP","scheduling constraints coincide")
        ddg=dict(data=actual,execution=deps,depth=max(levels(deps)),edge_count=sum(map(len,deps)),
                 lifetime="task outputs retained until end of invocation",publication="CTA barrier + GPU release/acquire")
    else:
        ops={"W1":["bias","gelu","residual"],"W2":["gemm","bias","gelu"],
             "W3":["reduce","normalize","consume"],"W4":["max_sum","normalize","consume"],
             "W5":["qk","softmax","pv"],"W7":["rms","gate_up","swiglu","down"]}[w]
        ddg=dict(operators=ops,edges=list(zip(ops,ops[1:])),site=site,
                 handoff="local" if s=="S1" else "statistics" if s=="S3" else "ordered slices" if s=="S4" else "materialized",
                 lifetime="private tile scratch until last same-CTA reader",consumers=p.get("consumers",1))
    if w=="W2":
        full=tiled_tensor((p["m"],p["n"]),(p["bm"],base*merge),p["dtype"])
    elif w=="W3":
        full=tiled_tensor((p["m"],p["k"]),(merge,power(p["k"])),p["dtype"])
    elif w=="W7":
        full=tiled_tensor((extent,p["n"]),(base*merge,p["bn"]),p["dtype"])
        proof["physical_placement"]="row tile is CTA, gate panels iterate in program order"
    elif w in ("W4","W5","W8"):
        rows=p["m"] if w=="W4" else p["batch"]*p["seq"] if w=="W5" else p["batch"]*14*(p["seq"] if p["phase"]=="prefill" else 1)
        full=tiled_tensor((rows,extent),(1,power(extent)),p["dtype"])
    else:full=new
    if s=="S1":
        proof["local_coordinate_witness"]=handoff(full,full,full.local_mask)
    if s=="S3":
        proof["statistics_demand"]="project (row,column) to row; immutable source is re-read by each consumer tile"
    return Plan(s,site,base*merge,merge,s=="S1",s=="S3",cut,schedule,proof,full.to_json(),ddg)

def uses_joint_backend(case):
    from .matrix import SUPPORT
    return case["strategy"] not in SUPPORT[case["workload"]]

def plan(case,enabled):
    if enabled and case["workload"]=="W4" and case["strategy"]=="S4" and case["params"]["consumer"]!="weighted":
        raise Rejected("NOT_APPLICABLE","existing S4 consumer reduction requires a weighted sum")
    if not uses_joint_backend(case):
        return native_plan(case,enabled)
    from .all_pairs import joint_single
    return joint_single(case,enabled)
