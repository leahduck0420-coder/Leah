"""Prefix-by-prefix verification over actual Stage A transformations and executable locations."""
from dataclasses import dataclass, asdict
from itertools import product
from math import ceil, prod
from stage_a.compiler import native_plan as single, power, tiled_tensor, tile_map, handoff, check_slices, slices, graph, levels, order_constraints
from stage_a.ownership import OwnershipMap
from stage_a.legality import validate_stats
from stage_a.util import digest, Rejected

@dataclass(frozen=True,order=True)
class Action:
    strategy:str
    location:str
    parameter:object

    def json(self):return asdict(self)

def inventory(case):
    w,p=case["workload"],case["params"]
    sites={"W1":["bias_to_gelu","element_tiles"],"W2":["dot_to_bias","output_columns","dot_reduction"],"W3":["norm_to_consumer","row_block","norm_statistics","stats_to_rebuild","norm_reduction"],
           "W4":["probability_handoff","softmax_rows","probability_statistics","weighted_reduction"],"W5":["probability_handoff","query_rows","probability_statistics","pv_reduction"],
           "W6":["task_groups","predecessor_reduction","task_order"],"W7":["gate_to_swiglu","up_to_swiglu","both_to_swiglu","mlp_rows","rms_statistics","down_reduction"],
           "W8":["head_groups","probability_handoff","probability_statistics","pv_reduction","task_order"]}[w]
    acts=[]
    for site in sites:
        if site in ("probability_handoff","bias_to_gelu","dot_to_bias","norm_to_consumer","stats_to_rebuild","gate_to_swiglu","up_to_swiglu","both_to_swiglu"):
            s,vs="S1",[1]
        elif site in ("task_groups","element_tiles","softmax_rows","query_rows","output_columns","row_block","mlp_rows","head_groups"):
            s,vs="S2",[2,4,8] if w=="W2" else [2,4]
        elif site.endswith("statistics"):
            s,vs="S3",[p.get("op","rms" if w=="W7" else "softmax")]
        elif site in ("down_reduction","norm_reduction","dot_reduction","predecessor_reduction"):
            s="S4"
            if w=="W6":vs=[2,4]
            elif w=="W2":vs=[2*p["bk"],4*p["bk"]]
            else:vs=[32,64,128,256]
        elif site in ("weighted_reduction","pv_reduction"):
            if w=="W4" and p["consumer"]!="weighted":continue
            s,vs="S4",[32,64,128,256] if w=="W4" else [32,64,128]
            if w=="W4" and p.get("cut")==512:vs[-1]=512
        else:s,vs="S5",sorted({1,2,4,8,p.get("workers",1)}) if w=="W6" else [1,2,4,8]
        acts.extend(Action(s,site,v) for v in vs)
    return tuple(acts)

@dataclass
class Executable:
    tile:int
    merge:int
    local:int
    stats:bool
    cut:int
    scheduling:str
    workers:int
    head_group:int
    ownership:dict
    ddg:dict
    proof:dict
    bounds:dict
    buffers:list
    def json(self):return asdict(self)

def seed_case(c,s,**changes):
    return dict(workload=c["workload"],strategy=s,params={**c["params"],**changes})

def head_dag(batch, factor):
    groups=ceil(14/factor);stride=groups+2
    return [[] if j==0 else list(range(i*stride+1,i*stride+groups+1)) if j==groups+1 else [i*stride]
            for i in range(batch) for j in range(stride)]

def rejection(status,message):raise Rejected(status,message)

def derive(case,path):
    p,w=case['params'],case['workload'];base=single(case,False)
    active={};sites=set()
    for a in path:
        if a not in inventory(case):rejection("INVALID_REJECTED","location or parameter outside domain")
        if a.location in sites:rejection("NO_OP","site already transformed")
        if a.location=="stats_to_rebuild" and "S3" not in active:rejection("INVALID_REJECTED","reconstruction value has no defining operation yet")
        sites.add(a.location);active.setdefault(a.strategy,[]).append(a)
    if sites.intersection({"norm_to_consumer","stats_to_rebuild"})=={"norm_to_consumer","stats_to_rebuild"}:
        rejection("NO_OP","both actions would remove the same statistics publication")
    if "both_to_swiglu" in sites and sites.intersection({"gate_to_swiglu","up_to_swiglu"}):
        rejection("NO_OP","overlapping projection handoff")
    if "probability_handoff" in sites and "S3" in active:
        rejection("NO_OP","S3 already removed the probability materialization targeted by S1")
    merge=1;heads=1;local=0;cut=0;stats="S3" in active;workers=p.get("workers",1)
    owner=base.ownership;ddg=dict(base.ddg);proofs={};s2=active.get("S2",[])
    if s2:
        f=s2[0].parameter
        if w=="W8":
            heads=f
            cover=[head for g in range(ceil(14/f)) for head in range(g*f,min(14,(g+1)*f))]
            if cover!=list(range(14)):rejection("INVALID_REJECTED","head assignment not unique")
        elif w=="W6":
            merge=f
            owner=tiled_tensor((p["tasks"],32),(f,32),p["dtype"]).to_json()
        elif w in ("W1","W4","W5"):
            if w=="W1" and p["layout"]=="cross_program":
                rejection("INVALID_REJECTED","cross-program permutation has no local lowering")
            extent=p["n"] if w=="W1" else p["m"] if w=="W4" else p["seq"]
            base_width=base.tile if w=="W1" else 1
            if extent<=base_width:rejection("NO_OP","no occupied program bit to move")
            merge=f
            if w=="W1":owner=tile_map(extent,base_width*f,p["dtype"]).to_json()
            elif w=="W4":owner=tiled_tensor((extent,p["k"]),(f,power(p["k"])),p["dtype"]).to_json()
            else:owner=tiled_tensor((p["batch"],extent,p["seq"]),(1,f,power(p["seq"])),p["dtype"]).to_json()
        else:
            q=single(seed_case(case,"S2",merge=f),True)
            owner=q.ownership;merge=f
        proofs["S2"]=dict(factor=f,mask="t < active extent",identity="old_block = group * factor + member",tail_unique=True)
    if w=="W5":
        owner=tiled_tensor((p["batch"],p["seq"],p["seq"]),(1,merge,power(p["seq"])),p["dtype"]).to_json()
    if w=="W8":
        queries=p["seq"] if p["phase"]=="prefill" else 1
        # Flatten the two CTA-local axes to keep the program grid within Triton's three dimensions.
        # For every active index v, divmod(v, seq) recovers the exact (query, key).
        qk=queries*p["seq"]
        owner=tiled_tensor((p["batch"],14,qk),(1,heads,power(qk)),p["dtype"]).to_json()
    om=OwnershipMap.from_json(owner)
    if om.copies()!=1:rejection("INVALID_REJECTED","ambiguous ordinary output writer")
    if "S1" in active:
        if w=="W1" and p["layout"]=="cross_program":rejection("INVALID_REJECTED","nonlocal read")
        local=(int(bool(sites & {"gate_to_swiglu","both_to_swiglu"}))+
               2*int(bool(sites & {"up_to_swiglu","both_to_swiglu"}))) if w=="W7" else 1
        resource_owner=tile_map(p["m"],merge,p["dtype"]) if stats and w=="W3" else om
        proofs["S1"]=dict(witness=handoff(resource_owner,resource_owner,resource_owner.local_mask),local_bits=local,
                          value="statistics SSA" if w=="W3" and stats else "producer SSA",
                          xor={"lane":1,"warp":32}.get(p.get("layout"),0))
    if stats:
        q=single(seed_case(case,"S3"),True);syms=q.proof["statistics"]
        kind="layer" if syms[0]=="mean" else "softmax" if syms[0]=="max" else "rms"
        validate_stats(kind,True,True,len(syms),True)
        proofs["S3"]=dict(statistics=syms,input_version="immutable v0",finalized_before_reconstruction=True,
                          rounding="input dtype at original producer boundary")
    if "S4" in active:
        cut=active["S4"][0].parameter
        length=p["k"] if w in ("W2","W3","W4") else p["n"] if w=="W7" else p["tasks"] if w=="W6" else p["seq"]
        if w=="W6" and not any(len({j//cut for j in ds})>=2 for ds in graph(p,w)):
            rejection("NO_OP","every task has at most one occupied predecessor slice")
        pieces=slices(length,cut);check_slices(pieces,length)
        if len(pieces)<2:rejection("NO_OP","single active slice")
        if w in ("W5","W8") and stats and cut==32:
            rejection("NO_OP","statistics reconstruction already consumes fixed 32-key panels")
        proofs["S4"]=dict(partition=pieces,accumulation_unit=p["bk"] if w=="W2" else 1 if w in ("W3","W6") else 32,tail_mask=True)
    schedule=base.scheduling
    if w in ("W6","W8"):
        data=head_dag(p["batch"],heads) if w=="W8" else graph(p,w)
        if w=="W6" and merge>1:
            original=data
            data=[sorted({j//merge for task in range(lo,min(lo+merge,len(original))) for j in original[task] if j//merge!=lo//merge})
                  for lo in range(0,len(original),merge)]
            levels(data)
            proofs["S2"]["internal_order"]="consecutive topological task IDs, CTA barrier after each typed task write"
        if "S5" in active:
            schedule="ready";workers=active["S5"][0].parameter
        execution=order_constraints(data,schedule)
        if "S5" in active and execution==order_constraints(data,base.scheduling):rejection("NO_OP","no scheduling constraints relaxed")
        if not all(set(parents)<=set(execution[i]) for i,parents in enumerate(data)):rejection("INVALID_REJECTED","unpublished predecessor")
        depth=max(levels(execution))
        ddg=dict(data=data,execution=execution,depth=depth,edge_count=sum(map(len,execution)),
                 publication="CTA barrier / gpu release-acquire; one successful claim per task",bounded_sweeps=len(data))
        if w=="W6" and merge>1:ddg["original_data"]=original
        if "S5" in active:proofs["S5"]=dict(workers=workers,topological_depth=depth,progress_bound=len(data),single_claim=True)
    else:
        ddg.update(handoff="reconstruct" if stats else "local" if local else "materialized",
                   partition=cut,logical_tile=base.tile*merge)
    if stats and cut and w in ("W4","W5","W8"):
        stages=[a.strategy for a in path]
        ddg["statistic_read_scope"]="row" if stages.index("S3")<stages.index("S4") else "slice"
        proofs["S3"]["placement"]=ddg["statistic_read_scope"]
    if w=="W8":ddg["head_group"]=heads
    tile=base.tile*merge;warps=p.get("warps",4)
    tile_values=p["bm"]*tile if w=="W2" else tile*p["bn"] if w=="W7" else 1
    bounds=dict(tile_values=tile_values,warps=1 if w=="W6" or w=="W1" and p.get("layout")=="lane" else warps,
                resource_source="STATIC_ESTIMATE; exact resource gate during JIT",task_count=len(ddg.get("data",[])))
    if tile_values>1048576 or warps not in (1,2,4,8,16,32):rejection("INVALID_REJECTED","illegal Triton launch geometry")
    buffers=[dict(value="input.v0",writer="caller",live_until="last consumer",access="immutable"),
             dict(value="output.v1",owner=owner["grid"],writer="single owner per active coordinate")]
    if stats:buffers.append(dict(value="finalized statistics",lifetime="reduction end to final rebuild",storage="SSA" if w=="W3" and local else "row scratch"))
    if cut:buffers.append(dict(value="accumulator",lifetime="entire ordered partition",storage="CTA local"))
    if "data" in ddg:buffers.append(dict(value="task outputs",lifetime="until all successors complete",storage="published GPU buffers"))
    return Executable(tile,merge,local,stats,cut,schedule,workers,heads,owner,ddg,proofs,bounds,buffers)

def key(case,exe):
    # Only executable state, not creation order or a proof's presentation.
    return digest([case["workload"],case["params"],exe.tile,exe.merge,exe.local,exe.stats,exe.cut,
                   exe.scheduling,exe.workers,exe.head_group,exe.ownership,exe.ddg])

def original_action(case):
    s,p,w=case["strategy"],case["params"],case["workload"]
    for a in inventory(case):
        if a.strategy!=s:continue
        if s=="S1" and a.location in ("stats_to_rebuild","gate_to_swiglu","up_to_swiglu"):continue
        v={"S2":p.get("merge"),"S4":p.get("cut"),"S5":p.get("workers")}.get(s,a.parameter)
        if a.parameter==v:return a
    raise ValueError("no matching Stage A action for "+case["id"])

def decode(seq):return tuple(Action(**a) for a in seq)
