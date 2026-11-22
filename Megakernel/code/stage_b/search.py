"""Static finite candidate universe, path-sensitive legality, and budgeted online selectors."""
from collections import defaultdict
from itertools import combinations
from time import perf_counter
from stage_a.util import Rejected,digest
from .planner import derive,inventory,key

class Pool:
    def __init__(self,case,depth=4,only=()):
        started=perf_counter()
        self.case=case;self.depth=depth
        self.actions=tuple(a for a in inventory(case) if not only or a.strategy in only)
        self.nodes={};self.plans={};self.edges=defaultdict(list);self.paths=[];self.rejects=[]
        origin=derive(case,());self.root=key(case,origin);self.nodes[self.root]=();self.plans[self.root]=origin
        self.by_path=defaultdict(list)
        self._visit((),self.root)
        self.build_seconds=perf_counter()-started

    def _visit(self,path,parent):
        if len(path)>=self.depth:return
        for a in self.actions:
            # Reapplying an action at an occupied site cannot introduce a new definition.
            if any(b.location==a.location for b in path):continue
            seq=path+(a,)
            try:plan=derive(self.case,seq)
            except Rejected as e:
                self.rejects.append((path,parent,a,e.status,str(e)));continue
            child=key(self.case,plan)
            if child==parent:
                self.rejects.append((path,parent,a,"NO_OP","executable state unchanged"));continue
            self.paths.append((seq,parent,child))
            if child not in self.nodes or len(seq)<len(self.nodes[child]):
                self.nodes[child]=seq;self.plans[child]=plan
            edge=(a,child)
            self.by_path[path].append(edge)
            if edge not in self.edges[parent]:self.edges[parent].append(edge)
            self._visit(seq,child)

    def hash_path(self,path):
        return key(self.case,derive(self.case,path))

    def rank(self,h,dag=True):
        # Priority only; never treated as a time prediction or reported as performance.
        p=self.plans[h];c=self.case["params"]
        score=p.bounds["tile_values"]/4096+abs(p.merge-2)*.35
        if p.local:score-=1
        if p.stats:score+=.15*c.get("consumers",1)
        if p.cut:score+=.04*len(p.proof["S4"]["partition"])
        if dag and p.ddg.get("data"):
            score+=p.ddg["depth"]/min(p.workers,len(p.ddg["data"]))+.02*p.ddg["edge_count"]
        return score,h

    def pairs(self):
        return {tuple(sorted((x.strategy,y.strategy))) for seq,_,_ in self.paths if len(seq)==2 for x,y in [seq]}

def finite_oracle(pool,records,limit=None):
    ids=set(pool.nodes) if limit is None else set(limit)
    valid=[h for h in ids if records.get(h,{}).get("status")=="PASS"]
    if not valid:return None
    return min(valid,key=lambda h:(records[h]["latency_us"],h))

def select(pool,method,evaluate,budget,width,depth,*,guided=True,dag=True,fixed_param=False,fixed_order=False):
    start=perf_counter();seen={};trace=[];attempts=pruned=0;compiles=0;compile_s=0.;charged_s=0.;callback_s=0.;calls=0;specializations=0
    reached={pool.root:()}
    def observe(h):
        nonlocal compiles,compile_s,charged_s,callback_s,calls,specializations
        if h not in seen:
            t=perf_counter();value=evaluate(h);callback_s+=perf_counter()-t;seen[h]=value
            compiles+=int(value.get("compile_attempts",0)>0);calls+=value.get("compile_attempts",0);specializations+=value.get("compiled_specializations",0);compile_s+=value.get("compile_seconds",0.)
            charged_s+=value.get("process_seconds",value.get("evaluation_seconds",0.))
            trace.append(dict(plan=h,sequence=[a.json() for a in reached[h]],status=value["status"],latency_us=value.get("latency_us"),ordinal=len(seen)))
        return seen[h]
    observe(pool.root);front=[pool.root];peak=1
    for level in range(1,depth+1):
        if method=="none" or len(seen)>=budget:break
        eligible={}
        for h in front:
            if seen[h]["status"]!="PASS":continue
            history=reached[h]
            if guided:edges=pool.by_path[history];attempts+=len(edges)
            else:
                edges=[]
                for a in pool.actions:
                    attempts+=1
                    try:
                        seq=history+(a,);nh=pool.hash_path(seq)
                        if nh in pool.nodes and nh!=h:edges.append((a,nh))
                    except Rejected:pruned+=1
            for a,child in edges:
                if child in seen:continue
                if fixed_param and a!=next(x for x in pool.actions if x.location==a.location):continue
                if (method=="fixed" or fixed_order) and history and a.strategy<history[-1].strategy:continue
                eligible.setdefault(child,history+(a,))
        ordered=sorted(eligible,key=lambda h:pool.rank(h,dag))
        if method=="fixed":ordered=sorted(ordered,key=lambda h:(eligible[h],pool.rank(h,dag)))[:1]
        if not ordered:break
        # Reserve evaluations for deeper expansions; all policies pay for baseline and every probe.
        room=budget-len(seen)
        cap=room if method=="single" or level==depth else max(1,room//(depth-level+1))
        candidates=ordered[:cap]
        for h in candidates:
            reached[h]=eligible[h];observe(h)
        passed=sorted((h for h in candidates if seen[h]["status"]=="PASS"),key=lambda h:(seen[h]["latency_us"],h))
        if method=="single":break
        if method in ("greedy","fixed"):front=passed[:1]
        else:
            # One representative per distinct action-site family, then fill by measured latency.
            groups=set();front=[]
            for h in passed:
                signature=tuple(sorted(a.location for a in reached[h]))
                if signature not in groups and len(front)<width:front.append(h);groups.add(signature)
            front+=( [h for h in passed if h not in front][:width-len(front)] )
        peak=max(peak,len(front))
        if not front:break
    if method=="single":
        best=finite_oracle(pool,seen,set(seen)-{pool.root})
    elif method=="fixed":
        best=next((step["plan"] for step in reversed(trace) if step["status"]=="PASS"),None)
    else:best=finite_oracle(pool,seen)
    elapsed=perf_counter()-start
    return dict(method=method,selected=best,selected_sequence=[a.json() for a in reached[best]] if best else None,selected_latency=seen[best]["latency_us"] if best else None,budget=budget,
                evaluated_candidates=len(seen),valid_evaluations=sum(x["status"]=="PASS" for x in seen.values()),
                compiled_candidates=compiles,compile_api_calls=calls,fresh_specializations=specializations,controller_seconds=elapsed-callback_s,charged_search_seconds=charged_s+elapsed-callback_s,compile_time=compile_s,charged_evaluation_seconds=charged_s,search_time=elapsed,
                common_pool_build_seconds=pool.build_seconds,standalone_total_seconds=pool.build_seconds+charged_s+elapsed-callback_s,unique_plans=len(pool.nodes),candidate_tests=attempts,rejected_checks=pruned,
                width=width,max_depth=depth,peak_frontier=peak,trace=trace)
