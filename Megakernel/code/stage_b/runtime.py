from math import ceil
from stage_a.compiler import power
from stage_a.workloads import Workload


class JointWorkload(Workload):
    def bind(self,plan,instrument=False):
        from . import kernels as g,model_kernels as m
        p,w=self.p,self.w
        hoist=plan.ddg.get("statistic_read_scope")=="row"
        if w=="W8" and self.qkv_bias_mode is None:
            self.qualify_qkv_projection()
        warps=p.get("warps",4)
        if w=="W6":
            # W6 tasks have 32 values. Keep readiness, claim and barriers within
            # one warp so concurrent state changes cannot split CTA warps.
            # B1 and B2 share this execution configuration.
            warps=1
        if w=="W1":
            if p["layout"]=="lane":warps=1
            fn=g.element; grid=(ceil(self.n/plan.tile),)
            args=(*self.args,self.n,plan.tile,plan.local,{"lane":1,"warp":32}.get(p["layout"],0),self.c,self.args[0].stride(0))
        elif w=="W2":
            fn=g.gemm;grid=(ceil(self.m/p["bm"]),ceil(self.n/plan.tile))
            args=(*self.args,self.m,self.n,self.k,p["bm"],plan.tile,p["bk"],plan.local,plan.cut)
        elif w=="W3":
            fn=g.norm;grid=(ceil(self.m/plan.merge),)
            args=(*self.args,self.m,self.k,power(self.k),self.c,plan.merge,p["op"]=="layer",p["consumer"]=="linear",plan.local,plan.stats,plan.cut)
        elif w=="W4":
            fn=g.soft;grid=(ceil(self.m/plan.merge),)
            args=(*self.args,self.k,power(self.k),self.c,p["consumer"]=="weighted",plan.stats,plan.cut,hoist,plan.local,plan.merge,self.m)
        elif w=="W5":
            fn=g.attention; grid=(p["batch"]*ceil(p["seq"]/plan.merge),)
            args=(*self.args,p["seq"],p["d"],power(p["seq"]),plan.cut or 32,p["causal"],plan.stats,bool(plan.cut),p["normalizer"]=="online",hoist,plan.local,plan.merge)
        elif w=="W7":
            fn=m.mlp;grid=(ceil(self.m/plan.tile),)
            args=(*self.args,self.m,self.k,self.n,p["bn"],plan.local,plan.stats,plan.tile,plan.cut)
        else:
            deps=plan.ddg["execution"];nt=len(deps);mode=plan.scheduling
            if w=="W6" and plan.merge>1 and getattr(self,"group_factor",1)!=plan.merge:
                self.group_factor=plan.merge
                self.orders={}
                self.state=self.t.empty(nt,dtype=self.t.int32,device=self.dev)
                self.count=self.t.empty(nt+p["tasks"],dtype=self.t.int32,device=self.dev)
                self.trace=self.t.empty((nt,3),dtype=self.t.int64,device=self.dev)
                def reset_group():
                    self.state.zero_()
                    self.count.zero_()
                self.reset=reset_group
            if mode not in self.orders:
                order=self.t.zeros((nt,nt),dtype=self.t.int32,device=self.dev)
                for i,parents in enumerate(deps):
                    if parents:order[i,parents]=1
                self.orders[mode]=order
            fn=m.dispatch;grid=(plan.workers,)
            if w=="W8":
                prefix=(self.x,self.qkv,self.bias,self.cos,self.sin,self.q,self.kcache,self.vcache,self.yhead,
                        self.ws["self_attn.o_proj.weight"],self.output,self.tmp,self.stats)
            else:prefix=(*([self.x]*10),self.output,self.x,self.x)
            # Give every physical lane a slot. Triton 3.4.0 masks redundant
            # atomic lanes without broadcasting their return values. A shorter
            # vector can otherwise make the ready branch diverge at a barrier.
            args=(*prefix,self.data,self.orders[mode],self.state,self.count,self.trace,nt,power(max(nt,32*warps)),
                  6 if w=="W6" else 8,getattr(self,"L",1),getattr(self,"KL",1),power(getattr(self,"KL",1)),
                  plan.cut or 32,plan.stats,bool(plan.cut),32,p.get("dag")=="mixed",instrument,
                  self.qkv_bias_mode if w=="W8" else 0,plan.head_group,hoist,plan.local,plan.merge if w=="W6" else 1,p.get("tasks",1))
        def launch():
            self.reset()
            return fn[grid](*args,num_warps=warps,num_stages=2,enable_fp_fusion=False)
        return launch,grid
