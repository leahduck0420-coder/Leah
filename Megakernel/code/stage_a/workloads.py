"""Immutable inputs, unfused execution, numerical oracle and plan-driven Triton dispatch."""
from math import ceil
from .compiler import power,graph

INPUT_CONTRACT_VERSION="activation_stress_fixed_model_weights_v2"
QKV_PROJECTION_CONTRACT_VERSION="native_qualified_qkv_bias_v2"

def stable_softmax_reference(logits):
    """Independent FP64 normalizer, with the large common offset removed first."""
    values=logits.double()
    centered=values-values.amax(-1,keepdim=True)
    numerator=centered.exp()
    return numerator/numerator.sum(-1,keepdim=True)


def projection_with_bias_reference(x,weight,bias,dtype,round_before_bias=True):
    projected=x.float()@weight.float().T
    if round_before_bias:
        projected=projected.to(dtype)
    return (projected.float()+bias.float()).to(dtype)


class Workload:
    def __init__(self,case,device,seed,assets,profile="normal"):
        import torch
        from .assets import load_weights
        self.t=torch
        self.case=case
        self.w=case["workload"]
        p=self.p=case["params"]
        self.dev=device
        self.dt=getattr(torch,p["dtype"])
        gen=torch.Generator(device=device).manual_seed(seed)
        def rand(*shape,stress=True):
            x=torch.randn(shape,device=device,dtype=self.dt,generator=gen)*0.18
            if profile=="extreme" and stress:
                x.reshape(-1)[::17]=4.
                x.reshape(-1)[::29]=-4.
            return x
        def empty(*shape,dtype=None):return torch.empty(shape,device=device,dtype=dtype or self.dt)
        self.empty=empty
        self.loaded=False
        self.qkv_bias_mode=None
        self.qkv_projection_qualification=None
        self.input_contract=dict(version=INPUT_CONTRACT_VERSION,profile=profile,
                                 qkv_projection=QKV_PROJECTION_CONTRACT_VERSION,
                                 qwen_synthetic_weights="same seeded weights for normal and extreme inputs",
                                 softmax_reference="FP64 shifted exp/sum with output-dtype probability boundary")
        self.output=None
        w=self.w
        self.m=p.get("m",1); self.k=p.get("k",1); self.n=p.get("n",1)
        m,k,n=self.m,self.k,self.n
        self.c=p.get("consumers",1)
        self.reset=lambda:None
        if w=="W1":
            x=rand(n*2)[::2] if profile=="stride" else rand(n)
            self.args=(x,rand(n),rand(self.c,n),empty(self.c,n),empty(n,dtype=torch.float32))
            self.output=self.args[3]
        elif w=="W2":
            self.args=(rand(m,k),rand(k,n),rand(n),empty(m,n),empty(m,n,dtype=torch.float32))
            self.output=self.args[3]
        elif w in ("W3","W4"):
            out=(self.c,m) if p["consumer"] in ("linear","weighted") else (self.c,m,k)
            x,g,v,y,tmp,stat=rand(m,k),rand(k),rand(self.c,k),empty(*out),empty(m,k),empty(2*m,dtype=torch.float32)
            self.args=(x,g,v,y,tmp,stat) if w=="W3" else (x,v,y,tmp,stat)
            self.output=y
        elif w=="W5":
            b,l,d=p["batch"],p["seq"],p["d"]
            self.args=(*[rand(b,l,d) for _ in range(3)],empty(b,l,d),empty(b,l,l,dtype=torch.float32),
                       empty(b,l,2,dtype=torch.float32))
            self.output=self.args[3]
        elif w=="W6":
            nt=p["tasks"]; self.edges=graph(p,w);self.x=rand(nt,32);self.output=empty(nt,32)
            self.dispatch_data(nt)
        else:
            if assets["qwen_status"]!="READY":raise RuntimeError("BLOCKED: Qwen config/weights unavailable")
            names=({"post_attention_layernorm.weight":(896,), "mlp.gate_proj.weight":(4864,896),
                    "mlp.up_proj.weight":(4864,896),"mlp.down_proj.weight":(896,4864)} if w=="W7" else
                   {**{"self_attn."+s+"_proj.weight":(n,896) for s,n in (("q",896),("k",128),("v",128),("o",896))},
                    **{"self_attn."+s+"_proj.bias":(n,) for s,n in (("q",896),("k",128),("v",128))}})
            keys={"model.layers.0."+key:size for key,size in names.items()}
            if assets["weights"]=="REAL_WEIGHTS":
                self.ws={name.split("layers.0.")[1]:x for name,x in load_weights(assets,keys,device,self.dt).items()}
                self.loaded=True
            else:
                self.ws={name:(torch.ones(size,device=device,dtype=self.dt) if "layernorm" in name else rand(*size,stress=False)*0.1)
                         for name,size in names.items()}
            if w=="W7":
                self.m=m=m*p["batch"]
                self.args=(rand(m,k),*(self.ws[key] for key in names),empty(m,k),empty(2,m,n),empty(m,n),empty(m,k),empty(m,dtype=torch.float32))
                self.output=self.args[-2]
            else:
                self.L=l=p["seq"] if p["phase"]=="prefill" else 1
                self.KL=kl=p["seq"]; b=p["batch"]
                self.x=rand(b,l,896)
                freq=1000000.**(-torch.arange(0,64,2,device=device,dtype=torch.float32)/64)
                angle=torch.outer(torch.arange(kl,device=device,dtype=torch.float32),freq).repeat(1,2)
                self.cos,self.sin=angle.cos().to(self.dt),angle.sin().to(self.dt)
                self.q=empty(b,l,14,64)
                self.kcache,self.vcache=rand(b,kl,2,64),rand(b,kl,2,64)
                self.oldk=self.kcache[:,:kl-l].clone();self.oldv=self.vcache[:,:kl-l].clone()
                self.yhead=empty(b,l,14,64)
                self.output=empty(b,l,896)
                self.qkv=torch.cat([self.ws["self_attn."+i+"_proj.weight"] for i in "qkv"])
                self.bias=torch.cat([self.ws["self_attn."+i+"_proj.bias"] for i in "qkv"])
                self.tmp=empty(b,l,14,kl,dtype=torch.float32)
                self.stats=empty(b,l,14,2,dtype=torch.float32)
                self.edges=graph(p,w);self.dispatch_data(len(self.edges))

    def qualify_qkv_projection(self):
        from time import perf_counter
        from .numeric import choose_bias_rounding
        started=perf_counter()
        mode=0;roles={}
        linear=self.t.nn.functional.linear
        for bit,name in enumerate(("q","k","v")):
            weight=self.ws["self_attn."+name+"_proj.weight"]
            bias=self.ws["self_attn."+name+"_proj.bias"]
            native=linear(self.x,weight,bias)
            acc=self.x.float()@weight.float().T
            single=(acc+bias.float()).to(self.dt)
            staged=(acc.to(self.dt).float()+bias.float()).to(self.dt)
            arrays=[value.detach().float().cpu().numpy() for value in (native,single,staged)]
            decision=choose_bias_rounding(*arrays)
            roles[name]=decision
            if decision["round_before_bias"]:
                mode|=1<<bit
        self.qkv_bias_mode=mode
        self.qkv_projection_qualification=dict(mode=mode,roles=roles,profile=self.input_contract["profile"],
            elapsed_seconds=perf_counter()-started,criterion="minimum projection MSE against native linear; no latency inputs")
        return self.qkv_projection_qualification

    def dispatch_data(self,n):
        t=self.t
        self.data=t.zeros((n,n),dtype=t.int32,device=self.dev)
        for i,parents in enumerate(self.edges):
            if parents:self.data[i,parents]=1
        self.state=t.empty(n,dtype=t.int32,device=self.dev)
        self.count=t.empty_like(self.state)
        self.trace=t.empty((n,3),dtype=t.int64,device=self.dev)
        self.orders={}
        def reset():
            self.state.zero_()
            self.count.zero_()
        self.reset=reset

    def reference(self,oracle=False,sdpa=False):
        t=self.t
        F=t.nn.functional
        w,p=self.w,self.p
        dt=self.dt
        if w=="W1":
            x,b,r,_,_=self.args
            v=x.float()+b.float()
            h=F.gelu(v) if oracle else v*.5*(1+t.erf(v*0.7071067811865476))
            return (h+r.float()).to(dt)
        if w=="W2":
            x,g,b,_,_=self.args
            v=x.float()@g.float()+b.float()
            return (F.gelu(v) if oracle else .5*v*(1+t.erf(v*0.7071067811865476))).to(dt)
        if w in ("W3","W4"):
            if w=="W3":
                x,g,v=self.args[:3];xf=x.float()
                if p["op"]=="layer":
                    if oracle: h=F.layer_norm(xf,[self.k],g.float(),eps=1.e-6)
                    else:
                        centered=xf-xf.mean(-1,keepdim=True)
                        h=centered*t.rsqrt(centered.square().mean(-1,keepdim=True)+1.e-6)*g.float()
                else:
                    h=(xf.double()/((xf.double()**2).sum(-1,keepdim=True)/self.k+1.e-6).sqrt()*g.double()).float() if oracle else xf*t.rsqrt(xf.square().mean(-1,keepdim=True)+1.e-6)*g.float()
            else:
                x,v=self.args[:2];xf=x.float()
                if oracle:h=t.softmax(xf,-1)
                else:
                    e=(xf-xf.amax(-1,keepdim=True)).exp()
                    h=e/e.sum(-1,keepdim=True)
            product=h.to(dt).float()[None,:,:]*v.float()[:,None,:]
            return (product.sum(-1) if p["consumer"] in ("linear","weighted") else product).to(dt)
        if w=="W6":
            values=[]
            inputs=self.x.cpu() if oracle else self.x
            for i,deps in enumerate(self.edges):
                v=inputs[i].float()
                for pred in deps:v=v+values[pred].float()*.125
                for _ in range(1+(i%7 if p["dag"]=="mixed" else 0)):v=v*.75+.03125
                values.append(v.to(dt))
            return t.stack(values).to(self.dev)
        if w=="W7":
            x,g,wg,wu,wd=self.args[:5]
            z=(x.float()*t.rsqrt(x.float().square().mean(-1,keepdim=True)+1.e-6)).to(dt)*g
            if oracle:
                gate=(z.float()@wg.float().T).to(dt)
                up=(z.float()@wu.float().T).to(dt)
                act=(gate.float()*t.sigmoid(gate.float())).to(dt)*up
                return (act.float()@wd.float().T).to(dt)
            gate=F.linear(z,wg);up=F.linear(z,wu)
            return F.linear(F.silu(gate)*up,wd)
        if w=="W5":
            q,k,v=self.args[:3];q=q[:,None];k=k[:,None];v=v[:,None]
            l=kl=p["seq"];causal=p["causal"]
        else:
            l,kl=self.L,self.KL
            if self.qkv_bias_mode is None:
                self.qualify_qkv_projection()
            def project(name,heads):
                weight=self.ws["self_attn."+name+"_proj.weight"]
                bias=self.ws["self_attn."+name+"_proj.bias"]
                bit={"q":0,"k":1,"v":2}[name]
                staged=bool(self.qkv_bias_mode&(1<<bit))
                projected=projection_with_bias_reference(self.x,weight,bias,dt,staged) if oracle else F.linear(self.x,weight,bias)
                return projected.view(p["batch"],l,heads,64)
            def rope(x):
                a,b=x.chunk(2,-1)
                return x*self.cos[-l:,None,:]+t.cat((-b,a),-1)*self.sin[-l:,None,:]
            q=rope(project("q",14)).transpose(1,2)
            cache_k=t.cat([self.oldk,rope(project("k",2))],1)
            cache_v=t.cat([self.oldv,project("v",2)],1)
            if oracle:self.expected_cache=(cache_k,cache_v)
            k=cache_k.repeat_interleave(7,2).transpose(1,2)
            v=cache_v.repeat_interleave(7,2).transpose(1,2)
            causal=True
        mask=(t.arange(kl,device=self.dev)[None,:]<=t.arange(kl-l,kl,device=self.dev)[:,None]) if causal else None
        if sdpa:
            h=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        else:
            logits=(q.float()@k.float().transpose(-1,-2))*q.shape[-1]**-.5
            if mask is not None:logits=logits.masked_fill(~mask,float("-inf"))
            probs=(stable_softmax_reference(logits) if oracle else t.softmax(logits,-1)).to(dt)
            h=(probs.float()@v.float()).to(dt)
        if w=="W5":return h.squeeze(1)
        return F.linear(h.transpose(1,2).reshape(p["batch"],l,896),self.ws["self_attn.o_proj.weight"])

    def bind(self,plan,instrument=False):
        from .all_pairs import SinglePlan
        if isinstance(plan,SinglePlan):
            from stage_b.runtime import JointWorkload
            if self.w=="W8" and self.edges!=plan.ddg["data"]:
                self.edges=plan.ddg["data"];self.dispatch_data(len(self.edges))
            return JointWorkload.bind(self,plan,instrument)
        from . import kernels as g,model_kernels as m
        p,w=self.p,self.w
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
            args=(*self.args,self.m,self.n,self.k,p["bm"],plan.tile,p["bk"],plan.local)
        elif w=="W3":
            fn=g.norm;grid=(ceil(self.m/plan.merge),)
            args=(*self.args,self.m,self.k,power(self.k),self.c,plan.merge,p["op"]=="layer",p["consumer"]=="linear",plan.local,plan.stats)
        elif w=="W4":
            fn=g.soft;grid=(self.m,)
            args=(*self.args,self.k,power(self.k),self.c,p["consumer"]=="weighted",plan.stats,plan.cut)
        elif w=="W5":
            fn=g.attention; grid=(p["batch"]*p["seq"],)
            args=(*self.args,p["seq"],p["d"],power(p["seq"]),plan.cut or 32,p["causal"],plan.stats,bool(plan.cut),p["normalizer"]=="online")
        elif w=="W7":
            fn=m.mlp;grid=(ceil(self.m/plan.tile),)
            args=(*self.args,self.m,self.k,self.n,p["bn"],plan.local,plan.stats,plan.tile)
        else:
            deps=plan.ddg["execution"];nt=len(deps);mode=plan.scheduling
            if mode not in self.orders:
                order=self.t.zeros_like(self.data)
                for i,parents in enumerate(deps):
                    if parents:order[i,parents]=1
                self.orders[mode]=order
            fn=m.dispatch;grid=(p.get("workers",1),)
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
                  self.qkv_bias_mode if w=="W8" else 0)
        def launch():
            self.reset()
            return fn[grid](*args,num_warps=warps,num_stages=2,enable_fp_fusion=False)
        return launch,grid
