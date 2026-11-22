"""Use verified local tutorial definitions only; never execute tutorial example/benchmark mains."""
import ast
import sys
import types
from pathlib import Path
from .assets import ROOT
from .util import file_hash


TUTORIAL_DEFINITIONS={
    "02":{"softmax_kernel"},
    "03":{"matmul","matmul_kernel","leaky_relu","is_cuda","get_cuda_autotune_config","get_hip_autotune_config","get_autotune_config"},
    "05":{"_layer_norm_fwd_fused"},
    "09":{"is_cuda","_matmul_launch_metadata","matmul_get_configs","_compute_pid","matmul_kernel_persistent","matmul_persistent"},
}


def tutorial_nodes(name,tree):
    number=name[:2]
    if number!="06":
        names=TUTORIAL_DEFINITIONS[number]
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes}!=names:
            raise ValueError("Missing pinned tutorial definitions: "+name)
        return nodes
    # Only the verified module prefix through the public callable is needed.
    # This includes the official autotuning conditions and excludes pytest/benchmarks.
    nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Assign,ast.If)):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="DEVICE" for t in node.targets):
                continue
            nodes.append(node)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="attention" for t in node.targets):
                if ast.unparse(node.value)!="_attention.apply":
                    raise ValueError("Unexpected FA2 callable")
                return nodes
    raise ValueError("Missing pinned upstream FA2 callable")


def tutorial(name,asset):
    import json,os
    import torch,triton,triton.language as tl
    from .toolchain import require_triton
    require_triton()
    builtin=ROOT/"third_party/triton_tutorials"/name
    path=Path(asset["tutorials"].get(name) or builtin)
    pins=json.loads((ROOT/"configs/tutorials_390.json").read_text(encoding="utf8"))["sha256"]
    if file_hash(path)!=pins[name]:raise ValueError("tutorial differs from pinned Triton 3.9.0 source: "+name)
    namekey="_stage_a_offline_390_"+name[:2]
    if namekey in sys.modules:return sys.modules[namekey]
    mod=types.ModuleType(namekey);mod.__file__=str(path)
    mod.__dict__.update(torch=torch,triton=triton,tl=tl,os=os,DEVICE=torch.device("cuda",torch.cuda.current_device()))
    if name.startswith("06"):
        from triton.tools.tensor_descriptor import TensorDescriptor
        mod.TensorDescriptor=TensorDescriptor
    nodes=tutorial_nodes(name,ast.parse(path.read_bytes()))
    sys.modules[namekey]=mod
    try:exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),mod.__dict__)
    except Exception:
        sys.modules.pop(namekey,None)
        raise
    return mod


def controls(job,asset):
    import torch
    p=job.p; f=torch.nn.functional
    if job.w=="W5" and p["seq"]%128==0:
        def fa():
            t=tutorial("06-fused-attention.py",asset)
            q,k,v=[x[:,None,:,:] for x in job.args[:3]]
            return t.attention(q,k,v,p["causal"],p["d"]**-.5,False).squeeze(1)
        yield "official_fused_attention",fa
    if job.w=="W2" and job.dt==torch.float16:
        for name,method in [("03-matrix-multiplication.py","matmul"),("09-persistent-matmul.py","matmul_persistent")]:
            def mm(name=name,method=method):
                x,w,b=job.args[:3];kernel=getattr(tutorial(name,asset),method)
                return f.gelu(kernel(x,w).float()+b.float()).to(job.dt)
            yield "official_"+method,mm
    if job.w=="W4":
        def soft():
            k=tutorial("02-fused-softmax.py",asset).softmax_kernel
            x,v,y,tmp,_=job.args
            k[(job.m,)](tmp,x,job.k,job.k,job.m,job.k,1<<(job.k-1).bit_length(),2,num_warps=4)
            z=tmp.float()[None]*v.float()[:,None]
            return (z.sum(-1) if p["consumer"]=="weighted" else z).to(job.dt)
        yield "official_softmax_consumer",soft
    if job.w=="W3" and p["op"]=="layer":
        def layer():
            x,g,v,_,tmp,st=job.args
            bias=torch.zeros_like(g)
            tutorial("05-layer-norm.py",asset)._layer_norm_fwd_fused[(job.m,)](x,tmp,g,bias,st,st[job.m:],job.k,job.k,1e-6,1<<(job.k-1).bit_length())
            z=tmp.float()[None]*v.float()[:,None]
            return (z.sum(-1) if p["consumer"]=="linear" else z).to(job.dt)
        yield "official_layernorm_consumer",layer
    if job.w in ("W3","W7"):
        def liger():
            if asset.get("liger_source") and Path(asset["liger_source"]).is_dir():
                sys.path.insert(0,asset["liger_source"])
            if job.w=="W7":
                from liger_kernel.ops.rms_norm import LigerRMSNormFunction
                from liger_kernel.ops.swiglu import LigerSiLUMulFunction
                x,g,wg,wu,wd=job.args[:5]
                z=LigerRMSNormFunction.apply(x,g,1.e-6)
                return f.linear(LigerSiLUMulFunction.apply(f.linear(z,wg),f.linear(z,wu)),wd)
            x,g,v=job.args[:3]
            if p["op"]=="rms":
                from liger_kernel.ops.rms_norm import LigerRMSNormFunction
                z=LigerRMSNormFunction.apply(x,g,1.e-6)
            else:
                from liger_kernel.ops.layer_norm import LigerLayerNormFunction
                z=LigerLayerNormFunction.apply(x,g,torch.zeros_like(g),1.e-6)
            product=z.float()[None]*v.float()[:,None]
            return (product.sum(-1) if p["consumer"]=="linear" else product).to(job.dt)
        yield "local_liger",liger


def main(reqpath,out):
    from .offline import activate
    activate()
    from .toolchain import require_triton
    version=require_triton()
    import json,torch
    from .workloads import Workload
    from .evidence import errors,measure
    from .util import save
    req=json.loads(Path(reqpath).read_text(encoding="utf8"));device=req["device"]
    from .numeric import configure_matmul_precision
    torch.cuda.set_device(device);configure_matmul_precision(torch)
    torch.cuda.set_per_process_memory_fraction(min(1.,req["max_memory_gb"]*1024**3/torch.cuda.get_device_properties(device).total_memory),device)
    job=Workload(req["case"],device,req["seed"],req["assets"])
    answer=job.reference(True); records={}
    def persist():
        save(out,{name:{**item,"triton_version":version} for name,item in records.items()})
    if req.get("probe_proton"):
        try:
            import triton.profiler as proton
            from .compiler import plan
            kernel,_=job.bind(plan(req["case"],True))
            kernel();torch.cuda.synchronize()
            target=Path(req["folder"])/"proton_b2"
            session=proton.start(str(target),backend="cupti")
            if session is None:raise RuntimeError("Proton session unavailable")
            try:
                with proton.scope("StageA_B2"):
                    kernel();torch.cuda.synchronize()
            finally:proton.finalize(session)
            output=target.with_suffix(".hatchet")
            tree=json.loads(output.read_text(encoding="utf8"))
            if not tree:raise RuntimeError("empty Proton tree")
            records["proton"]=dict(status="COLLECTED",file=output.name,sha256=file_hash(output))
        except Exception as ex:
            fatal=any(s in str(ex).lower() for s in ("device-side","illegal memory","device lost"))
            records["proton"]=dict(status="SKIPPED_OPTIONAL",reason=repr(ex),fatal_device=fatal)
            if fatal:
                persist()
                return
    for name,fn in controls(job,req["assets"]):
        try:
            check=errors(fn(),answer,job.w)
            if check["passed"]:
                raw,_=measure({name:fn},req["batches"],req["replays"],req["seed"],device,req["graph"])
                records[name]=dict(status="MEASURED",check=check,samples_us=raw[name])
            else:records[name]=dict(status="FAIL_CORRECTNESS",check=check)
        except Exception as ex:
            records[name]=dict(status="SKIPPED_OPTIONAL",reason=repr(ex))
            if any(x in str(ex).lower() for x in ("device-side","illegal memory","device lost")):
                records[name]["fatal_device"]=True
                persist()
                return
        persist()
    persist()

if __name__=="__main__":main(*sys.argv[1:])
