"""Offline target compilation. Record the actual compiler; no launch, no timing claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,triton
from hashlib import sha256
from stage_a.toolchain import require_triton
require_triton()
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from stage_b import kernels as g,model_kernels as m

def compile_one(fn,pt,values,warps=4):
    names=fn.arg_names
    import inspect
    values=list(values)
    for name in names[len(pt)+len(values):]:
        default=inspect.signature(fn.fn).parameters[name].default
        assert default is not inspect.Parameter.empty,name
        values.append(default)
    assert len(names)==len(pt)+len(values)
    obj=triton.compile(ASTSource(fn,dict(zip(names,pt+["constexpr"]*len(values))),dict(zip(names[len(pt):],values))),
                       target=GPUTarget("cuda",120,32),options={"num_warps":warps,"num_stages":2,"enable_fp_fusion":False})
    print(json.dumps(dict(kernel=fn.__name__,compiler=triton.__version__,parameters=values,signature=pt,num_warps=warps,
                          cubin_sha256=sha256(obj.asm["cubin"]).hexdigest(),shared=obj.metadata.shared)),flush=True)

if __name__=="__main__":
    for f in ("*fp16","*bf16"):
        a="*fp32";i="*i32"
        compile_one(g.element,[f]*4+[a],[259,128,True,1,2,2],1)
        compile_one(g.attention,[f]*4+[a,a],[133,64,256,64,True,False,True,True,False])
        compile_one(m.dispatch,[f]*13+[i,i,i,i,"*i64"],[16,32,6,1,1,1,32,False,False,32,True,True,0,1,False],1)
        compile_one(g.gemm,[f]*4+[a],[65,515,512,32,128,32,True])
        compile_one(g.norm,[f]*5+[a],[33,259,512,2,2,True,True,True,True])
        for hoist in (False,True):
            compile_one(g.soft,[f]*4+[a],[517,1024,2,True,True,128,hoist])
            compile_one(g.attention,[f]*4+[a,a],[133,64,256,64,True,True,True,True,hoist])
        for mask in (1,2,3):compile_one(m.mlp,[f]*9+[a],[33,896,4864,32,mask,True,32])
        for group in (1,2,4):
            nt=2*((14+group-1)//group+2)
            for hoist in (False,True):
                compile_one(m.dispatch,[f]*11+[a,a,i,i,i,i,"*i64"],[nt,128,8,16,128,128,64,True,True,32,False,False,6,group,hoist])

        compile_one(g.element,[f]*4+[a],[4099,512,False,32,2,1])
        compile_one(g.gemm,[f]*4+[a],[17,515,513,32,64,32,True,128])
        compile_one(g.norm,[f]*5+[a],[33,259,512,2,2,True,True,True,True,64])
        for local,group,cut in [(True,1,0),(True,2,32),(False,4,0)]:
            compile_one(g.soft,[f]*4+[a],[517,1024,2,True,False,cut,False,local,group,33])
            compile_one(g.attention,[f]*4+[a,a],[133,64,256,cut or 32,True,False,bool(cut),True,False,local,group])
        compile_one(m.mlp,[f]*9+[a],[33,896,4864,32,1,True,32,128])
        compile_one(m.dispatch,[f]*13+[i,i,i,i,"*i64"],[16,32,6,1,1,1,4,False,True,32,True,True,0,1,False,False],1)
        compile_one(m.dispatch,[f]*11+[a,a,i,i,i,i,"*i64"],[6,128,8,16,16,16,32,False,False,32,False,False,6,4,False,True])

        compile_one(m.dispatch,[f]*13+[i,i,i,i,"*i64"],[5,32,6,1,1,1,2,False,True,32,True,True,0,1,False,False,4,17],1)
