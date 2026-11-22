"""Optional offline codegen check. No CUDA driver or measurements; prints actual compiler version."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import triton
from stage_a.toolchain import require_triton
require_triton()
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from stage_a import kernels as g, model_kernels as m
from hashlib import sha256
def compile_one(fn,pointers,values,warps=4):
    if fn is m.dispatch:
        values=list(values)
        values[1]=max(values[1],32*warps)
        if len(values)==len(fn.arg_names)-len(pointers)-1:
            values.append(0)
    types=dict(zip(fn.arg_names,pointers+["constexpr"]*len(values)))
    constants=dict(zip(fn.arg_names[len(pointers):],values))
    assert len(types)==len(fn.arg_names)
    obj=triton.compile(ASTSource(fn,types,constants),target=GPUTarget("cuda",120,32),options={"num_warps":warps,"num_stages":2,"enable_fp_fusion":False})
    print(json.dumps(dict(fn=fn.__name__,compiler=triton.__version__,signature=types,constants=values,num_warps=warps,
                          ptx_shfl="shfl." in obj.asm["ptx"],ptx_shared="ld.shared" in obj.asm["ptx"],
                          sha=sha256(obj.asm["cubin"]).hexdigest())),flush=True)

if __name__=="__main__":
    f="*fp16"; a="*fp32"; i="*i32"
    for shift,warps in ((0,4),(1,1),(32,4)):
        compile_one(g.element,[f,f,f,f,a],[65536,256,True,shift,1,1],warps)
    for local in (False,True):
        compile_one(g.element,[f,f,f,f,a],[4099,128,local,32,2,2])
        compile_one(g.gemm,[f,f,f,f,a],[65,515,512,32,64 if local else 32,32,local])
    for local,stats in [(False,False),(True,False),(False,True)]:
        compile_one(g.norm,[f]*5+[a],[33,259,512,2,2,True,True,local,stats])
        compile_one(m.mlp,[f]*9+[a],[33,896,4864,64,local,stats,16 if local or stats else 32])
    for stats,cut in [(False,0),(True,0),(False,64)]:
        compile_one(g.soft,[f,f,f,f,a],[517,1024,2,True,stats,cut])
        for online in (False,True):
            compile_one(g.attention,[f,f,f,f,a,a],[133,64,256,cut or 32,True,stats,bool(cut),online])
    ptr=[f]*11+[a,a,i,i,i,i,"*i64"]
    for kind in (6,8):
        for trace in (False,True):
            compile_one(m.dispatch,ptr,[16,16,kind,1,16,16,32,False,False,32,True,trace])
    bf="*bf16"
    for stats,cut,l,kl in [(True,0,16,16),(False,64,1,128),(False,64,128,128)]:
        compile_one(m.dispatch,[bf]*11+[a,a,i,i,i,i,"*i64"],[32,32,8,l,kl,1<<(kl-1).bit_length(),cut or 32,stats,bool(cut),32,False,False])
    for stats in (False,True):
        compile_one(g.norm,[bf]*5+[a],[32,8192,8192,4,1,False,False,False,stats])
        compile_one(g.attention,[bf,bf,bf,bf,a,a],[512,128,512,32,False,stats,False,True])
        compile_one(m.mlp,[bf]*9+[a],[65,896,4864,32,False,stats,16])
