import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path
from .util import file_hash
from .toolchain import TRITON_VERSION, require_triton, toolkit_release


def command(args):
    try:
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=12)
        return p.stdout.strip() if p.returncode==0 else "UNAVAILABLE: "+p.stdout[-500:]
    except (OSError,subprocess.TimeoutExpired) as ex:return "UNAVAILABLE: "+str(ex)


def telemetry(index):
    return command(["nvidia-smi","-i",str(index),"--query-gpu=uuid,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used","--format=csv,noheader,nounits"])


def resolve_gpu_selector(device_uuid, logical_index, visible_devices=None):
    """Identify the same physical GPU in CUDA and nvidia-smi."""
    value = str(device_uuid or "").strip()
    if value.startswith(("GPU-", "MIG-")):
        return value
    if value:
        try:
            return "GPU-" + str(uuid.UUID(value))
        except ValueError:
            pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if visible:
        devices = [item.strip() for item in visible.split(",")]
        if logical_index >= len(devices) or not devices[logical_index]:
            raise ValueError("CUDA device index is outside CUDA_VISIBLE_DEVICES")
        return devices[logical_index]
    return str(logical_index)


def driver_cuda_version(banner):
    match = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*([\d.]+)", banner)
    return match.group(1) if match else "UNAVAILABLE"


def inspect(venv,device):
    env=dict(python=sys.executable,version=sys.version,platform=platform.platform(),venv=str(Path(venv).resolve()),
             prefix=sys.prefix,device=device,nvcc=command(["nvcc","--version"]),driver_banner=command(["nvidia-smi"]))
    errors=[]
    env["strategies"]={s:"REGISTERED; unavailable until preflight and case admission" for s in ("S1","S2","S3","S4","S5")}
    if Path(sys.prefix).resolve()!=Path(venv).resolve() or not (Path(sys.prefix)/"pyvenv.cfg").is_file():
        errors.append("Python prefix does not belong to requested venv")
    toolkit=toolkit_release(env["nvcc"])
    env["cuda_toolkit_release"]=".".join(map(str,toolkit)) if toolkit else "UNAVAILABLE"
    env["required_triton"]=TRITON_VERSION
    if toolkit < (12,8):errors.append("nvcc with SM120 toolkit support (CUDA 12.8 or newer) required; detected "+env["nvcc"])
    for pkg in ("torch","triton","matplotlib","numpy","safetensors","liger-kernel"):
        try: env[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:env[pkg]="NOT_AVAILABLE"
    try:
        env["triton"]=require_triton()
        import torch,triton
        from .numeric import configure_matmul_precision
        env["matmul_precision"]=configure_matmul_precision(torch)
        from triton._C import libtriton
        from triton import language as tl
        from triton import testing
        from triton.runtime.jit import JITFunction
        if not callable(getattr(JITFunction,"warmup",None)):errors.append("required JITFunction.warmup unavailable")
        env.update(torch=torch.__version__,torch_cuda_runtime=torch.version.cuda,triton=triton.__version__,
                   torch_path=torch.__file__,triton_path=triton.__file__,libtriton_sha256=file_hash(libtriton.__file__),
                   audit_entry=hasattr(getattr(getattr(libtriton,"passes",None),"ttgpuir",None),"seam_audit"),
                   cudagraph_bench=hasattr(testing,"do_bench_cudagraph"))
        env["triton_provenance"]="CUSTOM_AUDIT_ENTRY" if env["audit_entry"] else "NO_KNOWN_CUSTOM_ENTRY; stock origin unverified"
        if not re.fullmatch(r"cuda:\d+",device):raise ValueError("expected cuda:index")
        torch.cuda.set_device(device)
        p=torch.cuda.get_device_properties(device)
        selector=resolve_gpu_selector(getattr(p,"uuid",None),int(device.split(":")[1]))
        env.update(gpu=p.name,capability=list(torch.cuda.get_device_capability(device)),vram=p.total_memory,
                   sm_count=p.multi_processor_count,gpu_selector=selector,driver_state=telemetry(selector))
        env["driver_version"]=command(["nvidia-smi","-i",selector,"--query-gpu=driver_version","--format=csv,noheader"])
        if "RTX 5090" not in p.name or env["capability"]!=[12,0]:errors.append("wrong GPU or compute capability")
        api=("gather","dot","debug_barrier","atomic_cas","atomic_xchg","atomic_add","inline_asm_elementwise","erf","rsqrt")
        env["language_api"]={a:hasattr(tl,a) for a in api}
        if not all(env["language_api"].values()):errors.append("missing required language operations")
        if not errors:
            from .kernels import element
            x=torch.ones(128,device=device); y=x.clone(); scratch=x.clone()
            args=(x,x,x,y,scratch,128,128,True,0,1,1)
            options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False)
            prepared=element.warmup(*args,grid=(1,),**options)
            def launch():return element[(1,)](*args,**options)
            binary=launch();torch.cuda.synchronize()
            env["codegen_probe"]=dict(warmup_matches_launch=prepared.hash==binary.hash,
                ir_stages=[k for k in ("ttir","ttgir","llir","ptx","cubin") if binary.asm.get(k)],
                registers=binary.n_regs,spill=binary.n_spills,shared=binary.metadata.shared)
            expected=torch.nn.functional.gelu(x+x)+x
            env["sm120_probe"]=bool(prepared.hash==binary.hash and len(env["codegen_probe"]["ir_stages"])==5
                and binary.n_regs is not None and torch.allclose(y,expected,atol=1e-5,rtol=1e-5))
            if not env["sm120_probe"]: errors.append("SM120 launch failed")
            try:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):launch()
                graph.replay();torch.cuda.synchronize()
                env["cuda_graph"]=True
            except Exception as ex:env.update(cuda_graph=False,graph_error=repr(ex))
        env["strategies"]={s:"SOURCE_REGISTERED; GPU admission per case" for s in ("S1","S2","S3","S4","S5")}
    except Exception as ex:errors.append(type(ex).__name__+": "+str(ex))
    env["driver_supported_cuda"]=driver_cuda_version(env["driver_banner"])
    env["errors"]=errors
    return env
