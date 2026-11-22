import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
from .util import file_hash

ROOT=Path(__file__).resolve().parent.parent
KEYS=("model_type","hidden_size","intermediate_size","num_attention_heads","num_key_value_heads",
      "rms_norm_eps","rope_theta","hidden_act","use_sliding_window","num_hidden_layers")
TUTORIALS=("02-fused-softmax.py","03-matrix-multiplication.py","05-layer-norm.py",
           "06-fused-attention.py","09-persistent-matmul.py")


def assets(folder, model_dir=None, real=False):
    base=Path(folder)
    model=Path(model_dir) if model_dir else base/"qwen/weights"
    official=json.loads((ROOT/"configs/qwen2_5_0_5b.json").read_text())
    conf=next((p for p in (model/"config.json",base/"qwen/config/config.json") if p.is_file()),None)
    actual=json.loads(conf.read_text()) if conf else official
    mismatch=[k for k in KEYS if actual.get(k)!=official[k]]
    shards=sorted(model.glob("*.safetensors"))
    index=model/"model.safetensors.index.json"
    absent=[]
    if index.is_file():
        listed=set(json.loads(index.read_text())["weight_map"].values())
        if any(Path(n).name!=n or not n.endswith(".safetensors") for n in listed):raise ValueError("invalid shard name in weight index")
        absent=[n for n in listed if not (model/n).is_file()]
    elif len(shards)>1:
        absent=["model.safetensors.index.json"]
    ready=not mismatch and (not real or bool(conf and shards and not absent and importlib.util.find_spec("safetensors")))
    tdir=base/"triton"
    installed=importlib.util.find_spec("triton")
    paths=[tdir,ROOT/"third_party/triton_tutorials"]
    if installed and installed.origin: paths.insert(0,Path(installed.origin).resolve().parent.parent/"tutorials")
    pins=json.loads((ROOT/"configs/tutorials_390.json").read_text(encoding="utf8"))["sha256"]
    tutorials={name:next((str(f) for location in paths if location.is_dir() for f in location.rglob(name)
                          if f.is_file() and file_hash(f)==pins[name]),None) for name in TUTORIALS}
    liger=importlib.util.find_spec("liger_kernel")
    checkout=next((p.parent.parent for p in (base/"liger_kernel").rglob("liger_kernel/__init__.py")),None)
    return dict(qwen_status="READY" if ready else "BLOCKED",config=actual,
                config_path=str(conf or ROOT/"configs/qwen2_5_0_5b.json"),config_sha256=file_hash(conf or ROOT/"configs/qwen2_5_0_5b.json"),
                config_reference="Qwen/Qwen2.5-0.5B-Instruct config blob eaa56b5",config_mismatch=mismatch,
                weights="REAL_WEIGHTS" if real else "QWEN_SHAPE_SYNTHETIC_WEIGHTS",
                model_dir=str(model),weight_files={p.name:file_hash(p) for p in shards} if real else {},
                missing_shards=absent,weights_loaded=False,
                tutorials=tutorials,tutorial_sha256={n:file_hash(f) for n,f in tutorials.items() if f},
                liger_source=liger.origin if liger else str(checkout) if checkout else None,
                tutorials_optional=True,liger_optional=True)


def load_weights(info, names,device,dtype):
    from safetensors import safe_open
    if info["qwen_status"]!="READY": raise ValueError("invalid Qwen configuration or incomplete safetensors")
    folder=Path(info["model_dir"])
    found={}
    for filename in info["weight_files"]:
        with safe_open(str(folder/filename),framework="pt",device="cpu") as tensors:
            for name,shape in names.items():
                if name not in tensors.keys():continue
                if name in found: raise ValueError("duplicate tensor: "+name)
                x=tensors.get_tensor(name)
                if tuple(x.shape)!=tuple(shape):raise ValueError("wrong weight shape: "+name)
                found[name]=x.to(device=device,dtype=dtype)
    missing=set(names)-set(found)
    if missing: raise ValueError("missing layer weights: "+str(sorted(missing)))
    return found
