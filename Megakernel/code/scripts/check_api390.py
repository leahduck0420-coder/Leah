"""Static signature audit against a user-supplied official 3.9.0 source checkout."""
import ast
import hashlib
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]


def check(source):
    src=Path(source)/"python/triton"
    functions={}
    provenance={}
    for path in (src/"language/core.py",src/"language/math.py",src/"language/standard.py"):
        data=path.read_bytes();provenance[path.name]=hashlib.sha256(data).hexdigest()
        for node in ast.parse(data).body:
            if isinstance(node,ast.FunctionDef):functions[node.name]={a.arg for a in node.args.args+node.args.kwonlyargs}
    version=next(ast.literal_eval(n.value) for n in ast.parse((src/"__init__.py").read_bytes()).body
                 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="__version__" for t in n.targets))
    names=set();issues=[] if version=="3.9.0" else ["expected upstream Triton 3.9.0, got "+version]
    for path in (root/"stage_a/kernels.py",root/"stage_a/model_kernels.py",root/"stage_b/kernels.py",root/"stage_b/model_kernels.py"):
        for node in ast.walk(ast.parse(path.read_bytes())):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or not isinstance(node.func.value,ast.Name) or node.func.value.id!="tl":continue
            name=node.func.attr;names.add(name)
            if name in ("static_range","range","constexpr"):continue
            if name not in functions:issues.append("missing "+name);continue
            issues.extend(name+" unknown keyword "+str(k.arg) for k in node.keywords if k.arg not in functions[name])
    jit=(src/"runtime/jit.py").read_bytes();provenance["jit.py"]=hashlib.sha256(jit).hexdigest()
    if not any(n.name=="warmup" and n.args.vararg and n.args.kwarg and "grid" in [a.arg for a in n.args.kwonlyargs] for n in ast.walk(ast.parse(jit)) if isinstance(n,ast.FunctionDef)):issues.append("missing warmup(*args, grid, **kwargs)")
    compiler=(src/"compiler/compiler.py").read_text(encoding="utf8")
    for field in ("self.function","self.n_regs","self.n_spills","self.asm"):
        if field not in compiler:issues.append("missing "+field)
    testing=ast.parse((src/"testing.py").read_bytes())
    graph=next((n for n in testing.body if isinstance(n,ast.FunctionDef) and n.name=="do_bench_cudagraph"),None)
    graph_api=graph is not None and {"fn","rep","return_mode"}<={a.arg for a in graph.args.args}
    if not graph_api:issues.append("missing do_bench_cudagraph(fn,rep,return_mode)")
    for n in ("compiler/compiler.py","testing.py","__init__.py"):
        provenance[n]=hashlib.sha256((src/n).read_bytes()).hexdigest()
    return dict(upstream_version=version,apis=sorted(names),issues=sorted(set(issues)),source_sha256=provenance,
                graph_api=graph_api)

if __name__=="__main__":
    result=check(sys.argv[1]);print(json.dumps(result,indent=2));sys.exit(bool(result["issues"]))
