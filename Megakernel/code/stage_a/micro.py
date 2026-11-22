"""A0 controls, each supervised in a fresh process with a bounded lifetime."""
import argparse
from contextlib import nullcontext
import faulthandler
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .util import save


def items():
    result = [dict(id="H1." + name, family="H1", name=name)
              for name in ("gemm", "elementwise", "exp", "reduction", "triton_graph_api")]
    for size in (1 << 16, 1 << 22):
        for stride in (1, 8):
            for cold in (False, True):
                result.append(dict(id=f"H2.copy_{size}_{stride}_{int(cold)}", family="H2",
                                   name="strided_copy", size=size, stride=stride, eviction=cold))
    result.extend(dict(id="H3." + layout, family="H3", name=layout)
                  for layout in ("thread", "lane", "warp"))
    result.extend(dict(id="H5." + str(cut), family="H5", name=str(cut), cut=cut)
                  for cut in (32, 128))
    result.extend(dict(id="H6." + name, family="H6", name=name, active=active)
                  for name, active in (("fixed", False), ("ready", True)))
    return result


def run_process(command, log_path, timeout):
    """Reap the worker and its compiler children before returning on timeout."""
    started = time.monotonic()
    with Path(log_path).open("w", encoding="utf8") as log:
        print(json.dumps(command), file=log, flush=True)
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name == "posix")
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                interrupted = isinstance(error, KeyboardInterrupt)
                message = "INTERRUPTED" if interrupted else f"TIMEOUT after {timeout:g} seconds"
                print(message + "; terminating worker", file=log, flush=True)
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                reason = "worker terminated after timeout"
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    reason = f"worker PID {process.pid} did not exit after SIGKILL; inspect GPU/driver state"
                    print(reason, file=log, flush=True)
                if interrupted:
                    raise
                return dict(status="TIMEOUT", pid=process.pid, reason=reason,
                            elapsed_seconds=time.monotonic() - started)
            return dict(status="COMPLETED" if code == 0 else "FAILED", exit_code=code,
                        pid=process.pid, elapsed_seconds=time.monotonic() - started)
        except OSError as error:
            return dict(status="FAILED", reason=repr(error), elapsed_seconds=time.monotonic() - started)


def run(device, seed, graph, cap, output_dir=None, timeout_seconds=120, only_item=None):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("A0 timeout must be finite and positive")
    selected = [item for item in items() if only_item is None or item["id"] == only_item]
    if not selected:
        raise ValueError("unknown A0 item: " + str(only_item))
    context = tempfile.TemporaryDirectory(prefix="stage_a_micro_") if output_dir is None else nullcontext(str(output_dir))
    with context as directory:
        root = Path(directory).resolve()
        logs = root / "08_logs" / "micro"
        logs.mkdir(parents=True, exist_ok=True)
        rows = []
        quarantine = None
        for index, item in enumerate(selected, 1):
            if quarantine:
                rows.append(dict(family=item["family"], name=item["name"], item_id=item["id"],
                                 status="BLOCKED", reason=quarantine))
            else:
                print(f"[A0 {index}/{len(selected)}] START {item['id']} (timeout {timeout_seconds:g}s)", flush=True)
                request = logs / (item["id"] + ".request.json")
                result = logs / (item["id"] + ".result.json")
                log = logs / (item["id"] + ".log")
                result.unlink(missing_ok=True)
                save(request, dict(item=item, device=device, seed=seed, graph=graph, cap=cap,
                                   timeout_seconds=timeout_seconds, artifact_dir=str(logs / (item["id"] + ".artifacts"))))
                save(logs / "progress.json", dict(current=item["id"], completed=index - 1,
                                                  total=len(selected), state="RUNNING", log=str(log)))
                outcome = run_process([sys.executable, "-u", "-B", "-m", "stage_a.micro",
                                       str(request), str(result)], log, timeout_seconds)
                records = []
                if outcome["status"] == "COMPLETED":
                    try:
                        records = json.loads(result.read_text(encoding="utf8"))
                        if not isinstance(records, list) or not records:
                            raise ValueError("empty A0 worker result")
                        for record in records:
                            if not isinstance(record, dict) or not all(key in record for key in ("family", "name", "status")):
                                raise ValueError("invalid A0 worker result")
                    except (OSError, ValueError, TypeError) as error:
                        outcome.update(status="FAILED", reason=repr(error))
                if outcome["status"] != "COMPLETED":
                    records = [dict(family=item["family"], name=item["name"], status=outcome["status"],
                                    reason=outcome.get("reason", "A0 worker failed; inspect " + str(log)),
                                    fatal_device=True)]
                for record in records:
                    record.update(item_id=item["id"], log=str(log), elapsed_seconds=outcome["elapsed_seconds"],
                                  worker_pid=outcome.get("pid"), exit_code=outcome.get("exit_code"))
                    rows.append(record)
                if any(record.get("fatal_device") for record in records):
                    quarantine = "A0 device quarantined after " + item["id"] + "; inspect its worker log before retrying"
                statuses = ",".join(record["status"] for record in records)
                print(f"[A0 {index}/{len(selected)}] END {item['id']} {statuses} ({outcome['elapsed_seconds']:.1f}s)", flush=True)
            save(root / "00_environment" / "microbench.json", rows)
            save(logs / "progress.json", dict(last=item["id"], completed=index, total=len(selected),
                                              state="QUARANTINED" if quarantine else "COMPLETED", quarantine=quarantine))
        return rows


def measure_item(request):
    import torch
    from .evidence import measure
    from .numeric import configure_matmul_precision
    configure_matmul_precision(torch)

    item = request["item"]
    device, seed = request["device"], request["seed"]
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., request["cap"] * 1024 ** 3 / memory), device)
    base = dict(family=item["family"], name=item["name"])

    def timed(fn, extra=None):
        print("PHASE warmup / capture / CUDA Events", flush=True)
        samples, _ = measure({"test": fn}, 10, 2, seed, device, request["graph"])
        return [dict(**base, samples_us=samples["test"], status="MEASURED", **(extra or {}))]

    if item["family"] == "H1":
        a = torch.randn((512, 512), device=device, dtype=torch.float16)
        functions = dict(gemm=lambda: a @ a, elementwise=lambda: a + a,
                         exp=lambda: a.exp(), reduction=lambda: a.float().sum(-1))
        if item["name"] != "triton_graph_api":
            return timed(functions[item["name"]])
        from triton.testing import do_bench_cudagraph
        print("PHASE triton.testing.do_bench_cudagraph", flush=True)
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            median_ms = do_bench_cudagraph(lambda: a @ a, rep=20, return_mode="median")
        return [dict(family="H1", name="triton_do_bench_cudagraph_gemm",
                     median_us=float(median_ms) * 1000, status="MEASURED")]

    if item["family"] == "H2":
        size, stride, cold = item["size"], item["stride"], item["eviction"]
        if request["cap"] < .45 and cold:
            return [dict(**base, status="BLOCKED", reason="memory budget")]
        x = torch.randn(size, device=device)
        trash = torch.empty(48 << 20, device=device) if cold else None
        y = torch.empty_like(x[::stride])
        y.copy_(x[::stride])
        torch.cuda.synchronize(device)
        samples = []
        for index in range(10):
            if cold:
                trash.fill_(index)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y.copy_(x[::stride])
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        return [dict(**base, samples_us=samples, size=size, stride=stride, eviction=cold, status="MEASURED")]

    from .compiler import plan
    from .workloads import Workload
    if item["family"] == "H3":
        layout = item["name"]
        case = dict(workload="W1", strategy="S1", params=dict(n=65536, tile=256, layout=layout,
                                                            consumers=1, dtype="float16"))
    elif item["family"] == "H5":
        case = dict(workload="W4", strategy="S4", params=dict(m=32, k=517, cut=item["cut"],
                                                            consumer="weighted", consumers=1, dtype="float16"))
    else:
        case = dict(workload="W6", strategy="S5", params=dict(dag="fork_join", depth=4, tasks=16,
                                                            workers=4, schedule="fixed", dtype="float16"))
    job = Workload(case, device, seed, {})
    fn, grid = job.bind(plan(case, item.get("active", True)))
    print("PHASE compile / first kernel launch", flush=True)
    binary = fn()
    if request.get("artifact_dir"):
        folder = Path(request["artifact_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        for stage in ("ttir", "ttgir", "llir", "ptx", "cubin"):
            data = binary.asm.get(stage)
            if data is not None:
                (folder / ("kernel." + stage)).write_bytes(data if isinstance(data, bytes) else str(data).encode())
        save(folder / "metadata.json", dict(num_warps=binary.metadata.num_warps,
                                            shared=binary.metadata.shared, grid=list(grid)))
    print("PHASE first kernel synchronization", flush=True)
    torch.cuda.synchronize(device)
    if item["family"] == "H6":
        from .evidence import errors
        counts = job.count.cpu().tolist()
        states = job.state.cpu().tolist()
        if counts != [1] * len(counts) or states != [2] * len(states):
            raise RuntimeError("H6 task completion failed: counts=" + str(counts) + ", states=" + str(states))
        check = errors(job.output, job.reference(oracle=True), "W6")
        if not check["passed"]:
            raise RuntimeError("H6 numerical correctness failed: " + json.dumps(check))
        return timed(fn, dict(num_warps=binary.metadata.num_warps, task_counts=counts,
                              task_states=states, correctness=check))
    if item["family"] != "H3":
        return timed(fn)
    ptx = binary.asm["ptx"]
    shfl, shared = "shfl." in ptx, "ld.shared" in ptx
    witnessed = layout == "thread" or (shfl if layout == "lane" else shared)
    evidence = dict(ptx_shfl=shfl, ptx_shared=shared, num_warps=binary.metadata.num_warps)
    rows = timed(fn, evidence) if witnessed else [dict(**base, **evidence, status="UNSUPPORTED",
                                                     reason="requested transport absent from PTX")]
    rows.append(dict(family="H4", name=layout, status="COMPILER_REPORTED", registers=getattr(binary, "n_regs", None),
                     shared=binary.metadata.shared, spill=getattr(binary, "n_spills", None)))
    return rows


def main():
    parser = argparse.ArgumentParser(description="One isolated A0 microbenchmark")
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf8"))
    faulthandler.enable()
    faulthandler.dump_traceback_later(max(.1, request["timeout_seconds"] * .8))
    from .offline import activate
    activate()
    print("WORKER START " + request["item"]["id"], flush=True)
    try:
        from .toolchain import require_triton
        version=require_triton()
        rows = measure_item(request)
        for row in rows:row["triton_version"]=version
    except Exception as error:
        import traceback
        traceback.print_exc()
        device_error = any(text in str(error).lower() for text in
                           ("illegal memory", "device-side", "device lost", "launch timeout", "unspecified launch failure"))
        rows = [dict(family=request["item"]["family"], name=request["item"]["name"], status="FAILED",
                     reason=repr(error), fatal_device=device_error)]
    save(args.result, rows)
    faulthandler.cancel_dump_traceback_later()
    print("WORKER END " + request["item"]["id"], flush=True)


if __name__ == "__main__":
    main()
