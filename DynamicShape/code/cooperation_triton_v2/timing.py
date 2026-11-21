"""Request and supply-gap-inclusive event timing; optional CUDA-graph control."""
import statistics
import time


def measure(fn,repeats,graph=False):
    import torch
    for _ in range(3):fn()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns()
        fn();torch.cuda.synchronize()
        samples.append((time.perf_counter_ns()-start)/1000)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):fn()
    end.record();end.synchronize()
    result=dict(request_us=statistics.median(samples),request_samples_us=samples,
                device_span_us=start.elapsed_time(end)*1000/repeats)
    if graph:
        import triton.testing
        try:
            result['graph_replay_us']=float(triton.testing.do_bench_cudagraph(fn,rep=20,return_mode='median'))*1000
        except Exception as exc:
            result['graph_replay_us']=None
            result['graph_error']=f'{type(exc).__name__}: {exc}'
    return result
