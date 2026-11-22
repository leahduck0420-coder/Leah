"""Explicit accumulation policy and failure-stage numerical evidence."""


def choose_bias_rounding(native,single_round,round_before_bias):
    """Qualify arithmetic compatibility; this decision uses no timing data."""
    import numpy as np
    arrays=[np.asarray(value,dtype=np.float64) for value in (native,single_round,round_before_bias)]
    if any(value.shape!=arrays[0].shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError("invalid or nonfinite QKV qualification tensors")
    single_mse=float(np.mean((arrays[0]-arrays[1])**2))
    staged_mse=float(np.mean((arrays[0]-arrays[2])**2))
    return dict(round_before_bias=staged_mse<single_mse,single_round_mse=single_mse,
                round_before_bias_mse=staged_mse,
                single_round_exact=bool(np.array_equal(arrays[0],arrays[1])),
                round_before_bias_exact=bool(np.array_equal(arrays[0],arrays[2])))


def configure_matmul_precision(torch):
    backend = torch.backends.cuda.matmul
    try:
        fp32=backend.fp32_precision
    except AttributeError:
        fp32=None
    except RuntimeError as exc:
        if "unknown attribute" not in str(exc).lower():raise
        fp32=None
    if fp32 is not None:
        backend.fp32_precision="ieee"
        result={"fp32_precision":backend.fp32_precision}
    else:
        backend.allow_tf32=False
        result={"allow_tf32":backend.allow_tf32}
    for name in ("allow_fp16_reduced_precision_reduction","allow_bf16_reduced_precision_reduction"):
        setattr(backend,name,False)
        result[name]=getattr(backend,name)
    if hasattr(backend, "allow_fp16_accumulation"):
        backend.allow_fp16_accumulation = False
        result["allow_fp16_accumulation"] = backend.allow_fp16_accumulation
    return result


def projection_diagnostics(job, actual_available):
    from .evidence import errors
    t, dt = job.t, job.dt
    linear = t.nn.functional.linear
    result = {}
    if job.w == "W7":
        x, g, wg, wu, wd = job.args[:5]
        z = (x.float() * t.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(dt) * g
        gate, up = linear(z, wg), linear(z, wu)
        gate32 = linear(z.float(), wg.float()).to(dt)
        up32 = linear(z.float(), wu.float()).to(dt)
        native_silu = t.nn.functional.silu(gate)
        explicit_silu = (gate.float() * t.sigmoid(gate.float())).to(dt)
        activation = native_silu * up
        result.update(gate_native_vs_fp32=errors(gate, gate32, job.w),
                      up_native_vs_fp32=errors(up, up32, job.w),
                      silu_native_vs_explicit=errors(native_silu, explicit_silu, job.w),
                      down_native_vs_fp32=errors(linear(activation, wd), linear(activation.float(), wd.float()).to(dt), job.w))
    elif job.w == "W8":
        w, bias = job.ws["self_attn.v_proj.weight"], job.ws["self_attn.v_proj.bias"]
        native = linear(job.x, w, bias)
        fp32 = linear(job.x.float(), w.float(), bias.float()).to(dt)
        rounded_bias = (linear(job.x.float(), w.float()).to(dt) + bias).to(dt)
        cpu64 = linear(job.x.cpu().double(), w.cpu().double(), bias.cpu().double()).to(dt).to(job.dev)
        result.update(v_native_vs_fp32=errors(native, fp32, job.w),
                      v_native_vs_round_before_bias=errors(native, rounded_bias, job.w),
                      v_native_round_before_bias_bitwise=bool(t.equal(native,rounded_bias)),
                      v_native_vs_fp64=errors(native, cpu64, job.w),
                      v_round_before_bias_vs_fp64=errors(rounded_bias, cpu64, job.w))
        if actual_available:
            actual = job.vcache[:, -job.L:].reshape(native.shape)
            result.update(v_kernel_vs_native=errors(actual, native, job.w),
                          v_kernel_vs_fp32=errors(actual, fp32, job.w),
                          v_kernel_vs_fp64=errors(actual, cpu64, job.w),
                          v_kernel_vs_round_before_bias=errors(actual, rounded_bias, job.w))
    return result
