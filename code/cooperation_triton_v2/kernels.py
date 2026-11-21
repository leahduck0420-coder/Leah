"""FP16 GEMM boundary, FP32 accumulation and epilogue, FP16 output.

Derived and generalized from ../tail_tile_experiment/policy_kernels.py.
Small MMA means smaller tl.dot block, not a forced smaller ISA atom.
Hybrid is sequential and compiler-layout-managed, not fragment-native assembly.
"""
import triton
import triton.language as tl


@triton.jit
def finish(ACC, Bias, Residual, C, RM, RN, MASK,
           N: tl.constexpr, EP: tl.constexpr, SCALE: tl.constexpr):
    # Match torch FP16 matmul -> float epilogue; round AFTER all K contributions.
    y = ACC.to(tl.float16).to(tl.float32) * SCALE
    if EP != 'none':
        y += tl.load(Bias + RN, MASK, 0).to(tl.float32)
    if EP == 'bias_relu_residual':
        y = tl.maximum(y, 0.0)
    if EP == 'bias_gelu_residual':
        # tanh approximation to GELU, identical formula for every policy.
        z = 0.7978845608028654 * (y + 0.044715 * y * y * y)
        y = 0.5 * y * (1.0 + 2.0 * tl.sigmoid(2.0 * z) - 1.0)
    if EP == 'bias_relu_residual' or EP == 'bias_gelu_residual':
        y += tl.load(Residual + RM * N + RN, MASK, 0).to(tl.float32)
    tl.store(C + RM * N + RN, y, MASK)


@triton.jit
def tile(A, B, C, Bias, Residual, OM, ON,
         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         AM: tl.constexpr, AK: tl.constexpr, BK_STRIDE: tl.constexpr, BN_STRIDE: tl.constexpr,
         VM: tl.constexpr, VN: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         POLICY: tl.constexpr, EP: tl.constexpr, SCALE: tl.constexpr):
    if POLICY == 'hybrid':
        CM: tl.constexpr = VM // 16 * 16
        CN: tl.constexpr = VN // 16 * 16
    else:
        CM: tl.constexpr = VM
        CN: tl.constexpr = VN
    if CM > 0 and CN > 0:
        if POLICY == 'padding':
            TM: tl.constexpr = BM
            TN: tl.constexpr = BN
        else:
            TM: tl.constexpr = triton.next_power_of_2(max(16, CM))
            TN: tl.constexpr = triton.next_power_of_2(max(16, CN))
        rm = tl.arange(0, TM)
        rn = tl.arange(0, TN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((TM, TN), tl.float32)
        if POLICY == 'padding':
            STEPS: tl.constexpr = triton.cdiv(K, BK)
        else:
            STEPS: tl.constexpr = K // BK
        for step in range(STEPS):
            kk = step * BK + rk
            a = tl.load(A + (OM + rm[:, None]) * AM + kk[None, :] * AK,
                        (rm[:, None] < CM) & (kk[None, :] < K), 0)
            b = tl.load(B + kk[:, None] * BK_STRIDE + (ON + rn[None, :]) * BN_STRIDE,
                        (rn[None, :] < CN) & (kk[:, None] < K), 0)
            acc = tl.dot(a, b, acc)
        if POLICY == 'small_mma' and K % BK != 0:
            TK: tl.constexpr = triton.next_power_of_2(max(16, K % BK))
            kk = K // BK * BK + tl.arange(0, TK)
            a = tl.load(A + (OM + rm[:, None]) * AM + kk[None, :] * AK,
                        (rm[:, None] < CM) & (kk[None, :] < K), 0)
            b = tl.load(B + kk[:, None] * BK_STRIDE + (ON + rn[None, :]) * BN_STRIDE,
                        (rn[None, :] < CN) & (kk[:, None] < K), 0)
            acc = tl.dot(a, b, acc)
        if POLICY == 'hybrid' or POLICY == 'hybrid_k':
            # No intermediate global-memory store or extra K-tail launch.
            # Triton may insert convert_layout around these operations.
            for kk in range(K // BK * BK, K):
                a = tl.load(A + (OM + rm) * AM + kk * AK, rm < CM, 0).to(tl.float32)
                b = tl.load(B + kk * BK_STRIDE + (ON + rn) * BN_STRIDE, rn < CN, 0).to(tl.float32)
                acc = tl.fma(a[:, None], b[None, :], acc)
        finish(acc, Bias, Residual, C, OM + rm[:, None], ON + rn[None, :],
               (rm[:, None] < CM) & (rn[None, :] < CN), N, EP, SCALE)
    if POLICY == 'hybrid':
        RIGHT: tl.constexpr = CM * (VN - CN)
        EDGE: tl.constexpr = VM * VN - CM * CN
        if EDGE > 0:
            EB: tl.constexpr = triton.next_power_of_2(max(32, EDGE))
            WIDTH: tl.constexpr = max(1, VN - CN)
            e = tl.arange(0, EB)
            ok = e < EDGE
            er = tl.where(e < RIGHT, e // WIDTH, CM + (e - RIGHT) // VN)
            ec = tl.where(e < RIGHT, CN + e % WIDTH, (e - RIGHT) % VN)
            value = tl.zeros((EB,), tl.float32)
            for kk in range(K):
                a = tl.load(A + (OM + er) * AM + kk * AK, ok, 0).to(tl.float32)
                b = tl.load(B + kk * BK_STRIDE + (ON + ec) * BN_STRIDE, ok, 0).to(tl.float32)
                value = tl.fma(a, b, value)
            finish(value, Bias, Residual, C, OM + er, ON + ec, ok, N, EP, SCALE)


@triton.jit
def cooperative(A, B, C, Bias, Residual,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                AM: tl.constexpr, AK: tl.constexpr, BK_STRIDE: tl.constexpr, BN_STRIDE: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                POLICY: tl.constexpr, EP: tl.constexpr, SCALE: tl.constexpr,
                BATCH: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr,
                SC: tl.constexpr, SBIAS: tl.constexpr, SR: tl.constexpr):
    z = tl.program_id(2)
    A = A + z * SA
    B = B + z * SB
    C = C + z * SC
    Bias = Bias + z * SBIAS
    Residual = Residual + z * SR
    i = tl.program_id(0)
    j = tl.program_id(1)
    # CTA-uniform decisions. VM/VN remain compile-time constants in each branch.
    if M % BM != 0 and i == M // BM:
        if N % BN != 0 and j == N // BN:
            tile(A, B, C, Bias, Residual, i*BM, j*BN, M, N, K, AM, AK, BK_STRIDE, BN_STRIDE,
                 M % BM, N % BN, BM, BN, BK, POLICY, EP, SCALE)
        else:
            tile(A, B, C, Bias, Residual, i*BM, j*BN, M, N, K, AM, AK, BK_STRIDE, BN_STRIDE,
                 M % BM, BN, BM, BN, BK, POLICY, EP, SCALE)
    else:
        if N % BN != 0 and j == N // BN:
            tile(A, B, C, Bias, Residual, i*BM, j*BN, M, N, K, AM, AK, BK_STRIDE, BN_STRIDE,
                 BM, N % BN, BM, BN, BK, POLICY, EP, SCALE)
        else:
            tile(A, B, C, Bias, Residual, i*BM, j*BN, M, N, K, AM, AK, BK_STRIDE, BN_STRIDE,
                 BM, BN, BM, BN, BK, POLICY, EP, SCALE)


# Baseline: stock Triton autotuner, user-defined four-Tile candidate space.
# This is NOT an invented universal/default Triton GEMM policy.
padding_autotuned = triton.autotune(
    configs=[triton.Config(dict(BM=m, BN=n, BK=k), num_warps=4, num_stages=2)
             for m, n, k in ((32,32,32), (64,64,32), (128,64,32), (64,128,64))],
    key=['M', 'N', 'K', 'AM', 'AK', 'BK_STRIDE', 'BN_STRIDE', 'EP', 'SCALE',
         'BATCH', 'SA', 'SB', 'SC', 'SBIAS', 'SR'],
)(cooperative)
