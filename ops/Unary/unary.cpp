
// Copyright (c) 2026 ThinkNative. All rights reserved.

/* =========================================================================
 * tn-metal: src/ops/unary.cpp
 *
 * Element-wise unary op (square/abs/floor/neg) lowering.
 *
 * Launch contract (the caller's — the op body validates nothing):
 * any rank the tensor type admits (bounded by TN_MAX_DIMS), extents
 * nonzero, and both operands share one element count — a same-shape
 * launch. Per-dim extents never enter a descriptor — the launch is
 * re-chunked linearly — so no uint16 bound applies to them. Dtype
 * rules (square: in FP8/BF16/FP32, out BF16/FP32 — the output may be
 * wider than the input; abs: in S8/S16/S32 + FP8/BF16/FP32, out = in;
 * floor: in FP8/BF16/FP32, out = in; neg: integers + FP, out = in)
 * are the vec engine's contract, not re-checked here. rsqrt/sqrt/
 * sin/cos are reserved, rejected with TN_ERR_PARAM.
 *
 * neg has no vec instruction of its own: it is synthesized as
 * vec_sub(out, zero, in) over a Constant-layout zero tile — sub takes
 * a Constant operand on either side (mul takes it on the right only),
 * and sub keeps an FP8 output legal, which mul's FP-output rule
 * excludes. The unary param carries no ovf mode, so the sub runs
 * fixed truncate: two's-complement negation, INT_MIN -> INT_MIN
 * (NumPy wrap semantics); FP is fixed saturate + RNE either way (FP
 * signed zero follows sub semantics: 0-(+0) = +0, not a sign flip).
 *
 * Scalar: run this core's share once, in the host-fitted w block the
 * launch message carries (s->B0, fitted by host/unary_split.cpp —
 * the firmware image never carries that code). The share is one
 * contiguous slice of the launch (equal shares in core order, the
 * last clamped to what remains, cores past the launch idle), striped
 * over the greatest common lane count its size allows: c_slice lanes,
 * so every chunk — tail included — is an even stripe. The walk is a
 * sequence of contiguous (cur_w, 1, c_slice) tiles, min-clamped at
 * the share's end, each just a fresh base address on the operand; the
 * resident tile set is allocated once at the block shape and lived at
 * the chunk shape — set_shape's bump guard keeps it honest. Chunks
 * cover the share exactly: every element computed, no access past
 * the end.
 *
 * Sync net (kernel.h's sync contract; tnop_sync_* plumbing from
 * common.h, USYNC_OFF). Both tiles cross an engine boundary, so both
 * are double-buffered (ldesc[k][2]); chunk i uses buffer i&1. The
 * schedule is software-pipelined one chunk ahead: chunk i+1's load
 * issues before chunk i's store — else the store's wait would
 * serialize the ldst queue behind the vec — and its vec wait issues
 * before the next ldst publish (the next prefetch's stag), so the
 * wait's scoreboard edge lands on its own publish, not a later one.
 * The load publishes load_tag[buf_idx]; the vec waits it before the
 * compute and publishes vec_done[buf_idx] (its input read and its
 * out-tile write are the same instruction); the store waits
 * vec_done. There is no store-side tag: the next chunk's single vec
 * wait on load_tag covers both of its hazards — the RAW on the
 * freshly loaded input tile, and the WAR on its out tile, whose
 * previous reader is chunk i-1's store, already complete because
 * that store precedes this load in LDST's engine order. The input
 * tile's WAR needs no wait of its own either: its previous vec
 * reader is the compute two chunks back, whose publish the
 * same-engine store wait already merged, and that store precedes
 * this load in LDST's engine order. Both transitive covers hold
 * only while the prefetch stays exactly one chunk ahead of the
 * store. neg's zero constant never crosses an engine boundary —
 * created once in the prologue, it joins no tag traffic and takes
 * no per-chunk update.
 * =========================================================================
 */
#include "ops/ops.h"
#include "tndef.h"

#include <algorithm>

#include "common.h"
#include "kernel/kernel.h"

namespace {

// uniform compute signature; the zero tile is neg's left operand,
// ignored by the rest
using UnaryFn = void (*)(tdr_idx_t, tdr_idx_t, tdr_idx_t);

void wrap_square(tdr_idx_t o, tdr_idx_t i, tdr_idx_t) { vec_square(o, i); }
void wrap_abs(tdr_idx_t o, tdr_idx_t i, tdr_idx_t) { vec_abs(o, i); }
void wrap_floor(tdr_idx_t o, tdr_idx_t i, tdr_idx_t) { vec_floor(o, i); }

// 0 - in, fixed truncate: two's-complement negation (see the banner)
void wrap_neg(tdr_idx_t o, tdr_idx_t i, tdr_idx_t zero) {
    vec_sub(o, zero, i, OVF_MODE_TRUNCATE);
}

UnaryFn unary_fn(tnop_unary_op_t op) {
    switch (op) {
        case TNOP_UNARY_SQUARE:
            return wrap_square;
        case TNOP_UNARY_ABS:
            return wrap_abs;
        case TNOP_UNARY_FLOOR:
            return wrap_floor;
        case TNOP_UNARY_NEG:
            return wrap_neg;
        default:
            return nullptr;
    }
}

} // namespace

/* one launch of this core's share in the split's block_w-wide chunks.
 * block_w is a descriptor field: only uint16-representable values ever
 * arrive (the host's fit loop guards the rest). */
tn_status_t tnop_unary(const tn_tensor_t *out, const tn_tensor_t *in,
                       const tnop_unary_param_t *param, tnop_split_t *s) {
    const uint16_t block_w = s->B0;
    const tnkr_chip_info_t *info = tnkr_spec_info();
    const uint64_t total = tn_tensor_calc_num(out);
    const uint64_t per_core = (total + g_op_ctx.num_cores - 1) / g_op_ctx.num_cores;
    const tn_tensor_t *operands[2] = {out, in};
    const uint64_t start = g_op_ctx.core_idx * per_core;
    if (start >= total) {
        return TN_SUCCESS; // this core's share is empty: idle
    }
    const uint64_t share = std::min(per_core, total - start);
    const uint64_t c_slice = get_max_common_div(share, info->smc_num);
    const uint64_t w_slice = share / c_slice;
    const UnaryFn compute = unary_fn(param->op);
    if (compute == nullptr) {
        return TN_ERR_PARAM; // reserved op (rsqrt/sqrt/sin/cos): not yet implemented
    }

    /* the share is a sequence of contiguous (cur_w, 1, c_slice) tiles:
     * the chunk at idx_w covers cur_w*c_slice elements from
     * idx_w*c_slice into the share, striped over c_slice lanes */
    const dim4 block_shape{block_w, 1, (uint16_t)info->smc_num, 1};
    const dim4 first_chunk{(uint16_t)std::min((uint64_t)block_w, w_slice), 1, (uint16_t)c_slice, 1};
    tdr_idx_t gdesc[2], ldesc[2][2];

    for (int k = 0; k < 2; ++k) {
        TNOP_TRY(gtensor(operands[k]->addr + start * tn_dtype_bytes(operands[k]->dtype),
                         first_chunk, operands[k]->dtype, &gdesc[k]));
        TNOP_TRY(ltensor(block_shape, operands[k]->dtype, &ldesc[k][0]));
        TNOP_TRY(ltensor(block_shape, operands[k]->dtype, &ldesc[k][1]));
    }

    /* neg's left operand: one Constant-layout zero tile for the whole
     * launch; all-zero bits read as 0 in every tnkr_imm_t
     * interpretation, whatever the input dtype */
    tdr_idx_t zero_const{};
    if (param->op == TNOP_UNARY_NEG) {
        const tnkr_imm_t kZero = {};
        TNOP_TRY(lconstant(in->dtype, kZero, &zero_const));
    }

    /* the host pass runs the body only to fit the split: the
     * allocations above are the whole resource cost, so stop here —
     * the engine work below is the scalar side's */
    if (!g_op_ctx.is_scalar) {
        return TN_SUCCESS;
    }

    uint32_t load_tag[2] = {0, 0}, vec_done[2] = {0, 0};
    dim4 live_in{0, 0, 0, 0}, live_out{0, 0, 0, 0}, live_tile[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    const uint64_t chunks = (w_slice + block_w - 1) / block_w;

    auto chunk_shape = [&](uint64_t chunk_idx) -> dim4 {
        return {(uint16_t)std::min((uint64_t)block_w, w_slice - chunk_idx * block_w), 1,
                (uint16_t)c_slice, 1};
    };

    /* one chunk's input half: the input descriptor + the load + its
     * publish. Issued one chunk ahead of the chunk's compute, before
     * the previous chunk's store (see the banner) */
    auto prep_load = [&](uint64_t chunk_idx) -> tn_status_t {
        const unsigned buf_idx = (unsigned)(chunk_idx & 1);
        const dim4 chunk = chunk_shape(chunk_idx);
        for (int k = 0; k < 2; ++k) {
            if (chunk != live_tile[buf_idx]) {
                TNOP_TRY(set_shape(ldesc[k][buf_idx], chunk));
            }
        }
        if (chunk != live_in) {
            TNOP_TRY(set_shape(gdesc[1], chunk));
        }
        TNOP_TRY(set_addr(gdesc[1], operands[1]->addr + (start + chunk_idx * block_w * c_slice) *
                                      tn_dtype_bytes(operands[1]->dtype)));
        live_in = chunk;
        live_tile[buf_idx] = chunk;
        ldst_load(ldesc[1][buf_idx], gdesc[1], 0, LDST_MODE_NORMAL);
        load_tag[buf_idx] = tnop_sync_tag(SYNC_TARGET_LDST);
        tnop_sync_stag(SYNC_TARGET_LDST, load_tag[buf_idx]);
        return TN_SUCCESS;
    };

    TNOP_TRY(prep_load(0));
    tnop_sync_wtag(SYNC_TARGET_VEC, SYNC_TARGET_LDST, load_tag[0]);
    for (uint64_t chunk_idx = 0; chunk_idx < chunks; ++chunk_idx) {
        const unsigned buf_idx = (unsigned)(chunk_idx & 1);
        if (chunk_idx + 1 < chunks) {
            TNOP_TRY(prep_load(chunk_idx + 1));
        }
        compute(ldesc[0][buf_idx], ldesc[1][buf_idx], zero_const);
        vec_done[buf_idx] = tnop_sync_tag(SYNC_TARGET_VEC);
        tnop_sync_stag(SYNC_TARGET_VEC, vec_done[buf_idx]);
        if (chunk_idx + 1 < chunks) {
            /* the next chunk's vec wait leads its compute: issued
             * before the next ldst publish (the next prep_load's
             * stag), so its scoreboard edge lands on the load publish
             * it targets, not a later one; it queues behind this
             * chunk's compute, so it stalls the vec only after it is
             * done. This one wait covers both of the next compute's
             * cross-engine hazards: the RAW on the loaded input tile,
             * and the WAR on its out tile — whose previous reader is
             * chunk i-1's store, already complete because LDST is
             * engine serial and that store precedes this load (the
             * prefetch must stay one chunk ahead of the store) */
            tnop_sync_wtag(SYNC_TARGET_VEC, SYNC_TARGET_LDST, load_tag[buf_idx ^ 1]);
        }
        const dim4 chunk = chunk_shape(chunk_idx);
        if (chunk != live_out) {
            TNOP_TRY(set_shape(gdesc[0], chunk));
        }
        TNOP_TRY(set_addr(gdesc[0], operands[0]->addr + (start + chunk_idx * block_w * c_slice) *
                                         tn_dtype_bytes(operands[0]->dtype)));
        live_out = chunk;
        tnop_sync_wtag(SYNC_TARGET_LDST, SYNC_TARGET_VEC, vec_done[buf_idx]);
        ldst_store(gdesc[0], 0, ldesc[0][buf_idx], REDUCE_MODE_NONE, LDST_MODE_NORMAL);
    }
    return TN_SUCCESS;
}
