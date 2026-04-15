# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_kv_cache import (
    dequantize_turboquant_vectors,
    validate_turboquant_group_indices,
)
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

_FP8_E4B15: dict[int, int] = {}
def _unpack_uniform_values_reference(
    packed: torch.Tensor,
    head_dim: int,
    bits: int,
) -> torch.Tensor:
    data_bytes = math.ceil(head_dim * bits / 8)
    data = packed[..., :data_bytes]
    tail = packed[..., data_bytes : data_bytes + 4]
    scale = tail[..., :2].contiguous().view(torch.float16).to(torch.float32).unsqueeze(-1)
    v_min = tail[..., 2:4].contiguous().view(torch.float16).to(torch.float32).unsqueeze(-1)
    if bits == 4:
        q0 = data & 0xF
        q1 = (data >> 4) & 0xF
        q = torch.stack((q0, q1), dim=-1).reshape(*data.shape[:-1], head_dim)
    else:
        raw = data.reshape(*data.shape[:-1], head_dim // 8, 3).to(torch.int32)
        packed24 = raw[..., 0] | (raw[..., 1] << 8) | (raw[..., 2] << 16)
        shifts = (torch.arange(8, device=packed.device, dtype=torch.int32) * 3).view(
            *((1,) * (raw.ndim - 1)), 8
        )
        q = ((packed24.unsqueeze(-1) >> shifts) & 0x7).reshape(*raw.shape[:-2], head_dim)
    return q.to(torch.float32) * scale + v_min


def _use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9), else 0."""
    if device not in _FP8_E4B15:
        cap = torch.cuda.get_device_capability(device)
        _FP8_E4B15[device] = 1 if cap < (8, 9) else 0
    return _FP8_E4B15[device]


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,  # [B, Hq, D] float32
    Q_qjl_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,  # KV cache
    stride_bt_b,  # block_table stride per batch
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,  # key_packed_size
    QJL_BYTES: tl.constexpr,
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    QJL_ENABLED: tl.constexpr = 0,
    QJL_SCALE: tl.constexpr = 0.0,
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    q_qjl = tl.load(Q_qjl_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        )

        slot_bases = (
            block_nums * stride_cache_block
            + page_off * stride_cache_pos
            + kv_head * stride_cache_head
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0),
                    axis=1,
                )
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                axis=1,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            if QJL_ENABLED:
                qjl_base = slot_bases + MSE_BYTES + 2
                qjl_byte_idx = d_offs // 8
                qjl_bit_shift = d_offs % 8
                qjl_raw = tl.load(
                    KV_cache_ptr + qjl_base[:, None] + qjl_byte_idx[None, :],
                    mask=kv_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)
                qjl_sign = tl.where(
                    ((qjl_raw >> qjl_bit_shift[None, :]) & 0x1) > 0,
                    1.0,
                    -1.0,
                )
                term2 = tl.sum(
                    tl.where(d_mask[None, :], q_qjl[None, :] * qjl_sign, 0.0),
                    axis=1,
                )
                scores += vec_norms * QJL_SCALE * term2 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION
        # ============================================================
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


@triton.jit
def _grouped_tq_decode_stage1(
    Q_rot_0_ptr,
    Q_qjl_0_ptr,
    Q_rot_1_ptr,
    Q_qjl_1_ptr,
    KV_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_0_ptr,
    Centroids_1_ptr,
    Mid_o_ptr,
    stride_q0_b,
    stride_q0_h,
    stride_q1_b,
    stride_q1_h,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_cache_dim,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    G0_DIM: tl.constexpr,
    G0_PADDED: tl.constexpr,
    G0_MSE_BITS: tl.constexpr,
    G0_GROUP_OFFSET: tl.constexpr,
    G0_QJL_OFFSET: tl.constexpr,
    G0_VECTOR_NORM_OFFSET: tl.constexpr,
    G0_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G0_QJL_SCALE: tl.constexpr,
    G1_DIM: tl.constexpr,
    G1_PADDED: tl.constexpr,
    G1_MSE_BITS: tl.constexpr,
    G1_GROUP_OFFSET: tl.constexpr,
    G1_QJL_OFFSET: tl.constexpr,
    G1_VECTOR_NORM_OFFSET: tl.constexpr,
    G1_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G1_QJL_SCALE: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    kv_head = hid // KV_GROUP_SIZE
    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    offs_d0 = tl.arange(0, G0_PADDED)
    mask_d0 = offs_d0 < G0_DIM
    offs_d1 = tl.arange(0, G1_PADDED)
    mask_d1 = offs_d1 < G1_DIM

    q0_base = bid * stride_q0_b + hid * stride_q0_h
    q_rot_0 = tl.load(
        Q_rot_0_ptr + q0_base + offs_d0, mask=mask_d0, other=0.0
    ).to(tl.float32)
    q_qjl_0 = tl.load(
        Q_qjl_0_ptr + q0_base + offs_d0, mask=mask_d0, other=0.0
    ).to(tl.float32)
    q1_base = bid * stride_q1_b + hid * stride_q1_h
    q_rot_1 = tl.load(
        Q_rot_1_ptr + q1_base + offs_d1, mask=mask_d1, other=0.0
    ).to(tl.float32)
    q_qjl_1 = tl.load(
        Q_qjl_1_ptr + q1_base + offs_d1, mask=mask_d1, other=0.0
    ).to(tl.float32)

    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        )
        slot_bases = (
            block_nums * stride_cache_block
            + page_off * stride_cache_pos
            + kv_head * stride_cache_head
        )

        key_indices_0 = _grouped_unpack_fixed_indices(
            KV_cache_ptr,
            slot_bases,
            offs_d0,
            stride_cache_dim,
            G0_MSE_BITS,
            BLOCK_KV,
            G0_PADDED,
            G0_GROUP_OFFSET,
            kv_mask[:, None] & mask_d0[None, :],
        )
        key_centroids_0 = tl.load(
            Centroids_0_ptr + key_indices_0,
            mask=kv_mask[:, None] & mask_d0[None, :],
            other=0.0,
        )
        key_qjl_signs_0 = _grouped_unpack_signs(
            KV_cache_ptr,
            slot_bases,
            offs_d0,
            stride_cache_dim,
            G0_QJL_OFFSET,
            kv_mask[:, None] & mask_d0[None, :],
        )
        key_vector_norm_0_base = slot_bases + G0_VECTOR_NORM_OFFSET * stride_cache_dim
        kv0_lo = tl.load(KV_cache_ptr + key_vector_norm_0_base, mask=kv_mask, other=0).to(
            tl.uint16
        )
        kv0_hi = tl.load(
            KV_cache_ptr + key_vector_norm_0_base + stride_cache_dim,
            mask=kv_mask,
            other=0,
        ).to(tl.uint16)
        key_vector_norm_0 = (kv0_lo | (kv0_hi << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)
        kr0_base = slot_bases + G0_RESIDUAL_NORM_OFFSET * stride_cache_dim
        kr0_lo = tl.load(KV_cache_ptr + kr0_base, mask=kv_mask, other=0).to(tl.uint16)
        kr0_hi = tl.load(
            KV_cache_ptr + kr0_base + stride_cache_dim, mask=kv_mask, other=0
        ).to(tl.uint16)
        key_residual_norm_0 = (kr0_lo | (kr0_hi << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)

        key_indices_1 = _grouped_unpack_fixed_indices(
            KV_cache_ptr,
            slot_bases,
            offs_d1,
            stride_cache_dim,
            G1_MSE_BITS,
            BLOCK_KV,
            G1_PADDED,
            G1_GROUP_OFFSET,
            kv_mask[:, None] & mask_d1[None, :],
        )
        key_centroids_1 = tl.load(
            Centroids_1_ptr + key_indices_1,
            mask=kv_mask[:, None] & mask_d1[None, :],
            other=0.0,
        )
        key_qjl_signs_1 = _grouped_unpack_signs(
            KV_cache_ptr,
            slot_bases,
            offs_d1,
            stride_cache_dim,
            G1_QJL_OFFSET,
            kv_mask[:, None] & mask_d1[None, :],
        )
        key_vector_norm_1_base = slot_bases + G1_VECTOR_NORM_OFFSET * stride_cache_dim
        kv1_lo = tl.load(KV_cache_ptr + key_vector_norm_1_base, mask=kv_mask, other=0).to(
            tl.uint16
        )
        kv1_hi = tl.load(
            KV_cache_ptr + key_vector_norm_1_base + stride_cache_dim,
            mask=kv_mask,
            other=0,
        ).to(tl.uint16)
        key_vector_norm_1 = (kv1_lo | (kv1_hi << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)
        kr1_base = slot_bases + G1_RESIDUAL_NORM_OFFSET * stride_cache_dim
        kr1_lo = tl.load(KV_cache_ptr + kr1_base, mask=kv_mask, other=0).to(tl.uint16)
        kr1_hi = tl.load(
            KV_cache_ptr + kr1_base + stride_cache_dim, mask=kv_mask, other=0
        ).to(tl.uint16)
        key_residual_norm_1 = (kr1_lo | (kr1_hi << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)

        scores = key_vector_norm_0 * tl.sum(
            key_centroids_0 * q_rot_0[None, :], axis=1
        )
        scores += key_vector_norm_0 * key_residual_norm_0 * G0_QJL_SCALE * tl.sum(
            key_qjl_signs_0 * q_qjl_0[None, :], axis=1
        )
        scores += key_vector_norm_1 * tl.sum(
            key_centroids_1 * q_rot_1[None, :], axis=1
        )
        scores += key_vector_norm_1 * key_residual_norm_1 * G1_QJL_SCALE * tl.sum(
            key_qjl_signs_1 * q_qjl_1[None, :], axis=1
        )
        scores = tl.where(kv_mask, scores * ATTN_SCALE, -float("inf"))

        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        val_bases = slot_bases + KPS * stride_cache_dim
        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :] * stride_cache_dim
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + stride_cache_dim,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)
        else:
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :] * stride_cache_dim
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

        sc_bases = val_bases + VAL_DATA_BYTES * stride_cache_dim
        sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
        sc_hi = tl.load(
            KV_cache_ptr + sc_bases + stride_cache_dim, mask=kv_mask, other=0
        ).to(tl.uint16)
        v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(
            KV_cache_ptr + sc_bases + 2 * stride_cache_dim, mask=kv_mask, other=0
        ).to(tl.uint16)
        zr_hi = tl.load(
            KV_cache_ptr + sc_bases + 3 * stride_cache_dim, mask=kv_mask, other=0
        ).to(tl.uint16)
        v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = v_idx * v_scales[:, None] + v_zeros[:, None]

        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_prev + tl.log(safe_l))


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] float16
    V_out_ptr,  # [B, Hk, max_seq, D] float16
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    QJL_BYTES: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx)
    slot_base = (
        block_num * stride_cache_block
        + page_off * stride_cache_pos
        + hid * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs, mask=d_mask, other=0)
        if FP8_E4B15:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_key = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16_key >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    elif VQB == 3:
        # 3-bit value unpack: 8 values per 3 bytes
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_val = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16_val >> val_bit_shift) & 0x7).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "qjl_bytes": max(0, key_packed_size - math.ceil(D * mse_bits / 8) - 2),
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def triton_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor | None,  # [D, D] float32
    centroids: torch.Tensor | None,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,  # [D, D] pre-computed Pi.T contiguous
    PhiT: torch.Tensor | None = None,  # [D, D] second orthogonal sketch matrix
    qjl_scale: torch.Tensor | None = None,
    grouped_recipe: str | None = None,
    group_rotations: tuple[torch.Tensor, torch.Tensor] | None = None,
    group_qjl: tuple[torch.Tensor, torch.Tensor] | None = None,
    group_centroids: dict[int, torch.Tensor] | None = None,
    group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    if grouped_recipe is not None:
        if (
            group_rotations is None
            or group_qjl is None
            or group_centroids is None
            or group_indices is None
        ):
            raise ValueError("Grouped TurboQuant decode requires grouped tables.")
        validate_turboquant_group_indices(
            torch.empty(
                query.shape[0], kv_cache.shape[2], D, device=query.device, dtype=query.dtype
            ),
            group_indices,
        )
        output = torch.zeros_like(query)
        key_bytes = key_packed_size
        value_bytes = math.ceil(D * value_quant_bits / 8) + 4
        block_size = kv_cache.shape[1]
        for seq_idx, seq_len in enumerate(seq_lens.tolist()):
            num_blocks = math.ceil(seq_len / block_size)
            block_ids = block_table[seq_idx, :num_blocks].to(torch.int64)
            seq_cache = kv_cache.index_select(0, block_ids).reshape(
                num_blocks * block_size, kv_cache.shape[2], kv_cache.shape[3]
            )[:seq_len]
            k_seq = dequantize_turboquant_vectors(
                seq_cache[..., :key_bytes],
                grouped_recipe,
                D,
                group_rotations,
                group_qjl,
                group_centroids,
                group_indices,
                query.dtype,
            )
            v_seq = _unpack_uniform_values_reference(
                seq_cache[..., key_bytes : key_bytes + value_bytes],
                D,
                value_quant_bits,
            ).to(query.dtype)
            q_t = query[seq_idx : seq_idx + 1].transpose(0, 1).unsqueeze(0).contiguous()
            k_t = k_seq.transpose(0, 1).unsqueeze(0).contiguous()
            v_t = v_seq.transpose(0, 1).unsqueeze(0).contiguous()
            output[seq_idx : seq_idx + 1] = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                is_causal=False,
                scale=scale,
                enable_gqa=(k_t.shape[1] < q_t.shape[1]),
            )[0].transpose(0, 1)
        return output
    if Pi is None or centroids is None:
        raise ValueError("Pi and centroids are required for legacy TurboQuant decode.")

    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)
    if key_fp8:
        cfg = {**cfg, "qjl_bytes": 0}

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    # FP8 path: pass query directly (float16); kernel casts inline.
    # MSE path: still needs external GEMM (cuBLAS), so q_rot is float32.
    if key_fp8:
        q_rot = query.contiguous()
        q_qjl = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()
        if cfg["qjl_bytes"] > 0 and PhiT is not None:
            q_qjl = (q_rot @ PhiT).contiguous()
        else:
            q_qjl = q_rot

    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    # Stage 1: split-KV tiled attention scoring + value accumulation
    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    BLOCK_KV = 4
    grid = (B, Hq, NUM_KV_SPLITS)
    _tq_decode_stage1[grid](
        q_rot,
        q_qjl,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        QJL_BYTES=cfg["qjl_bytes"],
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=BLOCK_KV,
        KEY_FP8=1 if key_fp8 else 0,
        QJL_ENABLED=1 if cfg["qjl_bytes"] > 0 else 0,
        QJL_SCALE=(float(qjl_scale.item()) if qjl_scale is not None else 0.0),
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        num_warps=1,
        num_stages=1,
    )

    # Stage 2: Reduce across KV splits
    if output_buf is not None and output_buf.shape[0] >= B:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        num_warps=4,
        num_stages=2,
    )

    return output.to(query.dtype)
