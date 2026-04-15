# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant KV store.

Two kernels:
1. _tq_fused_store_fp8: FP8 key scatter + value uniform quantization.
2. _tq_fused_store_mse: Fused binary-search bucketize + MSE index
   packing + value quantization.

The launcher `triton_turboquant_store` selects the appropriate kernel.
"""

import math

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_kv_cache import (
    quantize_turboquant_vectors,
    validate_turboquant_group_indices,
)
from vllm.v1.attention.ops.triton_turboquant_decode import _use_fp8_e4b15


def _pack_uniform_values_reference(value: torch.Tensor, bits: int) -> torch.Tensor:
    levels = (1 << bits) - 1
    v_min = value.amin(dim=-1, keepdim=True)
    v_max = value.amax(dim=-1, keepdim=True)
    scale = (v_max - v_min) / max(levels, 1)
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(((value - v_min) / scale).round(), 0, levels).to(torch.uint8)
    if bits == 4:
        pairs = q.reshape(*q.shape[:-1], q.shape[-1] // 2, 2)
        data = (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()
    else:
        groups = q.reshape(*q.shape[:-1], q.shape[-1] // 8, 8).to(torch.int32)
        shifts = (torch.arange(8, device=value.device, dtype=torch.int32) * 3).view(
            *((1,) * (groups.ndim - 1)), 8
        )
        packed24 = torch.sum(groups << shifts, dim=-1)
        data = torch.stack(
            (
                (packed24 & 0xFF).to(torch.uint8),
                ((packed24 >> 8) & 0xFF).to(torch.uint8),
                ((packed24 >> 16) & 0xFF).to(torch.uint8),
            ),
            dim=-1,
        ).reshape(*q.shape[:-1], -1)
    tail = torch.cat(
        (
            scale.to(torch.float16).view(torch.uint8).reshape(*scale.shape[:-1], 2),
            v_min.to(torch.float16).view(torch.uint8).reshape(*v_min.shape[:-1], 2),
        ),
        dim=-1,
    )
    return torch.cat((data, tail), dim=-1)

# ═══════════════════════════════════════════════════════════════════════
# Shared: value uniform quantization + pack + scale/zero store
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _store_quantized_value(
    Value_ptr,
    KV_cache_ptr,
    base,  # pid * D offset into Value_ptr
    slot_base,  # byte offset into KV_cache_ptr for this slot+head
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    D: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Uniform quantization of values to VQB bits, pack, and store with scale/zero."""
    val_cache_offset = KPS

    if VQB == 3:
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 7.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        q_vals = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 7
        )

        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        q_grp = tl.reshape(q_vals, [BLOCK_GRP, 8])
        shifts_3bit = tl.arange(0, 8) * 3
        packed_24 = tl.sum(q_grp << shifts_3bit[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3,
            b0,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 1,
            b1,
            mask=grp_mask,
        )
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 2,
            b2,
            mask=grp_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )

    else:  # VQB == 4
        val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
        val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)
        v_scale = (val_max - val_min) / 15.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)

        # Quantize all D elements from register (no re-load)
        q_all = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 15
        )
        # Reshape to pairs and pack two 4-bit values per byte
        q_pairs = tl.reshape(q_all, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed_val = tl.sum((q_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        val_offs = tl.arange(0, BLOCK_D // 2)
        val_mask = val_offs < VAL_DATA_BYTES
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + val_offs,
            packed_val,
            mask=val_mask,
        )

        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 1,
            ((sc_u16 >> 8) & 0xFF).to(tl.uint8),
        )
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + sc_offset + 3,
            ((zr_u16 >> 8) & 0xFF).to(tl.uint8),
        )


# ═══════════════════════════════════════════════════════════════════════
# FP8 key store + value uniform quantization
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_fp8(
    Key_ptr,  # [NH, D] float16/bfloat16 — raw keys
    Value_ptr,  # [NH, D] float16/bfloat16 — raw values
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides (for computing byte offsets)
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    KPS: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    FP8_E4B15: tl.constexpr = 0,  # 1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """FP8 key cast+scatter + value uniform quantization."""
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos + head_idx * stride_cache_head
    )

    base = pid * D

    # ── FP8 KEY: cast to FP8 in-kernel and store ─────────────────
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    k_vals = tl.load(Key_ptr + base + d_offs, mask=d_mask, other=0.0)
    k_fp8 = k_vals.to(tl.float8e4b15) if FP8_E4B15 else k_vals.to(tl.float8e4nv)
    k_bytes = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + d_offs, k_bytes, mask=d_mask)

    # ── VALUE QUANTIZE + PACK ───────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused MSE store: bucketize + MSE index pack + norm store + value pack
# (eliminates 4 PyTorch kernel launches per layer vs pack-only kernel)
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _tq_fused_store_mse(
    # Post-rotation inputs
    Y_ptr,  # [NH, D] float32 — rotated normalized keys (x_hat @ PiT)
    Norms_ptr,  # [NH] float32 — key vector norms (||k||)
    Value_ptr,  # [NH, D] float32 — raw values
    QJLPacked_ptr,  # [NH, QJL_BYTES] uint8 — packed residual sign bits
    # Quantization tables
    Midpoints_ptr,  # [n_centroids-1] float32
    # Cache and indexing
    KV_cache_ptr,  # [total_bytes] uint8 (flattened view)
    Slot_mapping_ptr,  # [N] int32 — per-token slot indices
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # TQ layout
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    QJL_BYTES: tl.constexpr,
    # Value quantization
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Packing block sizes
    BLOCK_VAL: tl.constexpr,
    # MSE params
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Fused MSE quantize + pack + store.

    Performs binary-search bucketize, MSE index packing, norm storage,
    and value quantization in one kernel.
    """
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos + head_idx * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 1. BINARY SEARCH BUCKETIZE ───────────────────────────────────
    # Midpoints are sorted (N_CENTROIDS-1 values); binary search finds
    # insertion point in MSE_BITS iterations vs N_CENTROIDS-1 for linear.
    y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        # Clamp to valid midpoint index [0, N_CENTROIDS-2] for load safety;
        # the search result (lo) is still correct since converged lanes
        # don't change.
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    # ── 2. PACK MSE INDICES from register idx ─────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 2, b2, mask=grp_mask)
    elif MSE_BITS == 2:
        idx_quads = tl.reshape(idx, [BLOCK_D // 4, 4])
        shifts_2 = tl.arange(0, 4) * 2
        packed = tl.sum((idx_quads & 0x3) << shifts_2[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 4)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)

    # ── 3. STORE vec_norm (fp16, 2 bytes) ─────────────────────────────
    norm_offset = MSE_BYTES

    vn_f16 = tl.load(Norms_ptr + pid).to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    if QJL_BYTES > 0:
        qjl_offs = tl.arange(0, QJL_BYTES)
        qjl_vals = tl.load(QJLPacked_ptr + pid * QJL_BYTES + qjl_offs)
        tl.store(KV_cache_ptr + slot_base + norm_offset + 2 + qjl_offs, qjl_vals)

    # ── 4. VALUE QUANTIZE + PACK ──────────────────────────────────────
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


@triton.jit
def _tq_grouped_store_value(
    Value_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    stride_cache_dim: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    slot_base = (
        blk * stride_cache_block + off * stride_cache_pos + head_idx * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS * stride_cache_dim,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES * stride_cache_dim,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════


def triton_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    slot_mapping: torch.Tensor,  # [N] int32
    PiT: torch.Tensor | None,  # [D, D] float32
    midpoints: torch.Tensor | None,  # [n_centroids-1] float32
    PhiT: torch.Tensor | None,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    qjl_residual_bits: int = 0,
    grouped_recipe: str | None = None,
    group_rotations: tuple[torch.Tensor, torch.Tensor] | None = None,
    group_qjl: tuple[torch.Tensor, torch.Tensor] | None = None,
    group_centroids: dict[int, torch.Tensor] | None = None,
    group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Launch TQ store kernel or grouped reference store path."""
    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * mse_bits / 8)
    n_centroids = 2**mse_bits
    qjl_bytes = math.ceil(D * qjl_residual_bits / 8)

    val_data_bytes = math.ceil(D * value_quant_bits / 8)

    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)

    # Cache strides (element_size=1 for uint8, so stride in bytes = stride())
    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

    # ── FP8 PATH: in-kernel FP8 cast + scatter via fp8 kernel ──
    if grouped_recipe is not None:
        if (
            group_rotations is None
            or group_qjl is None
            or group_centroids is None
            or group_indices is None
        ):
            raise ValueError("Grouped TurboQuant store requires grouped tables.")
        validate_turboquant_group_indices(key, group_indices)
        valid_mask = slot_mapping >= 0
        if not torch.any(valid_mask):
            return
        valid_slots = slot_mapping[valid_mask].to(torch.int64)
        blocks = torch.div(valid_slots, kv_cache.shape[1], rounding_mode="floor")
        offsets = torch.remainder(valid_slots, kv_cache.shape[1])
        value_packed_size = val_data_bytes + 4
        kv_cache[blocks, offsets, :, :] = 0
        packed_value = _pack_uniform_values_reference(
            value.to(torch.float32), value_quant_bits
        )
        packed_key = quantize_turboquant_vectors(
            key.to(torch.float32),
            grouped_recipe,
            group_rotations,
            group_qjl,
            group_centroids,
            group_indices,
        )
        kv_cache[blocks, offsets, :, :key_packed_size] = packed_key[valid_mask]
        kv_cache[
            blocks,
            offsets,
            :,
            key_packed_size : key_packed_size + value_packed_size,
        ] = packed_value[valid_mask]
        return
    if PiT is None or midpoints is None:
        raise ValueError("PiT and midpoints are required for legacy TurboQuant store.")
    if key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()

        fp8_e4b15 = _use_fp8_e4b15(key.device.index or 0)

        grid = (NH,)
        _tq_fused_store_fp8[grid](
            k_flat,
            v_flat,
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            BLOCK_GRP=block_grp,
            FP8_E4B15=fp8_e4b15,
            num_warps=4,
            num_stages=1,
        )
        return

    # ── MSE PATH: external GEMM + fused bucketize/pack kernel ──
    # Normalize + rotation GEMM externally (cuBLAS is faster than in-kernel)
    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ PiT

    if qjl_residual_bits > 0:
        if PhiT is None:
            raise ValueError("PhiT is required when qjl_residual_bits > 0")
        # Reconstruct base centroids in PyTorch, then sketch the residual with
        # a second orthogonal sign transform.
        # bucketize returns [0, n_centroids-1] against sorted midpoints.
        idx = torch.bucketize(y, midpoints)
        centroid_table = get_centroids(D, mse_bits).to(device=y.device, dtype=y.dtype)
        c_vals = centroid_table[idx]
        resid = y - c_vals
        resid_proj = resid @ PhiT
        sign_bits = (resid_proj >= 0).to(torch.uint8)
        qjl_packed = torch.zeros(NH, qjl_bytes, device=y.device, dtype=torch.uint8)
        for bit in range(D):
            byte_idx = bit // 8
            bit_shift = bit % 8
            qjl_packed[:, byte_idx] |= sign_bits[:, bit] << bit_shift
    else:
        qjl_packed = torch.empty(NH, max(qjl_bytes, 1), device=y.device, dtype=torch.uint8)

    v_flat = value.float().reshape(NH, D)

    # Fused kernel: bucketize + MSE index pack + norm store + value pack
    grid = (NH,)
    _tq_fused_store_mse[grid](
        y,
        norms.squeeze(1),
        v_flat,
        qjl_packed,
        midpoints,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        MSE_BYTES=mse_bytes,
        KPS=key_packed_size,
        QJL_BYTES=qjl_bytes,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=block_grp,
        num_warps=4,
        num_stages=1,
    )
