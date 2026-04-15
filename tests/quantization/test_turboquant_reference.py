# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.turboquant_kv_cache import (
    apply_turboquant_query_transforms,
    build_turboquant_outlier_masks,
    canonicalize_turboquant_dtype,
    dequantize_turboquant_vectors,
    get_turboquant_bits,
    get_turboquant_centroids,
    get_turboquant_group_dims,
    get_turboquant_layout,
    get_turboquant_packed_dim,
    get_turboquant_qjl_matrix,
    get_turboquant_qjl_inverse_transform_matrix,
    get_turboquant_rotation,
    get_turboquant_mse_inverse_transform_matrix,
    get_turboquant_mse_to_qjl_matrix,
    pack_turboquant_indices,
    quantize_turboquant_vectors,
    unpack_turboquant_indices,
    validate_turboquant_group_indices,
)
from vllm.v1.attention.ops.turboquant_metadata import (
    TurboQuantTensorMetadata,
    build_default_turboquant_metadata,
    discover_turboquant_metadata_path,
    load_turboquant_metadata,
    save_turboquant_metadata,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store


def test_turboquant_aliases_match_reference_recipes():
    assert canonicalize_turboquant_dtype("turboquant_3bit") == "turboquant25"
    assert canonicalize_turboquant_dtype("turboquant_4bit") == "turboquant35"
    assert get_turboquant_bits("turboquant_3bit") == 2.5
    assert get_turboquant_bits("turboquant_4bit") == 3.5


def test_turboquant_layout_is_consistent():
    layout = get_turboquant_layout("turboquant_4bit", 128)
    high_dim, low_dim = get_turboquant_group_dims(128, "turboquant_4bit")
    assert (high_dim, low_dim) == (64, 64)
    assert layout.groups[0].dim == 64
    assert layout.groups[0].mse_bits == 3
    assert layout.groups[1].dim == 64
    assert layout.groups[1].mse_bits == 2
    assert layout.packed_dim == get_turboquant_packed_dim(128, "turboquant35")


def test_pack_unpack_roundtrip():
    values = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.uint8)
    packed = pack_turboquant_indices(values, 2)
    unpacked = unpack_turboquant_indices(packed, values.shape[-1], 2)
    assert torch.equal(values, unpacked)


def test_quantize_dequantize_reference_path_shapes():
    torch.manual_seed(0)
    x = torch.randn(3, 2, 128, dtype=torch.float32)
    high_idx, low_idx = build_turboquant_outlier_masks(x, "turboquant_3bit")
    device = torch.device("cpu")
    rotations = (
        get_turboquant_rotation(device, high_idx.shape[-1]),
        get_turboquant_rotation(device, low_idx.shape[-1]),
    )
    qjl_matrices = (
        get_turboquant_qjl_matrix(device, high_idx.shape[-1]),
        get_turboquant_qjl_matrix(device, low_idx.shape[-1]),
    )
    centroids = {
        1: get_turboquant_centroids(device, low_idx.shape[-1], 1),
        2: get_turboquant_centroids(device, high_idx.shape[-1], 2),
    }
    packed = quantize_turboquant_vectors(
        x, "turboquant_3bit", rotations, qjl_matrices, centroids, (high_idx, low_idx)
    )
    restored = dequantize_turboquant_vectors(
        packed,
        "turboquant_3bit",
        128,
        rotations,
        qjl_matrices,
        centroids,
        (high_idx, low_idx),
        x.dtype,
    )
    assert packed.shape[-1] == get_turboquant_layout("turboquant_3bit", 128).packed_dim
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()


def test_turboquant_metadata_roundtrip(tmp_path):
    metadata = build_default_turboquant_metadata(
        recipe="turboquant_4bit",
        head_size=128,
        num_kv_heads=2,
        layer_names=["model.layers.0.self_attn"],
        model_name="tests/turboquant",
    )
    path = tmp_path / "turboquant_kv.json"
    save_turboquant_metadata(metadata, path)
    loaded = load_turboquant_metadata(str(path))
    assert loaded.recipe == "turboquant35"
    assert loaded.get_layer("language_model.model.layers.0.self_attn.attn") == (
        loaded.layers["model.layers.0.self_attn"]
    )
    assert discover_turboquant_metadata_path(str(tmp_path), None) == str(path.resolve())


def test_turboquant_tensor_metadata_group_indices_shape():
    metadata = TurboQuantTensorMetadata(
        high_precision_indices=((0, 1, 2, 3, 4, 5, 6, 7),)
    )
    high, low = metadata.get_group_indices(
        device=torch.device("cpu"),
        head_size=32,
        kv_cache_dtype="turboquant_3bit",
    )
    assert high.shape == (1, 8)
    assert low.shape == (1, 24)


def test_grouped_op_layer_store_decode_reference_path():
    torch.manual_seed(0)
    device = torch.device("cpu")
    key = torch.randn(1, 2, 128, dtype=torch.float32, device=device)
    value = torch.randn(1, 2, 128, dtype=torch.float32, device=device)
    high_idx, low_idx = build_turboquant_outlier_masks(key, "turboquant_3bit")
    rotations = (
        get_turboquant_rotation(device, high_idx.shape[-1]),
        get_turboquant_rotation(device, low_idx.shape[-1]),
    )
    qjl_matrices = (
        get_turboquant_qjl_matrix(device, high_idx.shape[-1]),
        get_turboquant_qjl_matrix(device, low_idx.shape[-1]),
    )
    centroids = {
        1: get_turboquant_centroids(device, low_idx.shape[-1], 1),
        2: get_turboquant_centroids(device, high_idx.shape[-1], 2),
    }
    layout = get_turboquant_layout("turboquant_3bit", 128)
    value_bytes = (128 * 3 + 7) // 8 + 4
    kv_cache = torch.zeros(1, 16, 2, layout.packed_dim + value_bytes, dtype=torch.uint8)
    slot_mapping = torch.tensor([0], dtype=torch.int32)
    triton_turboquant_store(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        PiT=None,
        midpoints=None,
        PhiT=None,
        mse_bits=2,
        key_packed_size=layout.packed_dim,
        value_quant_bits=3,
        grouped_recipe="turboquant_3bit",
        group_rotations=rotations,
        group_qjl=qjl_matrices,
        group_centroids=centroids,
        group_indices=(high_idx, low_idx),
    )
    output = triton_turboquant_decode_attention(
        query=key,
        kv_cache=kv_cache,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        Pi=None,
        centroids=None,
        scale=1.0 / (128**0.5),
        mse_bits=2,
        key_packed_size=layout.packed_dim,
        value_quant_bits=3,
        grouped_recipe="turboquant_3bit",
        group_rotations=rotations,
        group_qjl=qjl_matrices,
        group_centroids=centroids,
        group_indices=(high_idx, low_idx),
    )
    assert output.shape == value.shape
    assert torch.isfinite(output).all()


def test_grouped_query_transforms_shapes():
    torch.manual_seed(0)
    query = torch.randn(2, 4, 128, dtype=torch.float32)
    group0 = torch.arange(64, dtype=torch.int64).repeat(2, 1)
    group1 = torch.arange(64, 128, dtype=torch.int64).repeat(2, 1)
    kv_head_for_query_head = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    rotations = (
        get_turboquant_rotation(torch.device("cpu"), 64),
        get_turboquant_rotation(torch.device("cpu"), 64, seed_offset=1),
    )
    qjl = (
        get_turboquant_qjl_matrix(torch.device("cpu"), 64),
        get_turboquant_qjl_matrix(torch.device("cpu"), 64, seed_offset=1),
    )
    (q_rot0, q_rot1), (q_qjl0, q_qjl1) = apply_turboquant_query_transforms(
        query,
        (group0, group1),
        rotations,
        qjl,
        kv_head_for_query_head=kv_head_for_query_head,
    )
    assert q_rot0.shape == (2, 4, 64)
    assert q_rot1.shape == (2, 4, 64)
    assert q_qjl0.shape == (2, 4, 64)
    assert q_qjl1.shape == (2, 4, 64)


def test_transform_matrix_helpers_shapes():
    device = torch.device("cpu")
    mse_inv = get_turboquant_mse_inverse_transform_matrix(device, 64)
    qjl_inv = get_turboquant_qjl_inverse_transform_matrix(device, 64)
    mse_to_qjl = get_turboquant_mse_to_qjl_matrix(device, 64)
    assert mse_inv.shape == (64, 64)
    assert qjl_inv.shape == (64, 64)
    assert mse_to_qjl.shape == (64, 64)


def test_group_index_validation_rejects_head_mismatch():
    x = torch.randn(1, 2, 128, dtype=torch.float32)
    bad_group0 = torch.arange(64, dtype=torch.int64).repeat(3, 1)
    bad_group1 = torch.arange(64, 128, dtype=torch.int64).repeat(3, 1)
    try:
        validate_turboquant_group_indices(x, (bad_group0, bad_group1))
    except ValueError as exc:
        assert "KV head count" in str(exc)
    else:
        raise AssertionError("Expected validate_turboquant_group_indices to fail")
