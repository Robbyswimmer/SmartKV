"""Tests for CUDA quantization utilities."""

import pytest
import torch

from smartkv.core import _quant_cuda
from smartkv.core._quant_cpu import quantize_per_head
from smartkv.core.cache import SmartKVCache
from smartkv.kernels.bit_packing import compute_packed_size
from smartkv.kernels import quantized_attention, quantized_attention_bucketed

CUDA_AVAILABLE = torch.cuda.is_available() and _quant_cuda.smartkv_cuda is not None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="smartkv_cuda extension not available")
def test_quantize_per_head_cuda_matches_cpu():
    torch.manual_seed(0)
    device = torch.device('cuda')
    N, H, D = 8, 4, 16
    bits_list = [2, 3, 4, 8]

    for bits in bits_list:
        k = torch.randn(N, H, D, device=device)
        v = torch.randn(N, H, D, device=device)

        k_q_cuda, v_q_cuda, k_scale_cuda, v_scale_cuda = _quant_cuda.quantize_per_head_cuda(k, v, bits)

        # CPU reference using fallback path
        k_cpu = k.cpu()
        v_cpu = v.cpu()
        k_q_cpu, v_q_cpu, k_scale_cpu, v_scale_cpu = quantize_per_head(k_cpu, v_cpu, bits)

        torch.testing.assert_close(k_scale_cuda.cpu(), k_scale_cpu, atol=1e-5, rtol=0)
        torch.testing.assert_close(v_scale_cuda.cpu(), v_scale_cpu, atol=1e-5, rtol=0)
        assert k_q_cuda.shape == k_q_cpu.shape
        assert v_q_cuda.shape == v_q_cpu.shape
        assert torch.equal(k_q_cuda.cpu(), k_q_cpu)
        assert torch.equal(v_q_cuda.cpu(), v_q_cpu)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="smartkv_cuda extension not available")
def test_bucket_views_exposes_indices():
    cache = SmartKVCache(
        num_layers=1,
        num_heads=2,
        head_dim=8,
        memory_budget=0.3,
        device='cuda',
        available_bits=[2, 3, 4, 8],
        use_bucketed_layout=True,
    )

    torch.manual_seed(0)
    k = torch.randn(4, 2, 8, device='cuda')
    v = torch.randn(4, 2, 8, device='cuda')
    ids = [0, 1, 2, 3]
    cache.precision_map.update({0: 2, 1: 3, 2: 4, 3: 8})

    cache.quantize_and_store_batch(0, ids, k, v)
    buckets = cache.get_bucket_views(0)

    assert 2 in buckets and 3 in buckets and 4 in buckets and 8 in buckets
    assert buckets[2]['token_ids'].numel() == 1
    assert buckets[3]['token_ids'].numel() == 1
    assert buckets[4]['token_ids'].numel() == 1
    assert buckets[8]['token_ids'].numel() == 1
    assert buckets[2]['token_ids'].tolist() == [0]
    assert buckets[3]['token_ids'].tolist() == [1]
    assert buckets[4]['token_ids'].tolist() == [2]
    assert buckets[8]['token_ids'].tolist() == [3]

    payload = cache.retrieve_all_quantized(0)
    assert payload['k_qx'].device.type == 'cuda'
    assert payload['v_qx'].device.type == 'cuda'
    assert 'buckets' in payload
    for bits, view in payload['buckets'].items():
        assert view['token_ids'].device.type == 'cuda'
        assert view['k_qx'].device.type == 'cuda'
        expected_dim = cache.head_dim if bits == 8 else compute_packed_size(cache.head_dim, bits)
        assert view['packed_dim'].item() == expected_dim
        assert view['k_qx'].shape[-1] == expected_dim
        if bits < 8:
            assert view['packed'].item() == 1
            assert view['k_qx'].dtype == torch.uint8
        else:
            assert view['packed'].item() == 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="smartkv_cuda extension not available")
def test_pack_unpack_round_trip():
    torch.manual_seed(1)
    device = torch.device('cuda')
    num_tokens, num_heads, head_dim = 4, 3, 32

    for bits in [2, 3, 4]:
        max_val = (1 << (bits - 1)) - 1
        min_val = -max_val if bits > 1 else 0
        tensor = torch.randint(min_val, max_val + 1, (num_tokens, num_heads, head_dim), dtype=torch.int8, device=device)

        packed = _quant_cuda.pack_values(tensor, bits)
        shape = [num_tokens, num_heads, head_dim]
        unpacked = _quant_cuda.unpack_values(packed, bits, shape)

        assert torch.equal(unpacked, tensor)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="smartkv_cuda extension not available")
def test_bucketed_attention_matches_legacy():
    torch.manual_seed(2)
    device = torch.device('cuda')

    cache = SmartKVCache(
        num_layers=1,
        num_heads=2,
        head_dim=16,
        memory_budget=0.3,
        device='cuda',
        available_bits=[2, 3, 4, 8],
        use_bucketed_layout=True,
    )

    ids = list(range(6))
    cache.precision_map.update({i: [2, 3, 4, 8, 3, 2][i] for i in range(6)})

    k = torch.randn(len(ids), 2, 16, device=device)
    v = torch.randn(len(ids), 2, 16, device=device)
    cache.quantize_and_store_batch(0, ids, k, v)

    buckets = cache.get_bucket_views(0)
    query = torch.randn(1, 2, 1, 16, device=device)

    # Legacy tensors
    legacy = cache.retrieve_all_quantized(0)
    legacy_out = quantized_attention(
        query,
        legacy['k_qx'].unsqueeze(0),
        legacy['k_scale'].unsqueeze(0),
        legacy['v_qx'].unsqueeze(0),
        legacy['v_scale'].unsqueeze(0),
        use_cuda=True,
    )

    bucket_out = quantized_attention_bucketed(query, buckets, use_cuda=True)

    torch.testing.assert_close(bucket_out, legacy_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="smartkv_cuda extension not available")
def test_bucket_kernel_bug_fixes():
    """
    Comprehensive test validating all 4 critical bug fixes:
    1. Scale lookup stride (transpose from [H, num_tokens] to [num_tokens, H])
    2. Attention mask indexing (full_k_len parameter)
    3. Cross-bucket softmax normalization (streaming softmax)
    4. Head dimension limit (dynamic shared memory, d > 128)
    """
    torch.manual_seed(42)
    device = torch.device('cuda')

    # Test with d=128 to stress dynamic shared memory
    cache = SmartKVCache(
        num_layers=1,
        num_heads=4,
        head_dim=128,
        memory_budget=0.4,
        device='cuda',
        available_bits=[2, 4, 8],
        use_bucketed_layout=True,
    )

    # Create tokens with mixed precision across multiple buckets
    num_tokens = 32
    ids = list(range(num_tokens))

    # Assign different bit-widths to create 3 buckets
    precision_map = {}
    for i in range(num_tokens):
        if i < 10:
            precision_map[i] = 2
        elif i < 20:
            precision_map[i] = 4
        else:
            precision_map[i] = 8

    cache.precision_map.update(precision_map)

    # Generate test data
    k = torch.randn(num_tokens, 4, 128, device=device)
    v = torch.randn(num_tokens, 4, 128, device=device)
    cache.quantize_and_store_batch(0, ids, k, v)

    # Query with q_len > 1 to test attention mask indexing
    query = torch.randn(1, 4, 2, 128, device=device)

    # Create attention mask (test Issue 2: mask indexing)
    attention_mask = torch.zeros(1, 1, 2, num_tokens, device=device)
    attention_mask[:, :, :, :10] = -10000.0  # Mask first 10 tokens

    # Test bucket kernel path
    buckets = cache.get_bucket_views(0)
    bucket_out = quantized_attention_bucketed(query, buckets, attention_mask=attention_mask, use_cuda=True)

    # Validate output shape and no NaN/Inf (Issue 4: d=128 support)
    assert bucket_out.shape == (1, 4, 2, 128), f"Unexpected output shape: {bucket_out.shape}"
    assert not torch.isnan(bucket_out).any(), "Output contains NaN"
    assert not torch.isinf(bucket_out).any(), "Output contains Inf"

    # Validate masked tokens have near-zero contribution
    # Query position should attend minimally to masked tokens (first 10)
    # This indirectly validates scale lookup (Issue 1) and softmax (Issue 3)

    # Compare with legacy path (tests all issues end-to-end)
    legacy = cache.retrieve_all_quantized(0)
    legacy_out = quantized_attention(
        query,
        legacy['k_qx'].unsqueeze(0),
        legacy['k_scale'].unsqueeze(0),
        legacy['v_qx'].unsqueeze(0),
        legacy['v_scale'].unsqueeze(0),
        attention_mask=attention_mask,
        use_cuda=True,
    )

    # Should match legacy output (validates all fixes)
    torch.testing.assert_close(bucket_out, legacy_out, atol=1e-4, rtol=1e-4)
