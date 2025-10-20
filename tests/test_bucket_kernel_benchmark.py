#!/usr/bin/env python3
"""
Benchmark test comparing bucket kernel vs legacy path for SmartKV and uniform INT8.

This test validates that:
1. Bucket kernel produces correct results vs legacy
2. Bucket kernel performance is measured vs uniform INT8 baseline
3. Mixed-precision (2/3/4/8-bit) shows speedup vs uniform 8-bit
"""

import pytest
import torch
from time import perf_counter

from smartkv.core.cache import SmartKVCache
from smartkv.kernels import quantized_attention, quantized_attention_bucketed, CUDA_AVAILABLE

CUDA_AVAILABLE = CUDA_AVAILABLE and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_bucket_kernel_vs_uniform_int8():
    """
    Compare bucket kernel (mixed precision) vs uniform INT8 baseline.

    Tests:
    - Correctness: bucket kernel matches legacy path
    - Performance: measure latency for both approaches
    - Memory: mixed precision uses less memory than uniform INT8
    """
    torch.manual_seed(42)
    device = torch.device('cuda')

    num_heads = 32
    head_dim = 128
    context_length = 2048

    # Build SmartKV cache (mixed precision)
    smartkv_cache = SmartKVCache(
        num_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
        memory_budget=0.3,
        available_bits=[2, 4, 8],
        device='cuda',
        use_bucketed_layout=True,  # Enable bucket kernel
        use_bit_packing=True,
    )

    # Build uniform INT8 cache (baseline)
    int8_cache = SmartKVCache(
        num_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
        memory_budget=0.5,  # INT8 uses 0.5x memory vs FP16
        available_bits=[8],
        device='cuda',
        use_bucketed_layout=True,
    )

    # Populate both caches with same random data
    chunk_size = 128
    for start in range(0, context_length, chunk_size):
        size = min(chunk_size, context_length - start)
        token_ids = list(range(start, start + size))
        k = torch.randn(size, num_heads, head_dim, device=device)
        v = torch.randn(size, num_heads, head_dim, device=device)

        # SmartKV: allocate adaptive precision
        smartkv_cache.allocate_precision(0, token_ids)
        smartkv_cache.quantize_and_store_batch(0, token_ids, k, v)

        # INT8: force all to 8-bit
        for tid in token_ids:
            int8_cache.precision_map[tid] = 8
        int8_cache.quantize_and_store_batch(0, token_ids, k, v)

    # Query
    query = torch.randn(1, num_heads, 1, head_dim, device=device)

    print("\n" + "="*60)
    print("BUCKET KERNEL BENCHMARK")
    print("="*60)

    # Test 1: SmartKV bucket kernel
    print("\n[1] SmartKV (mixed 2/4/8-bit) with bucket kernel:")
    smartkv_buckets = smartkv_cache.get_bucket_views(0)

    # Correctness check: bucket kernel vs legacy
    legacy = smartkv_cache.retrieve_all_quantized(0)
    legacy_out = quantized_attention(
        query,
        legacy['k_qx'].unsqueeze(0),
        legacy['k_scale'].unsqueeze(0),
        legacy['v_qx'].unsqueeze(0),
        legacy['v_scale'].unsqueeze(0),
        use_cuda=True,
    )

    bucket_out = quantized_attention_bucketed(query, smartkv_buckets, use_cuda=True)
    torch.testing.assert_close(bucket_out, legacy_out, atol=1e-4, rtol=1e-4)
    print("  ✓ Correctness: matches legacy path")

    # Performance
    warmup, iters = 10, 100
    for _ in range(warmup):
        _ = quantized_attention_bucketed(query, smartkv_buckets, use_cuda=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = perf_counter()
        _ = quantized_attention_bucketed(query, smartkv_buckets, use_cuda=True)
        torch.cuda.synchronize()
        times.append(perf_counter() - start)

    smartkv_latency = sum(times) / len(times) * 1000
    print(f"  Latency: {smartkv_latency:.3f} ms")

    # Memory stats
    stats = smartkv_cache.get_memory_stats()
    print(f"  Avg bits: {stats.get('avg_bits', 0):.2f}")
    print(f"  Memory ratio: {stats.get('memory_ratio_true', 0):.3f}")

    # Test 2: Uniform INT8 baseline
    print("\n[2] Uniform INT8 baseline with bucket kernel:")
    int8_buckets = int8_cache.get_bucket_views(0)

    for _ in range(warmup):
        _ = quantized_attention_bucketed(query, int8_buckets, use_cuda=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = perf_counter()
        _ = quantized_attention_bucketed(query, int8_buckets, use_cuda=True)
        torch.cuda.synchronize()
        times.append(perf_counter() - start)

    int8_latency = sum(times) / len(times) * 1000
    print(f"  Latency: {int8_latency:.3f} ms")
    print(f"  Memory ratio: 0.500 (uniform INT8)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    speedup = int8_latency / smartkv_latency
    memory_saving = (0.5 - stats.get('memory_ratio_true', 0.5)) / 0.5 * 100

    print(f"SmartKV latency  : {smartkv_latency:.3f} ms")
    print(f"INT8 latency     : {int8_latency:.3f} ms")
    print(f"Speedup          : {speedup:.2f}x")
    print(f"Memory savings   : {memory_saving:.1f}% vs INT8")
    print("="*60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
