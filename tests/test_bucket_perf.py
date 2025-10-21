#!/usr/bin/env python3
"""
Simple performance benchmark for bucket kernel across different context lengths.
"""

import pytest
import torch
from time import perf_counter

from smartkv.core.cache import SmartKVCache
from smartkv.kernels import quantized_attention_bucketed, CUDA_AVAILABLE

CUDA_AVAILABLE = CUDA_AVAILABLE and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_bucket_kernel_performance():
    """
    Benchmark bucket kernel at various context lengths.
    Measures latency, memory usage, and precision distribution.
    """
    device = torch.device('cuda')
    torch.manual_seed(42)

    # Test configurations
    num_heads = 32
    head_dim = 128
    context_lengths = [512, 1024, 2048, 4096]
    memory_budget = 0.3
    available_bits = [2, 4, 8]

    print("\n" + "="*80)
    print("BUCKET KERNEL PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Configuration: {num_heads} heads × {head_dim} dim, budget={memory_budget}")
    print(f"Available precisions: {available_bits}-bit")
    print("="*80)

    results = []

    for ctx_len in context_lengths:
        print(f"\n[Context length: {ctx_len}]")

        # Create cache
        cache = SmartKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_budget=memory_budget,
            available_bits=available_bits,
            device='cuda',
            use_bucketed_layout=True,
            use_bit_packing=True,
        )

        # Populate cache
        chunk_size = 128
        for start in range(0, ctx_len, chunk_size):
            size = min(chunk_size, ctx_len - start)
            token_ids = list(range(start, start + size))
            k = torch.randn(size, num_heads, head_dim, device=device)
            v = torch.randn(size, num_heads, head_dim, device=device)

            cache.allocate_precision(0, token_ids)
            cache.quantize_and_store_batch(0, token_ids, k, v)

        # Get bucket views
        buckets = cache.get_bucket_views(0)

        # Query
        query = torch.randn(1, num_heads, 1, head_dim, device=device)

        # Warmup
        for _ in range(10):
            _ = quantized_attention_bucketed(query, buckets, use_cuda=True)
        torch.cuda.synchronize()

        # Benchmark
        iters = 100
        times = []
        for _ in range(iters):
            start = perf_counter()
            output = quantized_attention_bucketed(query, buckets, use_cuda=True)
            torch.cuda.synchronize()
            times.append(perf_counter() - start)

        # Verify correctness
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Compute metrics
        avg_latency_ms = sum(times) / len(times) * 1000
        p50_latency_ms = sorted(times)[len(times)//2] * 1000
        p95_latency_ms = sorted(times)[int(len(times)*0.95)] * 1000

        stats = cache.get_memory_stats()
        avg_bits = stats.get('avg_bits', 0)
        memory_ratio = stats.get('memory_ratio_true', 0)

        # Count tokens per bucket
        bucket_dist = {}
        for bits in available_bits:
            if bits in buckets:
                bucket_dist[bits] = len(buckets[bits]['token_ids'])
            else:
                bucket_dist[bits] = 0

        # Print results
        print(f"  Latency (avg):  {avg_latency_ms:.3f} ms")
        print(f"  Latency (p50):  {p50_latency_ms:.3f} ms")
        print(f"  Latency (p95):  {p95_latency_ms:.3f} ms")
        print(f"  Avg precision:  {avg_bits:.2f} bits")
        print(f"  Memory ratio:   {memory_ratio:.3f}")
        print(f"  Bucket distribution:", end="")
        for bits in sorted(bucket_dist.keys()):
            count = bucket_dist[bits]
            pct = 100 * count / ctx_len if ctx_len > 0 else 0
            print(f" {bits}b:{count}({pct:.1f}%)", end="")
        print()

        results.append({
            'ctx_len': ctx_len,
            'latency_ms': avg_latency_ms,
            'p50_ms': p50_latency_ms,
            'p95_ms': p95_latency_ms,
            'avg_bits': avg_bits,
            'memory_ratio': memory_ratio,
            'bucket_dist': bucket_dist,
        })

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Context':<10} {'Latency(ms)':<15} {'P50(ms)':<12} {'P95(ms)':<12} {'Avg bits':<10} {'Mem ratio':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['ctx_len']:<10} {r['latency_ms']:<15.3f} {r['p50_ms']:<12.3f} {r['p95_ms']:<12.3f} {r['avg_bits']:<10.2f} {r['memory_ratio']:<10.3f}")
    print("="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
