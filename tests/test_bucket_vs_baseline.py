#!/usr/bin/env python3
"""
Benchmark comparing SmartKV bucket kernel vs standard INT8 and FP16 baselines.

Compares:
1. SmartKV (mixed 2/4/8-bit) with bucket kernel
2. Standard INT8 (non-bucketed) with legacy kernel
3. FP16 baseline with PyTorch standard attention
"""

import pytest
import torch
from time import perf_counter

from smartkv.core.cache import SmartKVCache
from smartkv.kernels import quantized_attention, quantized_attention_bucketed, CUDA_AVAILABLE

CUDA_AVAILABLE = CUDA_AVAILABLE and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_smartkv_vs_int8_vs_fp16():
    """
    Compare SmartKV bucket kernel vs standard INT8 vs FP16 baseline.

    Tests correctness and measures:
    - Latency (avg, p50, p95)
    - Memory usage
    - Precision distribution (SmartKV only)
    """
    device = torch.device('cuda')
    torch.manual_seed(42)

    # Configuration
    num_heads = 32
    head_dim = 128
    context_lengths = [512, 1024, 2048, 4096]
    memory_budget = 0.3
    available_bits = [2, 4, 8]

    print("\n" + "="*80)
    print("SMARTKV BUCKET KERNEL VS STANDARD INT8 VS FP16 BASELINE")
    print("="*80)
    print(f"Configuration: {num_heads} heads × {head_dim} dim")
    print(f"SmartKV budget: {memory_budget}, bits: {available_bits}")
    print("="*80)

    all_results = []

    for ctx_len in context_lengths:
        print(f"\n{'='*80}")
        print(f"Context length: {ctx_len}")
        print('='*80)

        # ================================================================
        # 1. Create SmartKV cache (mixed precision, bucketed)
        # ================================================================
        smartkv_cache = SmartKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_budget=memory_budget,
            available_bits=available_bits,
            device='cuda',
            use_bucketed_layout=True,
            use_bit_packing=True,
        )

        # ================================================================
        # 2. Create INT8 baseline cache (non-bucketed, legacy kernel)
        # ================================================================
        int8_cache = SmartKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_budget=0.5,
            available_bits=[8],
            device='cuda',
            use_bucketed_layout=False,  # Non-bucketed for legacy kernel
            use_bit_packing=False,      # No packing for 8-bit
        )

        # ================================================================
        # 3. Store FP16 KV for baseline
        # ================================================================
        fp16_k_list = []
        fp16_v_list = []

        # Populate all caches with same random data
        chunk_size = 128
        for start in range(0, ctx_len, chunk_size):
            size = min(chunk_size, ctx_len - start)
            token_ids = list(range(start, start + size))

            # Generate same random data for all approaches
            # Note: Quantization expects float32 input
            k = torch.randn(size, num_heads, head_dim, device=device, dtype=torch.float32)
            v = torch.randn(size, num_heads, head_dim, device=device, dtype=torch.float32)

            # SmartKV: allocate adaptive precision
            smartkv_cache.allocate_precision(0, token_ids)
            smartkv_cache.quantize_and_store_batch(0, token_ids, k, v)

            # INT8: force all to 8-bit
            for tid in token_ids:
                int8_cache.precision_map[tid] = 8
            int8_cache.quantize_and_store_batch(0, token_ids, k, v)

            # FP16: store directly
            fp16_k_list.append(k)
            fp16_v_list.append(v)

        # Concatenate FP16 KV
        fp16_k = torch.cat(fp16_k_list, dim=0)  # [ctx_len, num_heads, head_dim]
        fp16_v = torch.cat(fp16_v_list, dim=0)

        # Query tensor (float32 for quantized kernels)
        query = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=torch.float32)

        # ================================================================
        # Benchmark 1: SmartKV (bucketed, mixed precision)
        # ================================================================
        print("\n[1] SmartKV (mixed 2/4/8-bit, bucket kernel):")

        smartkv_buckets = smartkv_cache.get_bucket_views(0)

        # Correctness check
        smartkv_out = quantized_attention_bucketed(query, smartkv_buckets, use_cuda=True)
        assert not torch.isnan(smartkv_out).any(), "SmartKV output contains NaN"
        assert not torch.isinf(smartkv_out).any(), "SmartKV output contains Inf"

        # Warmup
        for _ in range(10):
            _ = quantized_attention_bucketed(query, smartkv_buckets, use_cuda=True)
        torch.cuda.synchronize()

        # Benchmark
        iters = 100
        times = []
        for _ in range(iters):
            start = perf_counter()
            _ = quantized_attention_bucketed(query, smartkv_buckets, use_cuda=True)
            torch.cuda.synchronize()
            times.append(perf_counter() - start)

        smartkv_latency = sum(times) / len(times) * 1000
        smartkv_p50 = sorted(times)[len(times)//2] * 1000
        smartkv_p95 = sorted(times)[int(len(times)*0.95)] * 1000

        smartkv_stats = smartkv_cache.get_memory_stats()
        smartkv_avg_bits = smartkv_stats.get('avg_bits', 0)
        smartkv_memory_ratio = smartkv_stats.get('memory_ratio_true', 0)

        # Bucket distribution
        bucket_dist = {}
        for bits in available_bits:
            if bits in smartkv_buckets:
                bucket_dist[bits] = len(smartkv_buckets[bits]['token_ids'])
            else:
                bucket_dist[bits] = 0

        print(f"  Latency (avg): {smartkv_latency:.3f} ms")
        print(f"  Latency (p50): {smartkv_p50:.3f} ms")
        print(f"  Latency (p95): {smartkv_p95:.3f} ms")
        print(f"  Avg bits:      {smartkv_avg_bits:.2f}")
        print(f"  Memory ratio:  {smartkv_memory_ratio:.3f}")
        print(f"  Bucket dist:  ", end="")
        for bits in sorted(bucket_dist.keys()):
            count = bucket_dist[bits]
            pct = 100 * count / ctx_len if ctx_len > 0 else 0
            print(f" {bits}b:{count}({pct:.1f}%)", end="")
        print()

        # ================================================================
        # Benchmark 2: Standard INT8 (non-bucketed, legacy kernel)
        # ================================================================
        print("\n[2] Standard INT8 (non-bucketed, legacy kernel):")

        int8_data = int8_cache.retrieve_all_quantized(0)

        # Correctness check
        int8_out = quantized_attention(
            query,
            int8_data['k_qx'].unsqueeze(0),
            int8_data['k_scale'].unsqueeze(0),
            int8_data['v_qx'].unsqueeze(0),
            int8_data['v_scale'].unsqueeze(0),
            use_cuda=True,
        )
        assert not torch.isnan(int8_out).any(), "INT8 output contains NaN"
        assert not torch.isinf(int8_out).any(), "INT8 output contains Inf"

        # Warmup
        for _ in range(10):
            _ = quantized_attention(
                query,
                int8_data['k_qx'].unsqueeze(0),
                int8_data['k_scale'].unsqueeze(0),
                int8_data['v_qx'].unsqueeze(0),
                int8_data['v_scale'].unsqueeze(0),
                use_cuda=True,
            )
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(iters):
            start = perf_counter()
            _ = quantized_attention(
                query,
                int8_data['k_qx'].unsqueeze(0),
                int8_data['k_scale'].unsqueeze(0),
                int8_data['v_qx'].unsqueeze(0),
                int8_data['v_scale'].unsqueeze(0),
                use_cuda=True,
            )
            torch.cuda.synchronize()
            times.append(perf_counter() - start)

        int8_latency = sum(times) / len(times) * 1000
        int8_p50 = sorted(times)[len(times)//2] * 1000
        int8_p95 = sorted(times)[int(len(times)*0.95)] * 1000

        print(f"  Latency (avg): {int8_latency:.3f} ms")
        print(f"  Latency (p50): {int8_p50:.3f} ms")
        print(f"  Latency (p95): {int8_p95:.3f} ms")
        print(f"  Memory ratio:  0.500 (uniform INT8)")

        # ================================================================
        # Benchmark 3: FP16 baseline (PyTorch standard attention)
        # ================================================================
        print("\n[3] FP16 baseline (PyTorch standard attention):")

        # Reshape for PyTorch attention: [B, H, seq_len, head_dim]
        # Convert to FP16 for fair FP16 baseline comparison
        fp16_k_attn = fp16_k.unsqueeze(0).transpose(1, 2).half()  # [1, num_heads, ctx_len, head_dim]
        fp16_v_attn = fp16_v.unsqueeze(0).transpose(1, 2).half()
        query_attn = query.half()  # Convert to FP16 for FP16 baseline

        # Correctness check
        fp16_out = torch.nn.functional.scaled_dot_product_attention(
            query_attn, fp16_k_attn, fp16_v_attn
        )
        assert not torch.isnan(fp16_out).any(), "FP16 output contains NaN"
        assert not torch.isinf(fp16_out).any(), "FP16 output contains Inf"

        # Warmup
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(
                query_attn, fp16_k_attn, fp16_v_attn
            )
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(iters):
            start = perf_counter()
            _ = torch.nn.functional.scaled_dot_product_attention(
                query_attn, fp16_k_attn, fp16_v_attn
            )
            torch.cuda.synchronize()
            times.append(perf_counter() - start)

        fp16_latency = sum(times) / len(times) * 1000
        fp16_p50 = sorted(times)[len(times)//2] * 1000
        fp16_p95 = sorted(times)[int(len(times)*0.95)] * 1000

        print(f"  Latency (avg): {fp16_latency:.3f} ms")
        print(f"  Latency (p50): {fp16_p50:.3f} ms")
        print(f"  Latency (p95): {fp16_p95:.3f} ms")
        print(f"  Memory ratio:  1.000 (FP16 baseline)")

        # ================================================================
        # Summary for this context length
        # ================================================================
        print(f"\n{'-'*80}")
        print(f"Summary (ctx={ctx_len}):")
        print(f"  SmartKV vs INT8 speedup:    {int8_latency/smartkv_latency:.2f}x")
        print(f"  SmartKV vs FP16 speedup:    {fp16_latency/smartkv_latency:.2f}x")
        print(f"  SmartKV memory savings:     {(1.0-smartkv_memory_ratio)*100:.1f}% vs FP16")
        print(f"  SmartKV memory savings:     {(0.5-smartkv_memory_ratio)/0.5*100:.1f}% vs INT8")
        print(f"{'-'*80}")

        all_results.append({
            'ctx_len': ctx_len,
            'smartkv_latency_ms': smartkv_latency,
            'smartkv_p50_ms': smartkv_p50,
            'smartkv_p95_ms': smartkv_p95,
            'smartkv_avg_bits': smartkv_avg_bits,
            'smartkv_memory_ratio': smartkv_memory_ratio,
            'int8_latency_ms': int8_latency,
            'int8_p50_ms': int8_p50,
            'int8_p95_ms': int8_p95,
            'fp16_latency_ms': fp16_latency,
            'fp16_p50_ms': fp16_p50,
            'fp16_p95_ms': fp16_p95,
            'bucket_dist': bucket_dist,
        })

    # ================================================================
    # Final summary table
    # ================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Context':<10} {'SmartKV(ms)':<14} {'INT8(ms)':<12} {'FP16(ms)':<12} {'Avg bits':<10} {'Mem ratio':<10}")
    print("-"*80)
    for r in all_results:
        print(f"{r['ctx_len']:<10} "
              f"{r['smartkv_latency_ms']:<14.3f} "
              f"{r['int8_latency_ms']:<12.3f} "
              f"{r['fp16_latency_ms']:<12.3f} "
              f"{r['smartkv_avg_bits']:<10.2f} "
              f"{r['smartkv_memory_ratio']:<10.3f}")
    print("="*80)
    print()

    # Calculate average speedups
    avg_smartkv_vs_int8 = sum(r['int8_latency_ms']/r['smartkv_latency_ms'] for r in all_results) / len(all_results)
    avg_smartkv_vs_fp16 = sum(r['fp16_latency_ms']/r['smartkv_latency_ms'] for r in all_results) / len(all_results)
    avg_memory_savings_vs_fp16 = sum((1.0-r['smartkv_memory_ratio'])*100 for r in all_results) / len(all_results)
    avg_memory_savings_vs_int8 = sum((0.5-r['smartkv_memory_ratio'])/0.5*100 for r in all_results) / len(all_results)

    print("Average across all context lengths:")
    print(f"  SmartKV vs INT8 speedup:     {avg_smartkv_vs_int8:.2f}x")
    print(f"  SmartKV vs FP16 speedup:     {avg_smartkv_vs_fp16:.2f}x")
    print(f"  SmartKV memory savings:      {avg_memory_savings_vs_fp16:.1f}% vs FP16")
    print(f"  SmartKV memory savings:      {avg_memory_savings_vs_int8:.1f}% vs INT8")
    print("="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
