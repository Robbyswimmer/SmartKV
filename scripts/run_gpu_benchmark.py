"""
GPU Performance Benchmark Runner.

Compares SmartKV performance against baselines:
- FP16 (no quantization)
- Uniform 8-bit
- Uniform 4-bit
- SmartKV mixed precision

Usage:
    python scripts/run_gpu_benchmark.py \\
        --baseline fp16 \\
        --context-lengths 512,1024,2048,4096 \\
        --device cuda:0 \\
        --output results/benchmark_fp16.csv
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="SmartKV GPU Performance Benchmark")

    # Baseline to compare
    parser.add_argument("--baseline", type=str, required=True,
                        choices=["fp16", "uniform_8bit", "uniform_4bit", "smartkv_mixed"],
                        help="Baseline configuration to test")

    # Test parameters
    parser.add_argument("--context-lengths", type=str, default="512,1024,2048,4096",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")

    # Model config
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension")

    # Benchmark settings
    parser.add_argument("--num-warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=100,
                        help="Number of benchmark iterations")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")

    return parser.parse_args()


def get_baseline_config(baseline: str) -> Dict[str, Any]:
    """Get configuration for each baseline."""
    configs = {
        "fp16": {
            "use_smartkv": False,
            "dtype": torch.float16,
            "description": "FP16 baseline (no quantization)"
        },
        "uniform_8bit": {
            "use_smartkv": True,
            "memory_budget": 0.5,  # 8-bit = 50% of FP16
            "available_bits": [8],
            "use_bit_packing": False,
            "description": "Uniform 8-bit quantization"
        },
        "uniform_4bit": {
            "use_smartkv": True,
            "memory_budget": 0.25,  # 4-bit = 25% of FP16
            "available_bits": [4],
            "use_bit_packing": True,
            "description": "Uniform 4-bit quantization"
        },
        "smartkv_mixed": {
            "use_smartkv": True,
            "memory_budget": 0.4,
            "available_bits": [2, 3, 4, 8],
            "use_bit_packing": True,
            "description": "SmartKV mixed precision (2/3/4/8-bit)"
        },
    }
    return configs[baseline]


def benchmark_fp16_attention(context_length: int, num_heads: int, head_dim: int,
                              num_iterations: int, device: str) -> Dict[str, float]:
    """Benchmark standard FP16 attention."""
    torch.manual_seed(42)

    # Create dummy KV cache
    past_k = torch.randn(1, num_heads, context_length, head_dim, dtype=torch.float16, device=device)
    past_v = torch.randn(1, num_heads, context_length, head_dim, dtype=torch.float16, device=device)

    # Query (single token)
    query = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(10):
        _ = torch.nn.functional.scaled_dot_product_attention(query, past_k, past_v)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()

        output = torch.nn.functional.scaled_dot_product_attention(query, past_k, past_v)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    # Memory usage
    memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 3  # GB

    return {
        'avg_latency_ms': np.mean(times) * 1000,
        'std_latency_ms': np.std(times) * 1000,
        'min_latency_ms': np.min(times) * 1000,
        'p50_latency_ms': np.percentile(times, 50) * 1000,
        'p95_latency_ms': np.percentile(times, 95) * 1000,
        'p99_latency_ms': np.percentile(times, 99) * 1000,
        'memory_gb': memory_allocated,
        'throughput_tps': context_length / (np.mean(times)) if np.mean(times) > 0 else 0,
    }


def benchmark_smartkv_attention(context_length: int, num_heads: int, head_dim: int,
                                 num_iterations: int, device: str, config: Dict) -> Dict[str, float]:
    """Benchmark SmartKV quantized attention."""
    from smartkv.core.cache import SmartKVCache

    torch.manual_seed(42)

    # Create SmartKV cache
    cache = SmartKVCache(
        num_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
        memory_budget=config['memory_budget'],
        available_bits=config['available_bits'],
        device=device,
        use_bit_packing=config.get('use_bit_packing', False)
    )

    # Store context
    chunk_size = 128
    for i in range(0, context_length, chunk_size):
        chunk_len = min(chunk_size, context_length - i)
        k_chunk = torch.randn(chunk_len, num_heads, head_dim).to(device)
        v_chunk = torch.randn(chunk_len, num_heads, head_dim).to(device)
        token_ids = list(range(i, i + chunk_len))

        cache.quantize_and_store_batch(0, token_ids, k_chunk, v_chunk)

    # Query (single token)
    query = torch.randn(1, num_heads, head_dim).to(device)

    # Check if we can use fused GPU kernel
    try:
        from smartkv.kernels import quantized_attention, CUDA_AVAILABLE
        use_fused = CUDA_AVAILABLE and device.startswith('cuda')
    except ImportError:
        use_fused = False

    # Warmup
    for _ in range(10):
        if use_fused:
            quant_data = cache.retrieve_all_quantized(0)
            k_qx = quant_data['k_qx'].unsqueeze(0).to(device)
            k_scale = quant_data['k_scale'].unsqueeze(0).to(device)
            v_qx = quant_data['v_qx'].unsqueeze(0).to(device)
            v_scale = quant_data['v_scale'].unsqueeze(0).to(device)
            _ = quantized_attention(query, k_qx, k_scale, v_qx, v_scale, use_cuda=True)
        else:
            keys, values = cache.retrieve_all(0)
            keys_expanded = keys.unsqueeze(0).transpose(1, 2).unsqueeze(2)
            values_expanded = values.unsqueeze(0).transpose(1, 2).unsqueeze(2)
            query_expanded = query.unsqueeze(0).unsqueeze(2)
            _ = torch.nn.functional.scaled_dot_product_attention(
                query_expanded, keys_expanded, values_expanded
            )

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()

        if use_fused:
            quant_data = cache.retrieve_all_quantized(0)
            k_qx = quant_data['k_qx'].unsqueeze(0).to(device)
            k_scale = quant_data['k_scale'].unsqueeze(0).to(device)
            v_qx = quant_data['v_qx'].unsqueeze(0).to(device)
            v_scale = quant_data['v_scale'].unsqueeze(0).to(device)
            output = quantized_attention(query, k_qx, k_scale, v_qx, v_scale, use_cuda=True)
        else:
            keys, values = cache.retrieve_all(0)
            keys_expanded = keys.unsqueeze(0).transpose(1, 2).unsqueeze(2)
            values_expanded = values.unsqueeze(0).transpose(1, 2).unsqueeze(2)
            query_expanded = query.unsqueeze(0).unsqueeze(2)
            output = torch.nn.functional.scaled_dot_product_attention(
                query_expanded, keys_expanded, values_expanded
            )

        torch.cuda.synchronize()
        times.append(time.time() - start)

    # Memory usage
    memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 3

    # Get cache stats
    stats = cache.get_memory_stats()

    return {
        'avg_latency_ms': np.mean(times) * 1000,
        'std_latency_ms': np.std(times) * 1000,
        'min_latency_ms': np.min(times) * 1000,
        'p50_latency_ms': np.percentile(times, 50) * 1000,
        'p95_latency_ms': np.percentile(times, 95) * 1000,
        'p99_latency_ms': np.percentile(times, 99) * 1000,
        'memory_gb': memory_allocated,
        'throughput_tps': context_length / (np.mean(times)) if np.mean(times) > 0 else 0,
        'avg_bits': stats['avg_bits'],
        'memory_ratio': stats['memory_ratio'],
        'use_fused_kernel': use_fused,
    }


def run_benchmark(args) -> List[Dict[str, Any]]:
    """Run benchmark across all context lengths."""
    config = get_baseline_config(args.baseline)
    context_lengths = [int(c) for c in args.context_lengths.split(',')]

    results = []

    print(f"\n{'='*60}")
    print(f"Benchmarking: {config['description']}")
    print(f"Device: {args.device}")
    print(f"Context lengths: {context_lengths}")
    print(f"Iterations: {args.num_iterations} (warmup: {args.num_warmup})")
    print(f"{'='*60}\n")

    for ctx_len in context_lengths:
        print(f"Testing context length: {ctx_len}...")

        # Clear GPU cache
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.device)

        # Run benchmark
        if config['use_smartkv']:
            metrics = benchmark_smartkv_attention(
                ctx_len, args.num_heads, args.head_dim,
                args.num_iterations, args.device, config
            )
        else:
            metrics = benchmark_fp16_attention(
                ctx_len, args.num_heads, args.head_dim,
                args.num_iterations, args.device
            )

        result = {
            'baseline': args.baseline,
            'context_length': ctx_len,
            **metrics
        }

        results.append(result)

        # Print summary
        print(f"  Latency: {metrics['avg_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f} ms")
        print(f"  Throughput: {metrics['throughput_tps']:.0f} tokens/sec")
        print(f"  Memory: {metrics['memory_gb']:.3f} GB")
        if 'avg_bits' in metrics:
            print(f"  Avg bits: {metrics['avg_bits']:.2f}")
            print(f"  Memory ratio: {metrics['memory_ratio']:.1%}")
            print(f"  Fused kernel: {metrics['use_fused_kernel']}")
        print()

    return results


def save_results(results: List[Dict], output_path: str):
    """Save results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_path}")


def main():
    args = parse_args()

    results = run_benchmark(args)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
