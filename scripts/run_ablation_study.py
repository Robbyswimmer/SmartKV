"""
Memory Budget Ablation Study Runner.

Systematically evaluates SmartKV across different memory budgets, bit configurations,
context lengths, and bit-packing settings.

Usage:
    python scripts/run_ablation_study.py \\
        --budget 0.5 \\
        --bits 2,3,4,8 \\
        --context-length 2048 \\
        --use-bit-packing \\
        --device cuda:0 \\
        --output results/ablation_b50_2348_ctx2048.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="SmartKV Memory Budget Ablation Study")

    # Configuration
    parser.add_argument("--budget", type=float, required=True,
                        help="Memory budget (fraction of FP16, e.g., 0.5)")
    parser.add_argument("--bits", type=str, required=True,
                        help="Comma-separated bit widths (e.g., '2,3,4,8')")
    parser.add_argument("--context-length", type=int, required=True,
                        help="Context length to test")
    parser.add_argument("--use-bit-packing", action="store_true",
                        help="Enable bit-packing for sub-byte storage")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on (e.g., 'cuda:0', 'cpu')")

    # Model (optional - use dummy if not provided)
    parser.add_argument("--model-name", type=str, default=None,
                        help="HuggingFace model name (if None, uses dummy cache)")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of layers (for dummy cache)")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of attention heads (for dummy cache)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension (for dummy cache)")

    # Generation settings
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of generation samples for metrics")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max new tokens to generate")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")

    return parser.parse_args()


def create_dummy_cache(args) -> "SmartKVCache":
    """Create SmartKV cache for testing (no actual model)."""
    from smartkv.core.cache import SmartKVCache

    bits_list = [int(b) for b in args.bits.split(',')]

    cache = SmartKVCache(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        memory_budget=args.budget,
        available_bits=bits_list,
        device=args.device,
        use_bit_packing=args.use_bit_packing
    )

    return cache


def measure_cache_memory(cache, context_length: int, device: str) -> Dict[str, float]:
    """Measure actual GPU/CPU memory usage of cache."""
    import torch

    # Store dummy KV to fill cache
    num_heads = cache.num_heads
    head_dim = cache.head_dim

    torch.manual_seed(42)

    # Store context_length tokens
    chunk_size = min(128, context_length)
    for i in range(0, context_length, chunk_size):
        chunk_len = min(chunk_size, context_length - i)
        k_chunk = torch.randn(chunk_len, num_heads, head_dim)
        v_chunk = torch.randn(chunk_len, num_heads, head_dim)

        if device.startswith('cuda'):
            k_chunk = k_chunk.cuda()
            v_chunk = v_chunk.cuda()

        token_ids = list(range(i, i + chunk_len))

        cache.quantize_and_store_batch(
            layer_idx=0,  # Just test one layer
            token_ids=token_ids,
            k_batch=k_chunk,
            v_batch=v_chunk
        )

    # Get memory stats
    stats = cache.get_memory_stats()

    # Measure actual device memory
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    else:
        import psutil
        process = psutil.Process()
        allocated = process.memory_info().rss / 1024 ** 3
        reserved = allocated

    return {
        'memory_ratio': stats['memory_ratio'],
        'avg_bits': stats['avg_bits'],
        'num_tokens': stats['num_tokens'],
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'precision_distribution': stats['precision_distribution'],
    }


def measure_throughput(cache, context_length: int, device: str, num_samples: int = 10) -> Dict[str, float]:
    """Measure tokens/sec throughput for retrieval."""
    import torch

    # Pre-fill cache (done in measure_cache_memory, so cache already has data)

    # Time retrieval
    times = []

    for _ in range(num_samples):
        if device.startswith('cuda'):
            torch.cuda.synchronize()

        start = time.time()

        # Retrieve all KV (simulates attention lookup)
        keys, values = cache.retrieve_all(layer_idx=0)

        if device.startswith('cuda'):
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    # Throughput = tokens / sec
    throughput = context_length / avg_time if avg_time > 0 else 0

    return {
        'avg_latency_ms': avg_time * 1000,
        'std_latency_ms': std_time * 1000,
        'throughput_tps': throughput,
    }


def compute_baseline_memory(context_length: int, num_layers: int, num_heads: int, head_dim: int) -> float:
    """Compute FP16 baseline memory in GB."""
    # FP16: 2 bytes per value
    # KV cache: 2 (K+V) * num_layers * num_heads * head_dim * context_length
    bytes_fp16 = 2 * 2 * num_layers * num_heads * head_dim * context_length
    return bytes_fp16 / 1024 ** 3


def run_ablation(args) -> Dict[str, Any]:
    """Run single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Running Ablation:")
    print(f"  Budget: {args.budget:.1%}")
    print(f"  Bits: {args.bits}")
    print(f"  Context: {args.context_length}")
    print(f"  Packing: {args.use_bit_packing}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # Create cache
    cache = create_dummy_cache(args)

    print(f"Cache created:")
    print(f"  Configured budget: {cache.memory_budget:.1%}")
    print(f"  Min achievable budget: {cache.min_budget:.1%}")
    print(f"  Packing enabled: {cache.use_packing}")

    # Measure memory
    print(f"\nMeasuring memory usage...")
    memory_metrics = measure_cache_memory(cache, args.context_length, args.device)

    # Measure throughput
    print(f"Measuring throughput...")
    throughput_metrics = measure_throughput(cache, args.context_length, args.device, args.num_samples)

    # Compute baseline
    baseline_memory_gb = compute_baseline_memory(
        args.context_length,
        args.num_layers,
        args.num_heads,
        args.head_dim
    )

    # Combine results
    results = {
        'config': {
            'budget': args.budget,
            'bits': args.bits,
            'context_length': args.context_length,
            'use_bit_packing': args.use_bit_packing,
            'device': args.device,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'head_dim': args.head_dim,
        },
        'memory': memory_metrics,
        'performance': throughput_metrics,
        'baseline': {
            'fp16_memory_gb': baseline_memory_gb,
        },
        'compression': {
            'actual_vs_fp16': memory_metrics['allocated_gb'] / baseline_memory_gb if baseline_memory_gb > 0 else 0,
            'theoretical_ratio': memory_metrics['memory_ratio'],
        }
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Memory ratio (theoretical): {memory_metrics['memory_ratio']:.1%}")
    print(f"  Actual vs FP16: {results['compression']['actual_vs_fp16']:.1%}")
    print(f"  Average bits: {memory_metrics['avg_bits']:.2f}")
    print(f"  Throughput: {throughput_metrics['throughput_tps']:.0f} tokens/sec")
    print(f"  Latency: {throughput_metrics['avg_latency_ms']:.2f} ± {throughput_metrics['std_latency_ms']:.2f} ms")
    print(f"  GPU allocated: {memory_metrics['allocated_gb']:.3f} GB")
    print(f"  FP16 baseline: {baseline_memory_gb:.3f} GB")
    print(f"  Precision distribution:")
    for bits, count in memory_metrics['precision_distribution'].items():
        pct = count / memory_metrics['num_tokens'] * 100 if memory_metrics['num_tokens'] > 0 else 0
        print(f"    {bits}: {count} tokens ({pct:.1f}%)")
    print(f"{'='*60}\n")

    return results


def main():
    args = parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run ablation
    results = run_ablation(args)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
