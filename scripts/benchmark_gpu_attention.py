#!/usr/bin/env python3
"""Compare SmartKV vs uniform INT8 attention using CUDA fused kernel."""

import argparse
import json
from typing import Dict
from time import perf_counter

import torch

from smartkv.core.cache import SmartKVCache
from smartkv.kernels import quantized_attention, CUDA_AVAILABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartKV vs uniform INT8 GPU attention benchmark")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="Number of tokens in the KV cache")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--memory-budget", type=float, default=0.3,
                        help="SmartKV memory budget (fraction of FP16)")
    parser.add_argument("--iters", type=int, default=200,
                        help="Iterations for timing loop")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-packing", action="store_true", default=False,
                        help="Enable bit-packing for SmartKV")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Print JSON summary only")
    return parser.parse_args()


def build_cache(name: str, num_heads: int, head_dim: int, seq_len: int,
                budget: float, device: torch.device, packing: bool) -> SmartKVCache:
    if name == "smartkv":
        cache = SmartKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_budget=budget,
            available_bits=[2, 3, 4, 8],
            device=device.type,
            use_bit_packing=packing,
        )
    elif name == "uniform_8bit":
        cache = SmartKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_budget=0.5,
            available_bits=[8],
            device=device.type,
            use_bit_packing=False,
        )
    else:
        raise ValueError(f"Unknown cache type: {name}")

    chunk = 128
    torch.manual_seed(123)
    for start in range(0, seq_len, chunk):
        size = min(chunk, seq_len - start)
        token_ids = list(range(start, start + size))
        k = torch.randn(size, num_heads, head_dim, device=device)
        v = torch.randn(size, num_heads, head_dim, device=device)

        if name == "smartkv":
            cache.allocate_precision(0, token_ids)
        else:
            for tid in token_ids:
                cache.precision_map[tid] = max(cache.available_bits)

        cache.quantize_and_store_batch(0, token_ids, k, v)
    return cache


def run_attention(cache: SmartKVCache, query: torch.Tensor, use_cuda: bool) -> torch.Tensor:
    quant = cache.retrieve_all_quantized(0)
    k_qx = quant['k_qx'].unsqueeze(0).to(query.device)
    v_qx = quant['v_qx'].unsqueeze(0).to(query.device)
    k_scale = quant['k_scale'].unsqueeze(0).to(query.device)
    v_scale = quant['v_scale'].unsqueeze(0).to(query.device)
    return quantized_attention(query, k_qx, k_scale, v_qx, v_scale, use_cuda=use_cuda)


def benchmark_cache(name: str, cache: SmartKVCache, query: torch.Tensor,
                    warmup: int, iters: int, device: torch.device) -> Dict[str, float]:
    use_cuda = CUDA_AVAILABLE and device.type == 'cuda'
    if not use_cuda:
        raise RuntimeError("CUDA fused kernel not available; build smartkv_cuda first")

    for _ in range(warmup):
        _ = run_attention(cache, query, use_cuda=True)
    torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        start = perf_counter()
        _ = run_attention(cache, query, use_cuda=True)
        torch.cuda.synchronize(device)
        times.append(perf_counter() - start)

    avg = sum(times) / len(times)
    return {
        "avg_latency_ms": avg * 1e3,
        "tokens_per_s": (query.shape[0] * query.shape[2]) / avg,
        "min_latency_ms": min(times) * 1e3,
        "max_latency_ms": max(times) * 1e3,
        "iterations": iters,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError("Please run this benchmark on a CUDA device.")
    if not CUDA_AVAILABLE:
        raise RuntimeError("smartkv_cuda extension not available. Rebuild before running.")

    torch.manual_seed(args.seed)

    caches = {
        "smartkv": build_cache(
            "smartkv",
            args.num_heads,
            args.head_dim,
            args.context_length,
            args.memory_budget,
            device,
            args.enable_packing,
        ),
        "uniform_8bit": build_cache(
            "uniform_8bit",
            args.num_heads,
            args.head_dim,
            args.context_length,
            budget=0.5,
            device=device,
            packing=False,
        ),
    }

    query = torch.randn(1, args.num_heads, 1, args.head_dim, device=device)

    results: Dict[str, Dict[str, float]] = {}
    for name, cache in caches.items():
        metrics = benchmark_cache(name, cache, query, args.warmup, args.iters, device)
        if name == "smartkv":
            stats = cache.get_memory_stats()
            metrics.update({
                "memory_ratio_true": stats.get("memory_ratio_true"),
                "avg_bits": stats.get("avg_bits"),
            })
        results[name] = metrics

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for name, metrics in results.items():
        print(f"\n{name.upper()} benchmark")
        print("=" * 40)
        print(f"Average latency : {metrics['avg_latency_ms']:.3f} ms")
        print(f"Throughput      : {metrics['tokens_per_s']:.1f} tokens/s")
        print(f"Min latency     : {metrics['min_latency_ms']:.3f} ms")
        print(f"Max latency     : {metrics['max_latency_ms']:.3f} ms")
        if 'avg_bits' in metrics:
            print(f"Avg bits        : {metrics['avg_bits']:.2f}")
            print(f"Memory ratio    : {metrics['memory_ratio_true']:.3f}")
    print()


if __name__ == "__main__":
    main()
