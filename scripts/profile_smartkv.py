#!/usr/bin/env python3
"""Micro-benchmark harness for SmartKV fused attention and cache utilities.

Run with optional CLI arguments to exercise the CUDA kernel as well as
SmartKVCache quantize/store/retrieve pathways. Designed for use on GPU nodes
but falls back to CPU execution when CUDA is unavailable.
"""

import argparse
import json
import time
from typing import Dict
import os
import sys

import torch

# Diagnostic: Check environment before importing smartkv
print(f"[DIAGNOSTIC] LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:200]}...")
print(f"[DIAGNOSTIC] Python sys.path[0]: {sys.path[0]}")
print(f"[DIAGNOSTIC] Current working directory: {os.getcwd()}")

# Try direct import first to diagnose
try:
    import smartkv_cuda
    print(f"[DIAGNOSTIC] Direct smartkv_cuda import: SUCCESS")
    print(f"[DIAGNOSTIC] Module location: {smartkv_cuda.__file__}")
except ImportError as e:
    print(f"[DIAGNOSTIC] Direct smartkv_cuda import: FAILED - {e}")

from smartkv.kernels import quantized_attention, CUDA_AVAILABLE
from smartkv.core.cache import SmartKVCache

print(f"[DIAGNOSTIC] CUDA_AVAILABLE after smartkv.kernels import: {CUDA_AVAILABLE}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_attention(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    dtype = torch.float32

    torch.manual_seed(args.seed)

    query = torch.randn(
        args.batch_size,
        args.num_heads,
        args.q_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    key_int8 = torch.randint(
        -128,
        127,
        (args.batch_size, args.num_heads, args.k_len, args.head_dim),
        device=device,
        dtype=torch.int8,
    )
    value_int8 = torch.randint(
        -128,
        127,
        (args.batch_size, args.num_heads, args.k_len, args.head_dim),
        device=device,
        dtype=torch.int8,
    )
    key_scale = torch.rand(
        args.batch_size,
        args.num_heads,
        args.k_len,
        device=device,
        dtype=dtype,
    )
    value_scale = torch.rand(
        args.batch_size,
        args.num_heads,
        args.k_len,
        device=device,
        dtype=dtype,
    )

    attention_mask = None
    if args.use_mask:
        attention_mask = torch.zeros(
            args.batch_size,
            1,
            args.q_len,
            args.k_len,
            device=device,
            dtype=dtype,
        )
        attention_mask[..., args.k_len // 2 :] = -1e9

    # Warmup
    for _ in range(args.warmup):
        out = quantized_attention(
            query,
            key_int8,
            key_scale,
            value_int8,
            value_scale,
            attention_mask=attention_mask,
            use_cuda=args.use_cuda,
        )
        if not args.disable_validation:
            ref = quantized_attention(
                query,
                key_int8,
                key_scale,
                value_int8,
                value_scale,
                attention_mask=attention_mask,
                use_cuda=False,
            )
            max_err = (out - ref).abs().max().item()
            if max_err > 1e-3:
                raise RuntimeError(f"Fused attention validation failed (max error {max_err:.4e})")

    _sync(device)
    start = time.perf_counter()
    for _ in range(args.iters):
        quantized_attention(
            query,
            key_int8,
            key_scale,
            value_int8,
            value_scale,
            attention_mask=attention_mask,
            use_cuda=args.use_cuda,
        )
    _sync(device)
    elapsed = time.perf_counter() - start

    tokens = args.batch_size * args.q_len * args.iters
    throughput = tokens / elapsed if elapsed > 0 else float("nan")

    return {
        "elapsed_s": elapsed,
        "tokens": float(tokens),
        "tokens_per_s": throughput,
        "device": args.device,
        "use_cuda": bool(args.use_cuda and CUDA_AVAILABLE and device.type == "cuda"),
    }


def exercise_cache(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    cache = SmartKVCache(
        num_layers=1,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        memory_budget=args.cache_budget,
        device=device.type,
        use_bit_packing=args.enable_packing,
        enable_forecast=args.enable_forecast,
        forecast_history=args.forecast_history,
        forecast_update_interval=args.forecast_update_interval,
        forecast_blend=args.forecast_blend,
        forecast_lr=args.forecast_lr,
    )

    seq_len = args.k_len
    torch.manual_seed(args.seed + 1)
    k_batch = torch.randn(seq_len, args.num_heads, args.head_dim, device=device)
    v_batch = torch.randn(seq_len, args.num_heads, args.head_dim, device=device)
    token_ids = list(range(seq_len))

    _sync(device)
    start_store = time.perf_counter()
    cache.quantize_and_store_batch(0, token_ids, k_batch, v_batch)
    _sync(device)
    store_time = time.perf_counter() - start_store

    _sync(device)
    start_retrieve = time.perf_counter()
    cache.retrieve_all_quantized(0)
    _sync(device)
    retrieve_time = time.perf_counter() - start_retrieve

    stats = cache.get_memory_stats()
    stats.update(
        {
            "store_s": store_time,
            "retrieve_s": retrieve_time,
            "device": device.type,
            "forecast_last_loss": getattr(cache, "forecast_last_loss", None),
        }
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SmartKV fused attention")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--k-len", type=int, default=1024)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Request CUDA kernel when available")
    parser.add_argument("--use-mask", action="store_true", default=False,
                        help="Apply a causal-style mask during benchmarking")
    parser.add_argument("--disable-validation", action="store_true", default=False,
                        help="Skip comparing fused output against PyTorch fallback")
    parser.add_argument("--cache-budget", type=float, default=0.5)
    parser.add_argument("--enable-packing", action="store_true", default=False)
    parser.add_argument("--enable-forecast", action="store_true", default=False)
    parser.add_argument("--forecast-history", type=int, default=2048)
    parser.add_argument("--forecast-update-interval", type=int, default=32)
    parser.add_argument("--forecast-blend", type=float, default=0.5)
    parser.add_argument("--forecast-lr", type=float, default=0.05)
    parser.add_argument("--skip-cache", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False,
                        help="Emit metrics as a single JSON blob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[DIAGNOSTIC] args.use_cuda: {args.use_cuda}")
    print(f"[DIAGNOSTIC] args.device: {args.device}")
    results = {
        "attention": benchmark_attention(args),
    }
    if not args.skip_cache:
        results["cache"] = exercise_cache(args)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("SmartKV Attention Benchmark")
        for key, value in results["attention"].items():
            print(f"  {key}: {value}")
        if "cache" in results:
            print("SmartKV Cache Metrics")
            for key, value in results["cache"].items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
