#!/usr/bin/env python3
"""
Benchmark SmartKV vs INT8 vs FP16 on Romeo & Juliet token sequences.

Reads the public domain text, creates deterministic pseudo-random KV tensors
based on token order, and measures kernel latency across increasing context
lengths. FP16 is capped at a configurable maximum to avoid OOM, INT8 is tested
up to its cap, and SmartKV can go to the longest context supported by the
benchmark. Results are printed and stored as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

import torch

from smartkv.core.cache import SmartKVCache
from smartkv.kernels import (
    quantized_attention,
    quantized_attention_bucketed,
)

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FLASH_ATTENTION_AVAILABLE = False

try:
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_fn
    BNB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    bnb = None
    bnb_fn = None
    BNB_AVAILABLE = False


KEY_TOKENS = {"romeo", "juliet", "tybalt", "mercutio", "verona", "capulet", "montague"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SmartKV using Romeo & Juliet text")
    parser.add_argument("--document", default="data/romeo_juliet.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--memory-budget", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32000, 48000],
        help="Context lengths to benchmark",
    )
    parser.add_argument("--fp-max", type=int, default=16384)
    parser.add_argument("--int8-max", type=int, default=32000)
    parser.add_argument("--smartkv-max", type=int, default=48000)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--fp-backend",
        choices=["sdpa", "flash"],
        default="sdpa",
        help="Floating-point baseline backend",
    )
    parser.add_argument(
        "--int8-backend",
        choices=["legacy", "bnb"],
        default="legacy",
        help="INT8 baseline backend",
    )
    return parser.parse_args()


def load_tokens(path: Path, max_tokens: int) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    tokens = raw.split()
    if not tokens:
        raise RuntimeError(f"Document {path} produced zero tokens")
    if len(tokens) >= max_tokens:
        return tokens[:max_tokens]
    repeats = (max_tokens + len(tokens) - 1) // len(tokens)
    extended = (tokens * repeats)[:max_tokens]
    return extended


def token_importance(tokens: Sequence[str]) -> List[float]:
    scores: List[float] = []
    for idx, tok in enumerate(tokens):
        score = 1.0
        lower = tok.lower().strip(".,;:!?\"()[]{}")
        if lower in KEY_TOKENS:
            score += 4.0
        if lower.istitle():
            score += 0.5
        # Mild positional decay for earlier tokens so late context can gain priority
        score *= 1.0 + 0.2 * (idx / max(1, len(tokens) - 1))
        scores.append(score)
    return scores


def make_random_kv(max_tokens: int, num_heads: int, head_dim: int, device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    data = torch.randn((max_tokens, num_heads, head_dim), generator=generator, device=device)
    return {
        "k": data.clone(),
        "v": torch.randn((max_tokens, num_heads, head_dim), generator=generator, device=device),
    }


def format_latency(times: List[float]) -> Dict[str, float]:
    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[int(len(times_sorted) * 0.95)]
    return {
        "avg_ms": mean(times) * 1000,
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
    }


def run_smartkv(
    ctx_len: int,
    cache: SmartKVCache,
    data: Dict[str, torch.Tensor],
    chunk_size: int,
    importance_scores: Sequence[float],
    device: torch.device,
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    cache.reset()
    if hasattr(cache, "_rank_state"):
        cache._rank_state.clear()
    cache.global_step = 0

    for start in range(0, ctx_len, chunk_size):
        size = min(chunk_size, ctx_len - start)
        token_ids = list(range(start, start + size))

        # Inject heuristic importance without building dense attention tensors
        for tid in token_ids:
            cache.importance_tracker.token_importance[tid] = importance_scores[tid]
            cache.last_seen[tid] = cache.global_step
        cache.global_step = token_ids[-1] + 1

        cache.allocate_precision(0, token_ids)
        cache.quantize_and_store_batch(0, token_ids, data["k"][start:start + size], data["v"][start:start + size])

    buckets = cache.get_bucket_views(0)
    query = torch.randn((1, cache.num_heads, 1, cache.head_dim), device=device)

    def run_once() -> None:
        quantized_attention_bucketed(query, buckets, use_cuda=(device.type == "cuda"))

    # Warmup
    for _ in range(warmup):
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        start = perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(perf_counter() - start)

    stats = format_latency(times)
    mem_stats = cache.get_memory_stats()
    bucket_dist = {int(bits): int(view["token_ids"].numel()) for bits, view in buckets.items()}
    stats.update(
        avg_bits=mem_stats.get("avg_bits", 0.0),
        memory_ratio=mem_stats.get("memory_ratio_true", mem_stats.get("memory_ratio", 0.0)),
        bucket_dist=bucket_dist,
    )
    return stats


def run_int8_legacy(
    ctx_len: int,
    cache: SmartKVCache,
    data: Dict[str, torch.Tensor],
    chunk_size: int,
    device: torch.device,
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    cache.reset()
    if hasattr(cache, "_rank_state"):
        cache._rank_state.clear()

    for start in range(0, ctx_len, chunk_size):
        size = min(chunk_size, ctx_len - start)
        token_ids = list(range(start, start + size))
        for tid in token_ids:
            cache.precision_map[tid] = 8
        cache.quantize_and_store_batch(0, token_ids, data["k"][start:start + size], data["v"][start:start + size])

    tensors = cache.retrieve_all_quantized(0)
    query = torch.randn((1, cache.num_heads, 1, cache.head_dim), device=device)

    def run_once() -> None:
        quantized_attention(
            query,
            tensors["k_qx"].unsqueeze(0),
            tensors["k_scale"].unsqueeze(0),
            tensors["v_qx"].unsqueeze(0),
            tensors["v_scale"].unsqueeze(0),
            use_cuda=(device.type == "cuda"),
        )

    for _ in range(warmup):
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        start = perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(perf_counter() - start)

    stats = format_latency(times)
    mem_stats = cache.get_memory_stats()
    stats.update(
        avg_bits=mem_stats.get("avg_bits", 0.0),
        memory_ratio=mem_stats.get("memory_ratio_true", mem_stats.get("memory_ratio", 0.0)),
        bucket_dist={8: ctx_len},
    )
    return stats


def _quantize_heads_bitsandbytes(
    tensor: torch.Tensor,
    transpose: bool = False,
    blocksize: int = 64,
) -> List[Tuple[torch.Tensor, Any]]:
    """Quantize per head using bitsandbytes blockwise quantization."""

    heads_first = tensor.transpose(0, 1)  # [num_heads, seq_len, head_dim]
    quantized: List[Tuple[torch.Tensor, Any]] = []
    for head in heads_first:
        if transpose:
            head_matrix = head.transpose(0, 1).contiguous()
        else:
            head_matrix = head.contiguous()
        q, state = bnb_fn.quantize_blockwise(head_matrix, blocksize=blocksize)
        quantized.append((q, state))
    return quantized


def run_int8_bitsandbytes(
    ctx_len: int,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    if not BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes backend requested but bitsandbytes is not installed")
    if device.type != "cuda":
        raise RuntimeError("bitsandbytes INT8 backend requires a CUDA device")

    dtype = torch.float16
    key_full = data["k"][:ctx_len].to(device=device, dtype=dtype)
    value_full = data["v"][:ctx_len].to(device=device, dtype=dtype)

    num_heads = key_full.size(1)
    head_dim = key_full.size(2)

    # Quantize keys (transpose so matmul expects [head_dim, seq_len]) and values ([seq_len, head_dim])
    quant_keys = _quantize_heads_bitsandbytes(key_full, transpose=True)
    quant_values = _quantize_heads_bitsandbytes(value_full, transpose=False)

    query = torch.randn((num_heads, head_dim), device=device, dtype=dtype)

    inv_sqrt_d = 1.0 / math.sqrt(head_dim)

    def run_once() -> None:
        outputs = []
        for h in range(num_heads):
            q_head = query[h : h + 1]  # [1, head_dim]
            qk, state_k = quant_keys[h]
            # qk shape [head_dim, seq_len]
            scores = bnb.matmul(q_head, qk, state_k)  # [1, seq_len]
            scores = (scores.float() * inv_sqrt_d)
            weights = torch.softmax(scores, dim=-1)
            qv, state_v = quant_values[h]
            # qv shape [seq_len, head_dim]
            head_out = bnb.matmul(weights.to(dtype), qv, state_v)
            outputs.append(head_out)
        _ = torch.cat(outputs, dim=0)

    for _ in range(warmup):
        run_once()
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iters):
        start = perf_counter()
        run_once()
        torch.cuda.synchronize()
        times.append(perf_counter() - start)

    stats = format_latency(times)

    # Estimate memory ratio: quantized values (8 bits) + scale per block
    total_elements = ctx_len * num_heads * head_dim
    payload_bits = 0
    scale_bits = 0
    for q, state in quant_keys:
        payload_bits += q.numel() * 8
        absmax = getattr(state, "absmax", None)
        if absmax is not None:
            scale_bits += absmax.numel() * 32
    for q, state in quant_values:
        payload_bits += q.numel() * 8
        absmax = getattr(state, "absmax", None)
        if absmax is not None:
            scale_bits += absmax.numel() * 32
    denom_bits = total_elements * 16 if total_elements > 0 else 1
    memory_ratio = (payload_bits + scale_bits) / denom_bits

    stats.update(avg_bits=8.0, memory_ratio=memory_ratio, bucket_dist={8: ctx_len})
    return stats


def run_fp_sdpa(
    ctx_len: int,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    k = data["k"][:ctx_len]
    v = data["v"][:ctx_len]
    batch = 1
    num_heads = k.size(1)
    head_dim = k.size(2)
    q = torch.randn((batch, 1, num_heads, head_dim), device=device, dtype=k.dtype)
    q_sdpa = q.permute(0, 2, 1, 3)
    k_sdpa = k.transpose(0, 1).unsqueeze(0)
    v_sdpa = v.transpose(0, 1).unsqueeze(0)

    def run_once() -> None:
        torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=None, dropout_p=0.0, is_causal=True
        )

    for _ in range(warmup):
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        start = perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(perf_counter() - start)

    stats = format_latency(times)
    stats.update(avg_bits=16.0, memory_ratio=1.0, bucket_dist={16: ctx_len})
    return stats


def run_fp_flash(
    ctx_len: int,
    data: Dict[str, torch.Tensor],
    device: torch.device,
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    if not FLASH_ATTENTION_AVAILABLE:
        raise RuntimeError("flash-attn not installed; install flash-attn to use --fp-backend flash")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    k = data["k"][:ctx_len].to(dtype).contiguous()
    v = data["v"][:ctx_len].to(dtype).contiguous()
    batch = 1
    num_heads = k.size(1)
    head_dim = k.size(2)
    q = torch.randn((batch, ctx_len, num_heads, head_dim), device=device, dtype=dtype).contiguous()
    k_flash = k.unsqueeze(0).contiguous()
    v_flash = v.unsqueeze(0).contiguous()

    def run_once() -> None:
        flash_attn_func(q, k_flash, v_flash, dropout_p=0.0, causal=True)

    for _ in range(warmup):
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        start = perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(perf_counter() - start)

    stats = format_latency(times)
    stats.update(avg_bits=16.0, memory_ratio=1.0, bucket_dist={16: ctx_len})
    return stats


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    max_context = min(max(args.contexts), args.smartkv_max)
    tokens = load_tokens(Path(args.document), max_context)
    importance_scores = token_importance(tokens)

    kv_data = make_random_kv(max_context, args.num_heads, args.head_dim, device, args.seed)

    smartkv_cache = SmartKVCache(
        num_layers=1,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        memory_budget=args.memory_budget,
        available_bits=[2, 4, 8],
        device=device.type,
        use_bucketed_layout=True,
        use_bit_packing=True,
        utility_alpha=0.75,
        high_precision_fraction=0.05,
        high_precision_boost=3.0,
        min_high_precision_tokens=128,
    )

    int8_cache = SmartKVCache(
        num_layers=1,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        memory_budget=0.5,
        available_bits=[8],
        device=device.type,
        use_bucketed_layout=False,
        use_bit_packing=False,
    )

    contexts_sorted = sorted(args.contexts)
    fp_runner = {
        "sdpa": run_fp_sdpa,
        "flash": run_fp_flash,
    }[args.fp_backend]
    fp_key = f"fp_{args.fp_backend}"
    int8_runner_map = {
        "legacy": lambda ctx_len: run_int8_legacy(
            ctx_len,
            int8_cache,
            kv_data,
            args.chunk_size,
            device,
            args.iters,
            args.warmup,
        ),
        "bnb": lambda ctx_len: run_int8_bitsandbytes(
            ctx_len,
            kv_data,
            device,
            args.iters,
            args.warmup,
        ),
    }

    int8_runner = int8_runner_map[args.int8_backend]
    int8_key = f"int8_{args.int8_backend}"

    results: Dict[str, Dict[int, Dict[str, float]]] = {
        "smartkv": {},
        int8_key: {},
        fp_key: {},
    }

    print("\n=== SmartKV / INT8 / FP Benchmark (Romeo & Juliet) ===")
    print(f"Device: {device}")
    print(f"Contexts: {contexts_sorted}")
    print(f"FP backend: {args.fp_backend}")
    print(f"INT8 backend: {args.int8_backend}")
    print(f"FP max: {args.fp_max}, INT8 max: {args.int8_max}, SmartKV max: {args.smartkv_max}")

    for ctx_len in contexts_sorted:
        if ctx_len > args.smartkv_max:
            continue
        print(f"\nContext length: {ctx_len}")
        print("-" * 80)

        skv_stats = run_smartkv(
            ctx_len,
            smartkv_cache,
            kv_data,
            args.chunk_size,
            importance_scores,
            device,
            args.iters,
            args.warmup,
        )
        results["smartkv"][ctx_len] = skv_stats
        print(
            f"[SmartKV] avg={skv_stats['avg_ms']:.3f} ms | p50={skv_stats['p50_ms']:.3f} ms | "
            f"p95={skv_stats['p95_ms']:.3f} ms | avg_bits={skv_stats['avg_bits']:.2f} | "
            f"mem={skv_stats['memory_ratio']:.3f}"
        )
        print(f"           buckets: {skv_stats['bucket_dist']}")

        if ctx_len <= args.int8_max:
            try:
                int8_stats = int8_runner(ctx_len)
                results[int8_key][ctx_len] = int8_stats
                print(
                    f"[INT8 {args.int8_backend}] avg={int8_stats['avg_ms']:.3f} ms | "
                    f"p50={int8_stats['p50_ms']:.3f} ms | p95={int8_stats['p95_ms']:.3f} ms | "
                    f"mem={int8_stats['memory_ratio']:.3f}"
                )
            except RuntimeError as err:
                print(f"[INT8 {args.int8_backend}] failed: {err}")
        else:
            print(f"[INT8 {args.int8_backend}] skipped (context above cap)")

        if ctx_len <= args.fp_max:
            fp_stats = fp_runner(
                ctx_len,
                kv_data,
                device,
                args.iters,
                args.warmup,
            )
            results[fp_key][ctx_len] = fp_stats
            print(
                f"[FP {args.fp_backend}]   avg={fp_stats['avg_ms']:.3f} ms | "
                f"p50={fp_stats['p50_ms']:.3f} ms | p95={fp_stats['p95_ms']:.3f} ms"
            )
        else:
            print("[FP]     skipped (context above cap)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {out_path}")
    else:
        print("\nJSON Results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
