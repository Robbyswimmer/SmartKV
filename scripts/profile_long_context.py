#!/usr/bin/env python3
"""Profile SmartKV on a long-context document (Romeo and Juliet)."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from smartkv.core.cache import SmartKVCache


def read_document(path: Path, chunk_size: int) -> List[str]:
    text = path.read_text(encoding="utf-8")
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


def simulate_attention(num_heads: int, seq_len: int, focus_idx: int) -> torch.Tensor:
    weights = torch.zeros(1, num_heads, seq_len, seq_len)
    focus_idx = max(0, min(seq_len - 1, focus_idx))
    weights[:, :, :, focus_idx] = 1.0
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return weights


def profile_document(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    caches: Dict[str, SmartKVCache] = {
        "smartkv": SmartKVCache(
            num_layers=1,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            memory_budget=args.memory_budget,
            device=device.type,
            use_bit_packing=args.enable_packing,
            enable_forecast=args.enable_forecast,
            forecast_history=args.forecast_history,
            forecast_update_interval=args.forecast_update_interval,
            forecast_blend=args.forecast_blend,
            forecast_lr=args.forecast_lr,
        )
    }

    if not args.skip_baseline:
        caches["uniform_8bit"] = SmartKVCache(
            num_layers=1,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            memory_budget=0.5,
            device=device.type,
            use_bit_packing=False,
            enable_forecast=False,
            available_bits=[8],
        )

    doc_path = Path(args.document)
    chunks = read_document(doc_path, args.chunk_tokens)

    print(f"Processing document: {doc_path}", flush=True)
    print(f"Total chunks: {len(chunks)}", flush=True)

    total_tokens = 0

    for chunk_idx, chunk in enumerate(chunks):
        token_ids = list(range(total_tokens, total_tokens + len(chunk.split())))
        total_tokens += len(token_ids)
        if not token_ids:
            continue

        # Progress indicator every 10 chunks
        if chunk_idx % 10 == 0:
            print(f"Processing chunk {chunk_idx}/{len(chunks)} ({total_tokens} tokens)...", flush=True)

        if not token_ids:
            continue

        attn = simulate_attention(args.num_heads, len(token_ids), focus_idx=len(token_ids) // 2)
        k_batch = torch.randn(len(token_ids), args.num_heads, args.head_dim, device=device)
        v_batch = torch.randn(len(token_ids), args.num_heads, args.head_dim, device=device)

        for name, cache in caches.items():
            if name == "smartkv":
                cache.update_attention(0, attn, token_ids)
                cache.allocate_precision(0, token_ids)
            else:
                max_bits = max(cache.available_bits)
                for tid in token_ids:
                    cache.precision_map[tid] = max_bits
            cache.quantize_and_store_batch(
                layer_idx=0,
                token_ids=token_ids,
                k_batch=k_batch.clone() if name != "smartkv" else k_batch,
                v_batch=v_batch.clone() if name != "smartkv" else v_batch,
            )

    print(f"Finished processing {len(chunks)} chunks, {total_tokens} total tokens", flush=True)

    results: Dict[str, Dict[str, float]] = {}
    for name, cache in caches.items():
        stats = cache.get_memory_stats()
        results[name] = {
            "document_tokens": total_tokens,
            "memory_ratio": stats["memory_ratio"],
            "memory_ratio_true": stats.get("memory_ratio_true"),
            "avg_bits": stats["avg_bits"],
            "precision_distribution": stats.get("precision_distribution", {}),
            "storage_mode": stats.get("storage_mode", "unknown"),
            "forecast_last_loss": cache.forecast_last_loss,
            "num_realloc": cache.realloc_counter,
            "num_tokens_cached": stats["num_tokens"],
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SmartKV on a long document")
    parser.add_argument("--document", default="data/romeo_juliet.txt")
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--memory-budget", type=float, default=0.5)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-packing", action="store_true", default=False,
                        help="Enable bit-packing for sub-byte storage")
    parser.add_argument("--enable-forecast", action="store_true", default=False)
    parser.add_argument("--forecast-history", type=int, default=4096)
    parser.add_argument("--forecast-update-interval", type=int, default=32)
    parser.add_argument("--forecast-blend", type=float, default=0.5)
    parser.add_argument("--forecast-lr", type=float, default=0.05)
    parser.add_argument("--skip-baseline", action="store_true", default=False,
                        help="Skip uniform 8-bit baseline run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = profile_document(args)

    # Pretty print results
    for label, metrics in results.items():
        print(f"\nLong-Context Document Profile Results ({label})", flush=True)
        print("=" * 50, flush=True)
        print(f"Document tokens: {metrics['document_tokens']}", flush=True)
        print(f"Tokens cached: {metrics['num_tokens_cached']}", flush=True)
        if label == "smartkv":
            print(f"Memory budget: {args.memory_budget:.2f}", flush=True)
        else:
            print("Memory budget: 0.50 (uniform 8-bit)", flush=True)
        print(f"Memory ratio (true): {metrics.get('memory_ratio_true', 0):.4f}", flush=True)
        print(f"Average bits: {metrics['avg_bits']:.2f}", flush=True)
        print(f"Storage mode: {metrics.get('storage_mode', 'unknown')}", flush=True)
        print(f"Precision distribution: {metrics.get('precision_distribution', {})}", flush=True)
        print(f"Reallocations: {metrics['num_realloc']}", flush=True)
        if metrics.get('forecast_last_loss') is not None:
            print(f"Forecast loss: {metrics['forecast_last_loss']:.4f}", flush=True)
        print("=" * 50, flush=True)

    # Also output JSON for scripting
    print("\nJSON Output:", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
