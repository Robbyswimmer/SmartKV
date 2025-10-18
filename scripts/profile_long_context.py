#!/usr/bin/env python3
"""Profile SmartKV on a long-context document (Romeo and Juliet)."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from smartkv.core.cache import SmartKVCache
from smartkv.core.importance import ImportanceTracker


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


def profile_document(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cache = SmartKVCache(
        num_layers=1,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        memory_budget=args.memory_budget,
        device=device.type,
        enable_forecast=args.enable_forecast,
        forecast_history=args.forecast_history,
        forecast_update_interval=args.forecast_update_interval,
        forecast_blend=args.forecast_blend,
        forecast_lr=args.forecast_lr,
    )

    doc_path = Path(args.document)
    chunks = read_document(doc_path, args.chunk_tokens)

    tracker = ImportanceTracker(num_layers=1, decay=cache.decay, device='cpu')
    total_tokens = 0

    for chunk_idx, chunk in enumerate(chunks):
        token_ids = list(range(total_tokens, total_tokens + len(chunk.split())))
        total_tokens += len(token_ids)
        if not token_ids:
            continue

        attn = simulate_attention(args.num_heads, len(token_ids), focus_idx=len(token_ids) // 2)
        cache.update_attention(0, attn, token_ids)
        cache.allocate_precision(0, token_ids)
        cache.quantize_and_store_batch(
            layer_idx=0,
            token_ids=token_ids,
            k_batch=torch.randn(len(token_ids), args.num_heads, args.head_dim, device=device),
            v_batch=torch.randn(len(token_ids), args.num_heads, args.head_dim, device=device),
        )

    stats = cache.get_memory_stats()
    results = {
        "document_tokens": total_tokens,
        "memory_ratio": stats["memory_ratio"],
        "memory_ratio_true": stats.get("memory_ratio_true"),
        "avg_bits": stats["avg_bits"],
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
    parser.add_argument("--enable-forecast", action="store_true", default=False)
    parser.add_argument("--forecast-history", type=int, default=4096)
    parser.add_argument("--forecast-update-interval", type=int, default=32)
    parser.add_argument("--forecast-blend", type=float, default=0.5)
    parser.add_argument("--forecast-lr", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = profile_document(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
