"""
Perplexity evaluation on WikiText for SmartKV vs baselines (CPU-friendly).

Runs:
- Baseline model perplexity
- SmartKV at multiple memory budgets

Outputs:
- JSON + CSV with perplexity, loss, tokens processed, memory stats (for SmartKV), and timing.

Example:
    python -m smartkv.experiments.run_wikitext_perplexity \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --dataset wikitext --subset wikitext-2-raw-v1 --split test \\
        --budgets 0.35 0.5 0.7 --block-size 512 --max-eval-tokens 32768 \\
        --device cpu --output-dir results/wikitext2_perplexity
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig
from smartkv.utils.logger import create_logger


@dataclass
class RunResult:
    name: str
    perplexity: float
    loss: float
    tokens: int
    wall_clock_s: float
    tokens_per_sec: float
    memory_ratio: Optional[float] = None
    memory_ratio_true: Optional[float] = None
    avg_bits: Optional[float] = None
    precision_distribution: Optional[Dict[str, int]] = None
    memory_budget: Optional[float] = None


def load_wikitext_tokens(
    dataset: str,
    subset: str,
    split: str,
    tokenizer,
    block_size: int,
    max_eval_tokens: Optional[int] = None,
    logger=None,
) -> torch.Tensor:
    """Load WikiText split and return token blocks [num_blocks, block_size]."""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required for WikiText perplexity.")

    ds = load_dataset(dataset, subset, split=split)
    text = "\n".join(ds["text"])
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    if max_eval_tokens is not None:
        encoded = encoded[:max_eval_tokens]

    total_len = (encoded.numel() // block_size) * block_size
    encoded = encoded[:total_len]
    if logger:
        logger.info(
            f"Tokenized {encoded.numel()} tokens -> {encoded.numel() // block_size} blocks of {block_size}"
        )
    blocks = encoded.view(-1, block_size)
    return blocks


def evaluate_perplexity_blocks(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    device: torch.device,
    reset_cache_fn=None,
    desc: str = "",
    stepwise: bool = False,
) -> Tuple[float, float, int, float]:
    """
    Evaluate perplexity over pre-tokenized blocks using autoregressive stepping
    (q_len = 1) to exercise fused quantized attention paths.

    Returns:
        perplexity, avg_loss, total_tokens, wall_clock_s
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()

    for block in blocks:
        input_ids = block.to(device)
        seq_len = input_ids.size(0)
        if reset_cache_fn is not None:
            reset_cache_fn()

        if stepwise:
            # Teacher-forced token-by-token loop (q_len=1) to engage fused quantized path
            for t in range(1, seq_len):
                ctx = input_ids[:t].unsqueeze(0)  # [1, t]
                target = input_ids[: t + 1].unsqueeze(0)  # shift by one
                with torch.no_grad():
                    outputs = model(ctx, labels=target)
                    loss = outputs.loss

                tokens = 1
                total_loss += loss.item() * tokens
                total_tokens += tokens
        else:
            # Batched block evaluation (may bypass fused path when q_len>1)
            input_ids_batched = input_ids.unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids_batched, labels=input_ids_batched)
                loss = outputs.loss
            tokens = seq_len - 1
            total_loss += loss.item() * tokens
            total_tokens += tokens

    wall_clock_s = time.perf_counter() - start
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss, total_tokens, wall_clock_s


def run_smartkv_perplexity(
    base_model,
    tokenizer,
    blocks: torch.Tensor,
    device: torch.device,
    budgets: List[float],
    args: argparse.Namespace,
    logger,
    stepwise: bool,
) -> List[RunResult]:
    """Evaluate SmartKV at multiple budgets using a single wrapped model."""
    results: List[RunResult] = []

    config = SmartKVConfig(
        enabled=True,
        memory_budget=budgets[0],
        decay=args.decay,
        realloc_freq=args.realloc_freq,
        available_bits=args.available_bits,
        device=str(device),
    )
    smartkv_model = LlamaSmartKV(base_model, config)
    smartkv_model.set_use_fused_gpu(False)
    smartkv_model.set_use_fused_cpu(True)

    def reset_cache():
        if hasattr(smartkv_model, "reset_cache"):
            smartkv_model.reset_cache()

    for budget in budgets:
        logger.info(f"Running SmartKV budget={budget}")
        smartkv_model.smartkv_config.memory_budget = budget
        if smartkv_model.smartkv_cache is not None:
            smartkv_model.smartkv_cache.memory_budget = budget
        reset_cache()

        ppl, avg_loss, tokens, wall = evaluate_perplexity_blocks(
            smartkv_model,
            blocks,
            device,
            reset_cache_fn=reset_cache,
            desc=f"smartkv_{budget}",
            stepwise=stepwise,
        )

        memory_ratio = None
        memory_ratio_true = None
        avg_bits = None
        precision_distribution = None
        mem_budget_effective = None
        if smartkv_model.smartkv_cache is not None:
            stats = smartkv_model.smartkv_cache.get_memory_stats()
            memory_ratio = stats.get("memory_ratio_true", stats.get("memory_ratio"))
            memory_ratio_true = stats.get("memory_ratio_true")
            avg_bits = stats.get("avg_bits")
            precision_distribution = stats.get("precision_distribution")
            mem_budget_effective = stats.get("memory_budget", getattr(smartkv_model.smartkv_cache, "memory_budget", None))

        tps = tokens / wall if wall > 0 else 0.0
        results.append(
            RunResult(
                name=f"smartkv_budget_{budget}",
                perplexity=ppl,
                loss=avg_loss,
                tokens=tokens,
                wall_clock_s=wall,
                tokens_per_sec=tps,
                memory_ratio=memory_ratio,
                memory_ratio_true=memory_ratio_true,
                avg_bits=avg_bits,
                precision_distribution=precision_distribution,
                memory_budget=mem_budget_effective,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="WikiText perplexity for SmartKV.")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.35, 0.5, 0.7])
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-eval-tokens", type=int, default=32768)
    parser.add_argument("--available-bits", type=int, nargs="+", default=[2, 3, 4, 8])
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--realloc-freq", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/wikitext_perplexity")
    parser.add_argument(
        "--stepwise",
        action="store_true",
        help="Evaluate token-by-token (q_len=1) to engage fused quantized path; slower but validates quantized attention.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("wikitext_perplexity", log_file=str(output_dir / "run.log"))

    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers is required for this script.")
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets is required for this script.")

    device = torch.device(args.device)
    logger.info(f"Loading tokenizer/model: {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    ).to(device)
    base_model.eval()

    logger.info(
        f"Loading dataset {args.dataset}/{args.subset} split={args.split} "
        f"block_size={args.block_size} max_eval_tokens={args.max_eval_tokens}"
    )
    blocks = load_wikitext_tokens(
        args.dataset,
        args.subset,
        args.split,
        tokenizer,
        block_size=args.block_size,
        max_eval_tokens=args.max_eval_tokens,
        logger=logger,
    )

    results: List[RunResult] = []

    # Baseline
    logger.info("Running baseline perplexity")

    def reset_noop():
        return None

    ppl, avg_loss, tokens, wall = evaluate_perplexity_blocks(
        base_model,
        blocks,
        device,
        reset_cache_fn=reset_noop,
        desc="baseline",
        stepwise=args.stepwise,
    )
    tps = tokens / wall if wall > 0 else 0.0
    results.append(
        RunResult(
            name="baseline_fp32",
            perplexity=ppl,
            loss=avg_loss,
            tokens=tokens,
            wall_clock_s=wall,
            tokens_per_sec=tps,
        )
    )

    # SmartKV runs
    results.extend(
        run_smartkv_perplexity(
            base_model,
            tokenizer,
            blocks,
            device,
            budgets=args.budgets,
            args=args,
            logger=logger,
            stepwise=args.stepwise,
        )
    )

    # Save results
    json_path = output_dir / "perplexity_results.json"
    csv_path = output_dir / "perplexity_results.csv"

    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    with open(csv_path, "w") as f:
        headers = [
            "name",
            "perplexity",
            "loss",
            "tokens",
            "wall_clock_s",
            "tokens_per_sec",
            "memory_ratio",
            "memory_ratio_true",
            "avg_bits",
            "memory_budget",
        ]
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                r.name,
                f"{r.perplexity:.6f}",
                f"{r.loss:.6f}",
                str(r.tokens),
                f"{r.wall_clock_s:.4f}",
                f"{r.tokens_per_sec:.2f}",
                "" if r.memory_ratio is None else f"{r.memory_ratio:.4f}",
                "" if r.memory_ratio_true is None else f"{r.memory_ratio_true:.4f}",
                "" if r.avg_bits is None else f"{r.avg_bits:.3f}",
                "" if r.memory_budget is None else f"{r.memory_budget:.3f}",
            ]
            f.write(",".join(row) + "\n")

    logger.info(f"Saved results to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
