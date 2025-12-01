#!/usr/bin/env python3
"""SmartKV experiment driver.

Loops over models, contexts, budgets, and baseline modes to run benchmarks
and collect JSON outputs. Designed to be executed on a GPU cluster where
the full dependencies (torch, bitsandbytes, flash-attn, datasets, evaluate)
are installed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_MODELS = [
    "NousResearch/Meta-Llama-3.1-8B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartKV experiment driver")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HuggingFace model names to evaluate",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[4096, 16000, 32000, 48000],
        help="Context lengths to benchmark",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.25, 0.33, 0.40, 0.50],
        help="Memory budgets (fraction of FP16)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["smartkv", "int8", "int4", "random", "evict"],
        help="Baseline modes to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiment_matrix",
        help="Directory to store JSON results",
    )
    parser.add_argument(
        "--bench-script",
        type=str,
        default="scripts/bench_real_context.py",
        help="Benchmark script to invoke for synthetic KV tests",
    )
    parser.add_argument(
        "--perplexity-script",
        type=str,
        default="scripts/eval_perplexity.py",
        help="Perplexity evaluation script",
    )
    parser.add_argument(
        "--long-context-script",
        type=str,
        default="scripts/eval_long_context.py",
        help="Long-context evaluation script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], dry_run: bool) -> int:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def benchmark_synthetic(
    model: str,
    contexts: List[int],
    budget: float,
    baseline: str,
    bench_script: Path,
    output_dir: Path,
    dry_run: bool,
) -> None:
    output_path = output_dir / f"{model.replace('/', '_')}_beta{budget:.2f}_{baseline}_synthetic.json"
    cmd = [
        sys.executable,
        str(bench_script),
        "--document",
        "data/romeo_juliet.txt",
        "--device",
        "cuda",
        "--memory-budget",
        f"{budget}",
        "--int8-backend",
        "bnb",
        "--fp-backend",
        "sdpa",
        "--output",
        str(output_path),
        "--contexts",
    ] + [str(ctx) for ctx in contexts]

    if baseline == "smartkv":
        pass
    elif baseline == "int8":
        cmd += ["--mode", "int8"]
    elif baseline == "int4":
        cmd += ["--mode", "int4"]
    elif baseline == "random":
        cmd += ["--mode", "random"]
    elif baseline == "evict":
        cmd += ["--mode", "evict"]
    else:
        print(f"Unknown baseline mode {baseline}; skipping synthetic benchmark")
        return

    ensure_directory(output_dir)
    ret = run_command(cmd, dry_run)
    if ret != 0:
        print(f"Synthetic benchmark failed with exit code {ret}")


def benchmark_perplexity(
    model: str,
    budget: float,
    baseline: str,
    datasets: List[str],
    perplexity_script: Path,
    output_dir: Path,
    dry_run: bool,
) -> None:
    for dataset in datasets:
        output_path = output_dir / f"{model.replace('/', '_')}_beta{budget:.2f}_{baseline}_{dataset}.json"
        cmd = [
            sys.executable,
            str(perplexity_script),
            "--model",
            model,
            "--dataset",
            dataset,
            "--budget",
            f"{budget}",
            "--baseline",
            baseline,
            "--output",
            str(output_path),
        ]
        ensure_directory(output_dir)
        ret = run_command(cmd, dry_run)
        if ret != 0:
            print(f"Perplexity benchmark failed ({dataset}) with exit code {ret}")


def benchmark_long_context(
    model: str,
    budget: float,
    baseline: str,
    suites: List[str],
    long_context_script: Path,
    output_dir: Path,
    dry_run: bool,
) -> None:
    for suite in suites:
        output_path = output_dir / f"{model.replace('/', '_')}_beta{budget:.2f}_{baseline}_{suite}.json"
        cmd = [
            sys.executable,
            str(long_context_script),
            "--model",
            model,
            "--suite",
            suite,
            "--budget",
            f"{budget}",
            "--baseline",
            baseline,
            "--output",
            str(output_path),
        ]
        ensure_directory(output_dir)
        ret = run_command(cmd, dry_run)
        if ret != 0:
            print(f"Long-context benchmark failed ({suite}) with exit code {ret}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    for model in args.models:
        print(f"\n=== Model: {model} ===")
        for budget in args.budgets:
            print(f"\n--- Budget β={budget:.2f} ---")
            for baseline in args.baselines:
                print(f"\n>>> Baseline: {baseline}")
                synthetic_dir = output_dir / "synthetic"
                benchmark_synthetic(
                    model,
                    args.contexts,
                    budget,
                    baseline,
                    Path(args.bench_script),
                    synthetic_dir,
                    args.dry_run,
                )

                perplexity_dir = output_dir / "perplexity"
                benchmark_perplexity(
                    model,
                    budget,
                    baseline,
                    ["wikitext103", "pg19"],
                    Path(args.perplexity_script),
                    perplexity_dir,
                    args.dry_run,
                )

                long_context_dir = output_dir / "long_context"
                benchmark_long_context(
                    model,
                    budget,
                    baseline,
                    ["ruler", "needle", "longbench"],
                    Path(args.long_context_script),
                    long_context_dir,
                    args.dry_run,
                )


if __name__ == "__main__":
    main()
