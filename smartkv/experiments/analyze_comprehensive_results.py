"""
Analyze and visualize comprehensive evaluation results.

Creates plots and detailed analysis of SmartKV performance across categories.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List


def load_results(results_file: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_category_comparison_plot(results: Dict[str, Any], output_dir: Path):
    """Create bar chart comparing performance across categories."""
    categories = []
    fp16_latencies = []
    smartkv_latencies = []
    overheads = []

    for category, cat_result in results['category_results'].items():
        categories.append(category.replace('_', ' ').title())
        fp16_latencies.append(cat_result['fp16_avg_latency'])
        smartkv_latencies.append(cat_result['smartkv_avg_latency'])
        overheads.append(cat_result['latency_overhead_pct'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Latency comparison
    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, fp16_latencies, width, label='FP16', alpha=0.8)
    ax1.bar(x + width/2, smartkv_latencies, width, label='SmartKV', alpha=0.8)
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Avg Latency (ms)')
    ax1.set_title('Latency Comparison by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Overhead percentages
    colors = ['green' if o < 0 else 'orange' if o < 10 else 'red' for o in overheads]
    ax2.bar(categories, overheads, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Overhead (%)')
    ax2.set_title('SmartKV Latency Overhead by Category')
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'category_comparison.png'}")


def create_quality_metrics_plot(results: Dict[str, Any], output_dir: Path):
    """Create plot showing quality metrics by category."""
    categories = []
    exact_matches = []
    word_overlaps = []
    prefix_matches = []

    for category, cat_result in results['category_results'].items():
        # Calculate category-specific quality metrics
        cat_comparisons = [c for c in results['comparisons'] if c['category'] == category]

        exact_match_rate = sum(1 for c in cat_comparisons if c['exact_match']) / len(cat_comparisons)
        avg_word_overlap = np.mean([c['word_overlap'] for c in cat_comparisons])
        avg_prefix_match = np.mean([c['prefix_match_tokens'] for c in cat_comparisons])

        categories.append(category.replace('_', ' ').title())
        exact_matches.append(exact_match_rate * 100)
        word_overlaps.append(avg_word_overlap * 100)
        prefix_matches.append(avg_prefix_match)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Exact match and word overlap
    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, exact_matches, width, label='Exact Match', alpha=0.8, color='green')
    ax1.bar(x + width/2, word_overlaps, width, label='Word Overlap', alpha=0.8, color='blue')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Output Quality Metrics by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    # Prefix match length
    ax2.bar(categories, prefix_matches, color='purple', alpha=0.7)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Avg Matching Prefix (tokens)')
    ax2.set_title('Average Prefix Match by Category')
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'quality_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'quality_metrics.png'}")


def create_detailed_report(results: Dict[str, Any], output_file: Path):
    """Create detailed text report."""
    report = []
    report.append("="*80)
    report.append("SMARTKV COMPREHENSIVE EVALUATION REPORT")
    report.append("="*80)
    report.append("")

    report.append(f"Model: {results['model']}")
    report.append(f"Memory Budget: {results['memory_budget']:.1%}")
    report.append(f"Max New Tokens: {results['max_new_tokens']}")
    report.append(f"Total Prompts Evaluated: {results['total_prompts']}")
    report.append("")

    # Overall statistics
    stats = results['overall_stats']
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"FP16 Avg Latency:        {stats['fp16_avg_latency_ms']:.2f} ms")
    report.append(f"SmartKV Avg Latency:     {stats['smartkv_avg_latency_ms']:.2f} ms")
    report.append(f"Latency Overhead:        {stats['latency_overhead_pct']:+.2f}%")
    report.append("")
    report.append(f"Exact Match Rate:        {stats['exact_match_rate']:.2%}")
    report.append(f"Avg Word Overlap:        {stats['avg_word_overlap']:.2%}")
    report.append(f"Avg Prefix Match:        {stats['avg_prefix_match_tokens']:.2f} tokens")
    report.append("")

    # Per-category breakdown
    report.append("PER-CATEGORY BREAKDOWN")
    report.append("-" * 80)

    for category, cat_result in results['category_results'].items():
        report.append(f"\n{category.upper().replace('_', ' ')}")
        report.append(f"  Prompts: {cat_result['num_prompts']}")
        report.append(f"  FP16 Latency:     {cat_result['fp16_avg_latency']:.2f} ms")
        report.append(f"  SmartKV Latency:  {cat_result['smartkv_avg_latency']:.2f} ms")
        report.append(f"  Overhead:         {cat_result['latency_overhead_pct']:+.2f}%")

        # Quality metrics for this category
        cat_comparisons = [c for c in results['comparisons'] if c['category'] == category]
        exact_matches = sum(1 for c in cat_comparisons if c['exact_match'])
        avg_overlap = np.mean([c['word_overlap'] for c in cat_comparisons])

        report.append(f"  Exact Matches:    {exact_matches}/{len(cat_comparisons)} ({exact_matches/len(cat_comparisons):.1%})")
        report.append(f"  Avg Word Overlap: {avg_overlap:.1%}")

    report.append("")

    # Detailed comparisons (first 5)
    report.append("SAMPLE OUTPUT COMPARISONS")
    report.append("-" * 80)

    for i, comp in enumerate(results['comparisons'][:5]):
        report.append(f"\n[{i+1}] Category: {comp['category']}")
        report.append(f"Prompt: {comp['prompt']}")
        report.append(f"Exact Match: {'✓' if comp['exact_match'] else '✗'}")
        report.append(f"Word Overlap: {comp['word_overlap']:.1%}")
        report.append(f"Prefix Match: {comp['prefix_match_tokens']} tokens")

    report.append("")
    report.append("="*80)

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved: {output_file}")
    return '\n'.join(report)


def analyze_results(results_file: str, output_dir: str = None):
    """Main analysis function."""
    results = load_results(results_file)

    if output_dir is None:
        output_dir = Path(results_file).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating analysis...")

    # Create plots
    try:
        create_category_comparison_plot(results, output_dir)
        create_quality_metrics_plot(results, output_dir)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
        print("Continuing with text report...")

    # Create detailed report
    report = create_detailed_report(results, output_dir / 'detailed_report.txt')

    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nFiles created in: {output_dir}")
    print("  - comprehensive_results.json")
    print("  - category_comparison.png")
    print("  - quality_metrics.png")
    print("  - detailed_report.txt")

    # Print key findings
    stats = results['overall_stats']
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n✓ Tested {results['total_prompts']} prompts across {len(results['categories_tested'])} categories")
    print(f"✓ Exact Match Rate: {stats['exact_match_rate']:.1%}")
    print(f"✓ Word Overlap: {stats['avg_word_overlap']:.1%}")
    print(f"✓ Latency Overhead: {stats['latency_overhead_pct']:+.1f}%")

    if stats['exact_match_rate'] > 0.5:
        print(f"\n🎉 EXCELLENT: >50% exact matches!")
    elif stats['word_overlap'] > 0.8:
        print(f"\n✅ GOOD: High word overlap despite differences")

    if abs(stats['latency_overhead_pct']) < 10:
        print(f"✅ GOOD: Latency overhead within ±10%")


def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive evaluation results")
    parser.add_argument(
        "--results",
        type=str,
        default="experiments/comprehensive_eval/comprehensive_results.json",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: same as results file)"
    )

    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        print("\nPlease run the evaluation first:")
        print("  python -m smartkv.experiments.comprehensive_evaluation")
        return

    analyze_results(args.results, args.output_dir)


if __name__ == "__main__":
    main()
