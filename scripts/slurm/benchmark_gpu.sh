#!/bin/bash -l

#SBATCH --job-name="SmartKV-Benchmark"
#SBATCH --output=logs/benchmark_%j.txt
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

set -euo pipefail

# Activate conda environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-smartkv-gpu}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

echo "Starting SmartKV performance benchmark at $(date)"

# Create logs directory
mkdir -p logs

# Configuration - Baselines to compare
BASELINES=${BASELINES:-"fp16 uniform_8bit uniform_4bit smartkv_mixed"}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"512,1024,2048,4096,8192"}
DEVICE=${DEVICE:-cuda:0}

# Model configuration
NUM_LAYERS=${NUM_LAYERS:-32}
NUM_HEADS=${NUM_HEADS:-32}
HEAD_DIM=${HEAD_DIM:-128}

# Benchmark settings
NUM_WARMUP=${NUM_WARMUP:-10}
NUM_ITERATIONS=${NUM_ITERATIONS:-100}

# Output
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/results/benchmarks/${SLURM_JOB_ID}"}
mkdir -p "$OUTPUT_ROOT"

echo "Benchmark Configuration:"
echo "  Baselines: $BASELINES"
echo "  Context lengths: $CONTEXT_LENGTHS"
echo "  Device: $DEVICE"
echo "  Model: ${NUM_LAYERS}L, ${NUM_HEADS}H, ${HEAD_DIM}D"
echo "  Warmup: $NUM_WARMUP, Iterations: $NUM_ITERATIONS"
echo "  Output: $OUTPUT_ROOT"
echo ""

# Check CUDA availability
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

baselines=($BASELINES)
total_baselines=${#baselines[@]}

echo ""
echo "Total baselines to benchmark: $total_baselines"
echo ""

# Benchmark each baseline
baseline_num=0
for baseline in "${baselines[@]}"; do
  baseline_num=$((baseline_num + 1))

  echo ""
  echo "========================================"
  echo "Baseline $baseline_num / $total_baselines: $baseline"
  echo "========================================"

  output_file="$OUTPUT_ROOT/benchmark_${baseline}.csv"

  # Run benchmark
  python scripts/run_gpu_benchmark.py \
    --baseline "$baseline" \
    --context-lengths "$CONTEXT_LENGTHS" \
    --device "$DEVICE" \
    --num-layers "$NUM_LAYERS" \
    --num-heads "$NUM_HEADS" \
    --head-dim "$HEAD_DIM" \
    --num-warmup "$NUM_WARMUP" \
    --num-iterations "$NUM_ITERATIONS" \
    --output "$output_file" \
    --verbose

  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Baseline $baseline completed successfully"

    # Print summary
    echo ""
    echo "Summary for $baseline:"
    python -c "
import csv
with open('$output_file') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ctx = row['context_length']
        lat = float(row['avg_latency_ms'])
        tput = float(row['throughput_tps'])
        mem = float(row['memory_gb'])
        print(f\"  Context {ctx:>5}: {lat:>7.2f} ms, {tput:>7.0f} tok/s, {mem:>6.3f} GB\")
"
  else
    echo "❌ Baseline $baseline failed (exit code: $EXIT_CODE)"
  fi

  echo ""
done

echo ""
echo "========================================"
echo "Benchmark completed at $(date)"
echo "========================================"
echo ""

# Combine all results into single comparison CSV
combined_csv="$OUTPUT_ROOT/benchmark_comparison.csv"
echo "Creating combined comparison CSV..."

python -c "
import csv
from pathlib import Path

results_dir = Path('$OUTPUT_ROOT')
baselines = '$BASELINES'.split()

# Read all baseline results
all_rows = []
for baseline in baselines:
    csv_file = results_dir / f'benchmark_{baseline}.csv'
    if csv_file.exists():
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            all_rows.extend(list(reader))

# Write combined results
if all_rows:
    with open('$combined_csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'Combined results saved to: $combined_csv')
    print(f'Total rows: {len(all_rows)}')

    # Print comparison table
    print('')
    print('Performance Comparison:')
    print('-' * 80)

    # Group by context length
    from collections import defaultdict
    by_context = defaultdict(list)
    for row in all_rows:
        by_context[row['context_length']].append(row)

    for ctx_len in sorted(by_context.keys(), key=int):
        print(f'Context Length: {ctx_len}')
        print(f'  {\"Baseline\":<20} {\"Latency (ms)\":>15} {\"Throughput\":>15} {\"Memory (GB)\":>15}')
        for row in by_context[ctx_len]:
            baseline = row['baseline']
            lat = float(row['avg_latency_ms'])
            tput = float(row['throughput_tps'])
            mem = float(row['memory_gb'])
            print(f'  {baseline:<20} {lat:>15.2f} {tput:>15.0f} {mem:>15.3f}')
        print('')
"

echo ""
echo "Benchmark suite complete!"
echo "Individual results: $OUTPUT_ROOT/benchmark_*.csv"
echo "Combined results: $combined_csv"
