#!/bin/bash -l

#SBATCH --job-name="SmartKV-Ablation"
#SBATCH --output=logs/ablation_%j.txt
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
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

echo "Starting SmartKV memory budget ablation study at $(date)"

# Create logs directory
mkdir -p logs

# Configuration - Ablation sweep parameters
MEMORY_BUDGETS=${MEMORY_BUDGETS:-"0.2 0.3 0.4 0.5 0.6 0.7"}
BIT_CONFIGS=${BIT_CONFIGS:-"2,3,4,8 4,8 2,4,8"}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"512 1024 2048 4096 8192"}
USE_BIT_PACKING_MODES=${USE_BIT_PACKING_MODES:-"true false"}
DEVICE=${DEVICE:-cuda:0}

# Model configuration
NUM_LAYERS=${NUM_LAYERS:-32}
NUM_HEADS=${NUM_HEADS:-32}
HEAD_DIM=${HEAD_DIM:-128}

# Sampling
NUM_SAMPLES=${NUM_SAMPLES:-10}

# Output
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/results/ablations/${SLURM_JOB_ID}"}
mkdir -p "$OUTPUT_ROOT"

echo "Ablation Configuration:"
echo "  Memory budgets: $MEMORY_BUDGETS"
echo "  Bit configs: $BIT_CONFIGS"
echo "  Context lengths: $CONTEXT_LENGTHS"
echo "  Packing modes: $USE_BIT_PACKING_MODES"
echo "  Device: $DEVICE"
echo "  Model: ${NUM_LAYERS}L, ${NUM_HEADS}H, ${HEAD_DIM}D"
echo "  Output: $OUTPUT_ROOT"
echo ""

# Check CUDA availability
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Count total experiments
budgets=($MEMORY_BUDGETS)
bits=($BIT_CONFIGS)
contexts=($CONTEXT_LENGTHS)
packings=($USE_BIT_PACKING_MODES)

total_experiments=$((${#budgets[@]} * ${#bits[@]} * ${#contexts[@]} * ${#packings[@]}))
echo ""
echo "Total experiments to run: $total_experiments"
echo ""

# Progress counter
experiment_num=0

# Create summary CSV header
summary_csv="$OUTPUT_ROOT/ablation_summary.csv"
echo "experiment,budget,bits,context_length,use_packing,memory_ratio,avg_bits,allocated_gb,throughput_tps,avg_latency_ms" > "$summary_csv"

# Sweep over all configurations
for budget in "${budgets[@]}"; do
  for bit_config in "${bits[@]}"; do
    for context_len in "${contexts[@]}"; do
      for use_packing in "${packings[@]}"; do
        experiment_num=$((experiment_num + 1))

        # Format experiment name
        packing_str=$([ "$use_packing" = "true" ] && echo "packed" || echo "unpacked")
        bits_str=$(echo "$bit_config" | tr ',' '_')
        exp_name="b${budget}_bits${bits_str}_ctx${context_len}_${packing_str}"

        echo ""
        echo "========================================"
        echo "Experiment $experiment_num / $total_experiments"
        echo "  Budget: $budget"
        echo "  Bits: $bit_config"
        echo "  Context: $context_len"
        echo "  Packing: $use_packing"
        echo "========================================"

        # Build command
        packing_flag=$([ "$use_packing" = "true" ] && echo "--use-bit-packing" || echo "")

        output_file="$OUTPUT_ROOT/${exp_name}.json"

        # Run ablation
        python scripts/run_ablation_study.py \
          --budget "$budget" \
          --bits "$bit_config" \
          --context-length "$context_len" \
          $packing_flag \
          --device "$DEVICE" \
          --num-layers "$NUM_LAYERS" \
          --num-heads "$NUM_HEADS" \
          --head-dim "$HEAD_DIM" \
          --num-samples "$NUM_SAMPLES" \
          --output "$output_file" \
          --verbose

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
          # Extract key metrics and append to summary
          memory_ratio=$(python -c "import json; data=json.load(open('$output_file')); print(data['memory']['memory_ratio'])")
          avg_bits=$(python -c "import json; data=json.load(open('$output_file')); print(data['memory']['avg_bits'])")
          allocated_gb=$(python -c "import json; data=json.load(open('$output_file')); print(data['memory']['allocated_gb'])")
          throughput=$(python -c "import json; data=json.load(open('$output_file')); print(data['performance']['throughput_tps'])")
          latency=$(python -c "import json; data=json.load(open('$output_file')); print(data['performance']['avg_latency_ms'])")

          echo "$experiment_num,$budget,$bit_config,$context_len,$use_packing,$memory_ratio,$avg_bits,$allocated_gb,$throughput,$latency" >> "$summary_csv"

          echo "✅ Experiment $exp_name completed successfully"
        else
          echo "❌ Experiment $exp_name failed (exit code: $EXIT_CODE)"
        fi

        echo ""
      done
    done
  done
done

echo ""
echo "========================================"
echo "Ablation study completed at $(date)"
echo "========================================"
echo ""
echo "Summary saved to: $summary_csv"
echo "Individual results in: $OUTPUT_ROOT"
echo ""

# Print summary statistics
echo "Summary Statistics:"
python -c "
import csv
with open('$summary_csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    if rows:
        print(f'  Total experiments: {len(rows)}')
        budgets = sorted(set(float(r['budget']) for r in rows))
        print(f'  Budget range: {min(budgets):.1%} - {max(budgets):.1%}')
        bits = sorted(set(float(r['avg_bits']) for r in rows))
        print(f'  Avg bits range: {min(bits):.2f} - {max(bits):.2f}')
        throughputs = [float(r['throughput_tps']) for r in rows]
        print(f'  Throughput range: {min(throughputs):.0f} - {max(throughputs):.0f} tokens/sec')
"

echo ""
echo "Ablation study complete!"
