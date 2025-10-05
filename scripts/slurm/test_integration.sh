#!/bin/bash -l

#SBATCH --job-name="SmartKV-Integration"
#SBATCH --output=logs/integration_%j.txt
#SBATCH --error=logs/integration_%j.err
#SBATCH --time=02:00:00
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

echo "Starting SmartKV integration tests at $(date)"

# Create logs directory
mkdir -p logs

# Configuration
DEVICE=${DEVICE:-cuda:0}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"512 1024 2048 4096"}
USE_FUSED_GPU=${USE_FUSED_GPU:-true}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/results/integration/${SLURM_JOB_ID}"}

mkdir -p "$OUTPUT_ROOT"

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Context lengths: $CONTEXT_LENGTHS"
echo "  Use fused GPU: $USE_FUSED_GPU"
echo "  Output: $OUTPUT_ROOT"
echo ""

# Check CUDA availability
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Running integration tests..."

# Run integration tests with pytest
pytest tests/test_gpu_generation.py \
  -v \
  -s \
  --tb=short \
  -k "not slow or test_long_context_gpu" \
  --junitxml="$OUTPUT_ROOT/junit_integration.xml" \
  | tee "$OUTPUT_ROOT/integration_output.txt"

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ All integration tests passed!"
else
  echo ""
  echo "❌ Some integration tests failed (exit code: $TEST_EXIT_CODE)"
fi

echo ""
echo "SmartKV integration tests completed at $(date)"
echo "Results saved to: $OUTPUT_ROOT"

exit $TEST_EXIT_CODE
