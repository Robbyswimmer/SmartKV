#!/bin/bash -l

#SBATCH --job-name="SmartKV-UnitTests"
#SBATCH --output=logs/unit_tests_%j.txt
#SBATCH --error=logs/unit_tests_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
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

echo "Starting SmartKV CUDA kernel unit tests at $(date)"

# Create logs directory
mkdir -p logs

# Configuration
DEVICE=${DEVICE:-cuda:0}
TEST_MODULES=${TEST_MODULES:-"test_cuda_kernels test_bit_packing"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/results/unit_tests/${SLURM_JOB_ID}"}

mkdir -p "$OUTPUT_ROOT"

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Test modules: $TEST_MODULES"
echo "  Output: $OUTPUT_ROOT"
echo ""

# Check CUDA availability
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Running pytest unit tests..."

# Run unit tests with pytest
pytest tests/test_cuda_kernels.py \
  tests/test_bit_packing.py \
  -v \
  -s \
  --tb=short \
  --junitxml="$OUTPUT_ROOT/junit_results.xml" \
  | tee "$OUTPUT_ROOT/test_output.txt"

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ All unit tests passed!"
else
  echo ""
  echo "❌ Some unit tests failed (exit code: $TEST_EXIT_CODE)"
fi

echo ""
echo "SmartKV CUDA kernel unit tests completed at $(date)"
echo "Results saved to: $OUTPUT_ROOT"

exit $TEST_EXIT_CODE
