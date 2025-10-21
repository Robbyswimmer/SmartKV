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

set -eo pipefail  # -u breaks conda activate (ADDR2LINE unset in binutils script)

# Activate conda environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-smartkv}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

# Ensure consistent toolchain for CUDA builds/tests (matches profiling script)
echo "Ensuring GCC 11 toolchain for CUDA..."
conda install -y -c conda-forge gxx_linux-64=11 -q

# Reactivate to pick up compiler wrappers
conda deactivate
conda activate "${CONDA_ENV}"

# Export toolchain-related environment variables to avoid unbound var issues
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export ADDR2LINE=${ADDR2LINE:-addr2line}
export AR=${AR:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ar}
export RANLIB=${RANLIB:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ranlib}
export LD=${LD:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ld}
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=${CXX}
export NVCC_PREPEND_FLAGS="--compiler-bindir ${CXX}"
export TORCH_NVCC_FLAGS="--compiler-bindir ${CXX}"
echo "Using GCC toolchain: $(${CXX} --version | head -1)"

# Add PyTorch lib directory to LD_LIBRARY_PATH so libc10.so and other torch libs can be found
TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

echo "Starting SmartKV CUDA kernel unit tests at $(date)"

# Create logs directory
mkdir -p logs

# Always run from repository root (Slurm sets SLURM_SUBMIT_DIR)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

# Load explicit CUDA module if requested; otherwise try common defaults
if command -v module >/dev/null 2>&1; then
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    echo "Loading CUDA module '${CUDA_MODULE}'"
    module load "${CUDA_MODULE}" >/dev/null 2>&1 || echo "Warning: failed to load ${CUDA_MODULE}"
  fi

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not detected; attempting to load a default CUDA module"
    CUDA_MODULE_CANDIDATES=(cuda cuda/latest cuda/12.2 cuda/12.1 cuda/12.0 cuda/11.8 cuda/11.7 cuda/11.6)
    for candidate in "${CUDA_MODULE_CANDIDATES[@]}"; do
      module load "${candidate}" >/dev/null 2>&1 || continue
      if command -v nvcc >/dev/null 2>&1; then
        echo "Loaded CUDA module '${candidate}'"
        break
      fi
    done
  fi
fi

# Discover nvcc and CUDA_HOME so setup.py builds the extension
if command -v nvcc >/dev/null 2>&1; then
  NVCC_PATH=$(command -v nvcc)
  export CUDA_HOME="${CUDA_HOME:-$(dirname "${NVCC_PATH}")/..}"
  echo "Detected nvcc at ${NVCC_PATH}"
  echo "Setting CUDA_HOME=${CUDA_HOME}"
else
  echo "❌ nvcc not found on PATH. Please load a CUDA module (set CUDA_MODULE) or export CUDA_HOME before submitting."
  exit 1
fi

echo "Rebuilding SmartKV CUDA extension (force fresh .so for this job)..."
rm -rf build smartkv_cuda.*.so
LOG_SUFFIX=${SLURM_JOB_ID:-$$}
python -m pip install -e . --no-deps --force-reinstall >/tmp/smartkv_cuda_build_${LOG_SUFFIX}.log 2>&1 || {
  echo "❌ CUDA extension rebuild failed. See /tmp/smartkv_cuda_build_${LOG_SUFFIX}.log"
  cat /tmp/smartkv_cuda_build_${LOG_SUFFIX}.log
  exit 1
}
python - <<'PY'
import smartkv_cuda
print("Loaded smartkv_cuda from:", smartkv_cuda.__file__)
PY

# Configuration
DEVICE=${DEVICE:-cuda:0}
TEST_MODULES=${TEST_MODULES:-"test_cuda_kernels test_bit_packing test_quant_cuda"}
RUN_BENCHMARK=${RUN_BENCHMARK:-false}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/results/unit_tests/${SLURM_JOB_ID}"}

mkdir -p "$OUTPUT_ROOT"

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Test modules: $TEST_MODULES"
echo "  Run benchmark: $RUN_BENCHMARK"
echo "  Output: $OUTPUT_ROOT"
echo ""

# Check CUDA availability
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Running pytest unit tests..."

# Run unit tests with pytest
pytest tests/test_cuda_kernels.py \
  tests/test_bit_packing.py \
  tests/test_quant_cuda.py \
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

# Run benchmark if requested
if [ "$RUN_BENCHMARK" = "true" ]; then
  echo ""
  echo "Running bucket kernel benchmark..."
  pytest tests/test_bucket_kernel_benchmark.py::test_bucket_kernel_vs_uniform_int8 \
    -v \
    -s \
    | tee "$OUTPUT_ROOT/benchmark_output.txt"

  BENCH_EXIT_CODE=$?
  if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo "⚠️  Benchmark failed (exit code: $BENCH_EXIT_CODE)"
  fi
fi

echo ""
echo "SmartKV CUDA kernel unit tests completed at $(date)"
echo "Results saved to: $OUTPUT_ROOT"

exit $TEST_EXIT_CODE
