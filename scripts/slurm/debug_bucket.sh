#!/bin/bash -l

#SBATCH --job-name="SmartKV-Debug"
#SBATCH --output=logs/debug_%j.txt
#SBATCH --error=logs/debug_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -p gpu

set -eo pipefail

# Activate conda
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-smartkv}
conda activate "${CONDA_ENV}"

# GCC toolchain
conda install -y -c conda-forge gxx_linux-64=11 -q
conda deactivate
conda activate "${CONDA_ENV}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=${CXX}
export NVCC_PREPEND_FLAGS="--compiler-bindir ${CXX}"
export TORCH_NVCC_FLAGS="--compiler-bindir ${CXX}"

# Load CUDA module (needed for nvcc during build)
if command -v module >/dev/null 2>&1; then
  CUDA_MODULE_CANDIDATES=(cuda cuda/latest cuda/12.2 cuda/12.1 cuda/12.0 cuda/11.8)
  for candidate in "${CUDA_MODULE_CANDIDATES[@]}"; do
    module load "${candidate}" >/dev/null 2>&1 || continue
    if command -v nvcc >/dev/null 2>&1; then
      echo "Loaded CUDA module '${candidate}'"
      break
    fi
  done
fi

# Set CUDA_HOME
if command -v nvcc >/dev/null 2>&1; then
  NVCC_PATH=$(command -v nvcc)
  export CUDA_HOME="${CUDA_HOME:-$(dirname "${NVCC_PATH}")/..}"
  echo "CUDA_HOME=${CUDA_HOME}"
else
  echo "❌ nvcc not found, CUDA extension build will fail"
  exit 1
fi

# Add PyTorch lib directory to LD_LIBRARY_PATH so libc10.so and other torch libs can be found
TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

# Rebuild the CUDA extension
echo "Rebuilding SmartKV CUDA extension..."
rm -rf build smartkv_cuda.*.so
pip install -e . --no-build-isolation >/tmp/smartkv_debug_build.log 2>&1 || {
  echo "❌ CUDA extension rebuild failed. See /tmp/smartkv_debug_build.log"
  cat /tmp/smartkv_debug_build.log
  exit 1
}

# Ensure build completes with explicit build_ext
CUDAHOSTCXX=${CXX} python setup.py build_ext --inplace >>/tmp/smartkv_debug_build.log 2>&1

# Find and verify .so file
SO_FILE=$(find . -name "smartkv_cuda*.so" -type f 2>/dev/null | head -1)
if [[ -n "${SO_FILE}" ]]; then
  echo "Found extension: ${SO_FILE}"
else
  echo "❌ No .so file found after build!"
  cat /tmp/smartkv_debug_build.log
  exit 1
fi

# Add project root to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== Environment Info ==="
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

echo ""
echo "=== Checking smartkv_cuda Extension ==="
python - <<'PY'
try:
    import smartkv_cuda
    print("✓ smartkv_cuda loaded from:", smartkv_cuda.__file__)
    print("  Available functions:", dir(smartkv_cuda))

    # Check if bucket kernel is available
    if hasattr(smartkv_cuda, 'quantized_attention_bucket_forward'):
        print("✓ quantized_attention_bucket_forward found")
    else:
        print("✗ quantized_attention_bucket_forward NOT found")
except ImportError as e:
    print("✗ Failed to import smartkv_cuda:", e)
PY

echo ""
echo "=== Running Debug Test ==="
python tests/test_debug_bucket.py

echo ""
echo "Debug completed at $(date)"
