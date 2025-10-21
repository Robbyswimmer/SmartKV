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

# Add PyTorch lib directory to LD_LIBRARY_PATH so libc10.so and other torch libs can be found
TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

# Rebuild the CUDA extension to ensure it's fresh and properly linked
echo "Rebuilding SmartKV CUDA extension..."
rm -rf build smartkv_cuda.*.so
python -m pip install -e . --no-deps --force-reinstall >/tmp/smartkv_debug_build.log 2>&1 || {
  echo "❌ CUDA extension rebuild failed. See /tmp/smartkv_debug_build.log"
  cat /tmp/smartkv_debug_build.log
  exit 1
}

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
