#!/bin/bash -l

#SBATCH --job-name="SmartKV-Llama31-Eval"
#SBATCH --output=logs/llama31_eval_%j.txt
#SBATCH --error=logs/llama31_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

export PYTHONUNBUFFERED=1
export ADDR2LINE=${ADDR2LINE:-addr2line}
export AR=${AR:-ar}
export RANLIB=${RANLIB:-ranlib}
export LD=${LD:-ld}
export ADDR2LINE=${ADDR2LINE:-addr2line}

# Navigate to project directory
cd /data/SalmanAsif/RobbyMoseley/SmartKV/SmartKV

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

mkdir -p logs experiments

# Install modern GCC compiler for CUDA compilation
echo "Installing GCC 11 for CUDA compilation..."
conda install -y -c conda-forge gxx_linux-64=11 -q

# Reactivate environment to load compiler paths
set +u  # Temporarily disable unbound variable check for conda
conda deactivate
conda activate "${CONDA_ENV}"
set -u  # Re-enable unbound variable check

# Set compiler environment variables
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
echo "Using GCC: $(${CXX} --version | head -1)"

# Build CUDA extensions for the assigned GPU
echo "Building CUDA extensions for this GPU..."
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Install package
pip uninstall smartkv -y -q
pip install -e . --no-build-isolation 2>&1 | tee logs/build_output.log

# Force build CUDA extensions
echo "Compiling CUDA extensions..."
python setup.py build_ext --inplace 2>&1 | tee -a logs/build_output.log

echo "Verifying CUDA kernels..."
python -c "from smartkv.kernels import CUDA_AVAILABLE; print(f'SmartKV CUDA kernels available: {CUDA_AVAILABLE}')"
echo "CUDA build complete."

# Configuration
MODEL_PATH=${MODEL_PATH:-/data/SalmanAsif/RobbyMoseley/SmartKV/SmartKV/models/llama-3.1-8b}
DEVICE=${DEVICE:-cuda}
BUDGET=${BUDGET:-0.5}
MAX_TOKENS=${MAX_TOKENS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/llama31_eval}

echo ""
echo "=== SmartKV Comprehensive Evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Device: ${DEVICE}"
echo "Memory Budget: ${BUDGET}"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run comprehensive evaluation
python -u smartkv/experiments/comprehensive_evaluation.py \
    --model "${MODEL_PATH}" \
    --device "${DEVICE}" \
    --budget "${BUDGET}" \
    --max-tokens "${MAX_TOKENS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Evaluation Complete at $(date) ==="
