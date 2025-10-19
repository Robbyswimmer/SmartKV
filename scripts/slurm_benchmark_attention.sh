#!/bin/bash -l

#SBATCH --job-name="SmartKV-GPU-Benchmark"
#SBATCH --output=logs/benchmark_attention_%j.txt
#SBATCH --error=logs/benchmark_attention_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu

set -eo pipefail

PROJECT_DIR=/data/SalmanAsif/RobbyMoseley/SmartKV/SmartKV
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-smartkv}
GCC_VERSION=${GCC_VERSION:-11}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "$CONDA_ENV"

echo "Installing GCC ${GCC_VERSION} for CUDA compilation..."
conda install -y -c conda-forge gxx_linux-64=${GCC_VERSION} -q

conda deactivate
conda activate "$CONDA_ENV"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=${CXX}
export NVCC_PREPEND_FLAGS="--compiler-bindir ${CXX}"
export TORCH_NVCC_FLAGS="--compiler-bindir ${CXX}"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
echo "Using GCC: $(${CXX} --version | head -1)"

mkdir -p logs

PYTHONUNBUFFERED=1 python - <<'PY'
import importlib
spec = importlib.util.find_spec('smartkv_cuda')
if spec is None:
    raise SystemExit(1)
PY

if [[ $? -ne 0 ]]; then
  echo "smartkv_cuda not found; building extension..."
  pip install -e . --no-build-isolation
  CUDAHOSTCXX=${CXX} python setup.py build_ext --inplace
fi

PYTHONUNBUFFERED=1 python - <<'PY'
import importlib
import sys
sys.exit(0 if importlib.util.find_spec('smartkv_cuda') else 1)
PY

if [[ $? -ne 0 ]]; then
  echo "ERROR: smartkv_cuda extension is still unavailable after build." >&2
  exit 1
fi

: "${CONTEXT_LENGTH:=4096}"
: "${NUM_HEADS:=32}"
: "${HEAD_DIM:=128}"
: "${MEMORY_BUDGET:=0.3}"
: "${DEVICE:=cuda}"
: "${ITERS:=200}"
: "${WARMUP:=20}"
: "${SEED:=42}"

python scripts/benchmark_gpu_attention.py \
  --context-length "$CONTEXT_LENGTH" \
  --num-heads "$NUM_HEADS" \
  --head-dim "$HEAD_DIM" \
  --device "$DEVICE" \
  --memory-budget "$MEMORY_BUDGET" \
  --iters "$ITERS" \
  --warmup "$WARMUP" \
  --seed "$SEED" \
  --enable-packing

