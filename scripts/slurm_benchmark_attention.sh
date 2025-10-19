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
export ADDR2LINE=${ADDR2LINE:-addr2line}
export AR=${AR:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ar}
export RANLIB=${RANLIB:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ranlib}
export LD=${LD:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-ld}
export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=${CXX}
export NVCC_PREPEND_FLAGS="--compiler-bindir ${CXX}"
export TORCH_NVCC_FLAGS="--compiler-bindir ${CXX}"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
echo "Using GCC: $(${CXX} --version | head -1)"

mkdir -p logs

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [[ "${TORCH_VERSION}" == "missing" ]]; then
  echo "PyTorch not found in environment. Install torch with CUDA support before running." >&2
fi

if ! python - <<'PY'
import importlib
spec = importlib.util.find_spec('smartkv_cuda')
if spec is None:
    raise SystemExit(1)
PY
then
  echo "smartkv_cuda not found; (re)building extension..."
  pip uninstall smartkv -y -q || true
  pip install -e . --no-build-isolation
  CUDAHOSTCXX=${CXX} python setup.py build_ext --inplace

  SO_FILE=$(find . -name "smartkv_cuda*.so" -type f 2>/dev/null | head -1)
  if [[ -n "${SO_FILE}" ]]; then
    echo "Found extension at ${SO_FILE}"
    if [[ ! -f "./$(basename ${SO_FILE})" ]]; then
      cp "${SO_FILE}" .
      echo "Copied extension to project root"
    fi
  else
    echo "WARNING: smartkv_cuda .so not found after build" >&2
  fi
fi

if ! python - <<'PY'
import importlib
import sys
sys.exit(0 if importlib.util.find_spec('smartkv_cuda') else 1)
PY
then
  echo "ERROR: smartkv_cuda extension is still unavailable after build." >&2
  exit 1
fi

TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

echo "Verifying CUDA kernel availability..."
python -c "from smartkv.kernels import CUDA_AVAILABLE; print(f'SmartKV CUDA kernels available: {CUDA_AVAILABLE}')"

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

