#!/bin/bash -l

#SBATCH --job-name="SmartKV-Profile"
#SBATCH --output=logs/smartkv_profile_%j.txt
#SBATCH --error=logs/smartkv_profile_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -eo pipefail  # Removed -u to avoid conda activation issues

export PYTHONUNBUFFERED=1

# Navigate to project directory
PROJECT_DIR=/data/SalmanAsif/RobbyMoseley/SmartKV/SmartKV
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-smartkv}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

mkdir -p logs

# Install modern GCC compiler for CUDA compilation
echo "Installing GCC 11 for CUDA compilation..."
conda install -y -c conda-forge gxx_linux-64=11 -q

# Reactivate environment to load compiler paths
conda deactivate
conda activate "${CONDA_ENV}"

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
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
echo "Using GCC: $(${CXX} --version | head -1)"

# Build CUDA extensions for the assigned GPU
echo "Building CUDA extensions for this GPU..."
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Ensure PyTorch matches CUDA version
echo "Reinstalling PyTorch for CUDA 12.1..."
pip install --force-reinstall torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install package
echo "Installing SmartKV with CUDA extensions..."
pip uninstall smartkv -y -q
pip install -e . --no-build-isolation 2>&1 | tee logs/build_output.log

# Force build CUDA extensions with explicit compiler
echo "Compiling CUDA extensions..."
CUDAHOSTCXX=${CXX} python setup.py build_ext --inplace 2>&1 | tee -a logs/build_output.log

# Ensure the .so file is findable (check both possible locations)
echo "Locating compiled extension..."
SO_FILE=$(find . -name "smartkv_cuda*.so" -type f 2>/dev/null | head -1)
if [[ -n "${SO_FILE}" ]]; then
    echo "Found extension: ${SO_FILE}"
    # Copy to root if not already there
    if [[ ! -f "./smartkv_cuda$(basename ${SO_FILE} | grep -o '\..*')" ]]; then
        cp "${SO_FILE}" .
        echo "Copied to project root for import"
    fi
else
    echo "WARNING: No .so file found after build!"
fi

echo "Verifying CUDA kernels..."
python -c "from smartkv.kernels import CUDA_AVAILABLE; print(f'SmartKV CUDA kernels available: {CUDA_AVAILABLE}')"

echo "Testing CUDA kernel import directly..."
python -c "
try:
    import smartkv_cuda
    print('Direct import of smartkv_cuda: SUCCESS')
    print(f'Module location: {smartkv_cuda.__file__}')
except ImportError as e:
    print(f'Direct import of smartkv_cuda: FAILED - {e}')
    import sys
    print(f'Python path: {sys.path[:3]}')
"
echo "CUDA build complete."

PROFILE_DEVICE=${PROFILE_DEVICE:-cuda}
BATCH_SIZES=${BATCH_SIZES:-"1 2 4"}
K_LEN=${K_LEN:-2048}
HEADS=${HEADS:-16}
HEAD_DIM=${HEAD_DIM:-128}
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}
CACHE_BUDGET=${CACHE_BUDGET:-0.4}
ENABLE_PACKING=${ENABLE_PACKING:-0}
USE_MASK=${USE_MASK:-1}
USE_CUDA=${USE_CUDA:-1}
SEED=${SEED:-42}
ENABLE_FORECAST=${ENABLE_FORECAST:-0}
FORECAST_HISTORY=${FORECAST_HISTORY:-2048}
FORECAST_UPDATE_INTERVAL=${FORECAST_UPDATE_INTERVAL:-32}
FORECAST_BLEND=${FORECAST_BLEND:-0.5}
FORECAST_LR=${FORECAST_LR:-0.05}

packing_flag=()
if [[ "${ENABLE_PACKING}" != "0" ]]; then
  packing_flag+=(--enable-packing)
fi

mask_flag=()
if [[ "${USE_MASK}" != "0" ]]; then
  mask_flag+=(--use-mask)
fi

cuda_flag=()
if [[ "${USE_CUDA}" != "0" ]]; then
  cuda_flag+=(--use-cuda)
fi

forecast_flags=()
if [[ "${ENABLE_FORECAST}" != "0" ]]; then
  forecast_flags+=(--enable-forecast --forecast-history "${FORECAST_HISTORY}" \
                   --forecast-update-interval "${FORECAST_UPDATE_INTERVAL}" \
                   --forecast-blend "${FORECAST_BLEND}" \
                   --forecast-lr "${FORECAST_LR}")
fi

for bs in ${BATCH_SIZES}; do
  echo "\n=== Profiling SmartKV (batch_size=${bs}) at $(date) ==="
  python -u scripts/profile_smartkv.py \
    --device "${PROFILE_DEVICE}" \
    --batch-size "${bs}" \
    --num-heads "${HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --k-len "${K_LEN}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    --cache-budget "${CACHE_BUDGET}" \
    --seed "${SEED}" \
    ${packing_flag[@]} \
    ${mask_flag[@]} \
    ${cuda_flag[@]} \
    ${forecast_flags[@]}
done

echo "SmartKV profiling sweep complete at $(date)"

# Optional long-context profiling with Romeo and Juliet
if [[ "${ENABLE_LONG_CONTEXT:-0}" != "0" ]]; then
  echo "\n=== Profiling long-context document (${PROFILE_DEVICE}) at $(date) ==="
  python -u scripts/profile_long_context.py \
    --document "${LONG_CONTEXT_DOC:-data/romeo_juliet.txt}" \
    --device "${PROFILE_DEVICE}" \
    --memory-budget "${CACHE_BUDGET}" \
    --num-heads "${HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --chunk-tokens "${LONG_CONTEXT_CHUNK:-256}" \
    --seed "${SEED}" \
    $( [[ "${ENABLE_FORECAST}" != "0" ]] && echo --enable-forecast ) \
    --forecast-history "${FORECAST_HISTORY}" \
    --forecast-update-interval "${FORECAST_UPDATE_INTERVAL}" \
    --forecast-blend "${FORECAST_BLEND}" \
    --forecast-lr "${FORECAST_LR}"
fi
