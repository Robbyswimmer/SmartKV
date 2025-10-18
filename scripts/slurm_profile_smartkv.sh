#!/bin/bash -l

#SBATCH --job-name="SmartKV-Profile"
#SBATCH --output=logs/smartkv_profile_%j.txt
#SBATCH --error=logs/smartkv_profile_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

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

# Build CUDA extensions for the assigned GPU
echo "Building CUDA extensions for this GPU..."
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

pip uninstall smartkv -y -q
pip install -e . --no-build-isolation 2>&1 | tee logs/build_output.log

echo "Verifying CUDA kernels..."
python -c "from smartkv.kernels import CUDA_AVAILABLE; print(f'SmartKV CUDA kernels available: {CUDA_AVAILABLE}')"
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
