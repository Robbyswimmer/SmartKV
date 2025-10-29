#!/bin/bash -l

#SBATCH --job-name="SmartKV-Romeo"
#SBATCH --output=logs/long_context_%j.txt
#SBATCH --error=logs/long_context_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH -p gpu

set -eo pipefail

# Activate conda environment (matches other SmartKV SLURM jobs)
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-smartkv}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

# Ensure GCC toolchain compatible with CUDA builds (optional but consistent)
echo "Ensuring GCC 11 toolchain for CUDA builds..."
conda install -y -c conda-forge gxx_linux-64=11 -q
conda deactivate
conda activate "${CONDA_ENV}"

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

# Add PyTorch lib directory to LD_LIBRARY_PATH for extension loading
TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

mkdir -p logs

# Always execute from repository root
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

echo "Rebuilding SmartKV CUDA extension at $(date)"
rm -rf build smartkv_cuda.*.so smartkv.egg-info dist *.egg-info
touch smartkv/csrc/*.cu smartkv/csrc/*.cpp smartkv/csrc/*.h
LOG_SUFFIX=${SLURM_JOB_ID:-$$}

CUDAHOSTCXX=${CXX} python setup.py build_ext --inplace --force \
  >/tmp/smartkv_cuda_build_${LOG_SUFFIX}.log 2>&1 || {
    echo "❌ CUDA extension rebuild failed. See /tmp/smartkv_cuda_build_${LOG_SUFFIX}.log"
    cat /tmp/smartkv_cuda_build_${LOG_SUFFIX}.log
    exit 1
  }

SO_FILE=$(find . -maxdepth 1 -name "smartkv_cuda*.so" -type f | head -1)
if [[ -z "${SO_FILE}" ]]; then
  echo "❌ smartkv_cuda shared library not found after build"
  cat /tmp/smartkv_cuda_build_${LOG_SUFFIX}.log
  exit 1
fi
echo "Found extension: ${SO_FILE}"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

python - <<'PY'
import smartkv_cuda
print("Loaded smartkv_cuda from:", smartkv_cuda.__file__)
PY

# Experiment parameters (override via environment)
MODEL_NAME=${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
CONTEXT_FILE=${CONTEXT_FILE:-data/romeo_juliet.txt}
DEVICE=${DEVICE:-cuda}
BUDGETS=${BUDGETS:-"0.50 0.35 0.25"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-65536}
OUTPUT_DIR=${OUTPUT_DIR:-results/long_context/${SLURM_JOB_ID}}
USE_FUSED_CPU=${USE_FUSED_CPU:-0}

mkdir -p "${OUTPUT_DIR}"

echo "\n=== Long-context evaluation on Romeo and Juliet ==="
echo "Model:        ${MODEL_NAME}"
echo "Context file: ${CONTEXT_FILE}"
echo "Device:       ${DEVICE}"
echo "Budgets:      ${BUDGETS}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Max tokens:   ${MAX_NEW_TOKENS}"
echo "Max context:  ${MAX_CONTEXT_TOKENS}"
echo "Use fused CPU: ${USE_FUSED_CPU}"

IFS=' ' read -r -a BUDGET_ARR <<< "${BUDGETS}"

CMD_ARGS=(
  -m smartkv.experiments.long_context_test
  --model "${MODEL_NAME}"
  --context-file "${CONTEXT_FILE}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
  --max-tokens "${MAX_NEW_TOKENS}"
  --max-context-tokens "${MAX_CONTEXT_TOKENS}"
)

if [[ "${USE_FUSED_CPU}" != "0" ]]; then
  CMD_ARGS+=(--use-fused-cpu)
fi

CMD_ARGS+=(--budgets)
for budget in "${BUDGET_ARR[@]}"; do
  CMD_ARGS+=("${budget}")
done

echo "\nRunning long-context evaluation..."
python "${CMD_ARGS[@]}"
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
  echo "❌ Long-context evaluation failed with status ${STATUS}"
else
  echo "\n✅ Long-context evaluation completed successfully"
  echo "Results directory: ${OUTPUT_DIR}"
  if [[ -f "${OUTPUT_DIR}/long_context_results.json" ]]; then
    echo "Summary:"
    cat "${OUTPUT_DIR}/long_context_results.json"
  fi
fi

exit $STATUS
