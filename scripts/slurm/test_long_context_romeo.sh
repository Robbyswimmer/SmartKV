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

# Benchmark parameters (override via environment)
CONTEXT_FILE=${CONTEXT_FILE:-data/romeo_juliet.txt}
DEVICE=${DEVICE:-cuda}
MEMORY_BUDGET=${MEMORY_BUDGET:-0.25}
CONTEXTS=${CONTEXTS:-"4096 8192 16384 32000 48000"}
FP16_MAX=${FP16_MAX:-16384}
INT8_MAX=${INT8_MAX:-32000}
SMARTKV_MAX=${SMARTKV_MAX:-48000}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-80}
OUTPUT_DIR=${OUTPUT_DIR:-results/real_context/${SLURM_JOB_ID}}

mkdir -p "${OUTPUT_DIR}"

echo "\n=== Real-context SmartKV benchmark (Romeo & Juliet) ==="
echo "Context file: ${CONTEXT_FILE}"
echo "Device:       ${DEVICE}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Contexts:     ${CONTEXTS}"
echo "FP16 max:     ${FP16_MAX}"
echo "INT8 max:     ${INT8_MAX}"
echo "SmartKV max:  ${SMARTKV_MAX}"

IFS=' ' read -r -a CONTEXT_ARR <<< "${CONTEXTS}"

CMD_ARGS=(
  scripts/bench_real_context.py
  --document "${CONTEXT_FILE}"
  --device "${DEVICE}"
  --memory-budget "${MEMORY_BUDGET}"
  --fp16-max "${FP16_MAX}"
  --int8-max "${INT8_MAX}"
  --smartkv-max "${SMARTKV_MAX}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --output "${OUTPUT_DIR}/bench_results.json"
  --contexts
)

for ctx in "${CONTEXT_ARR[@]}"; do
  CMD_ARGS+=("${ctx}")
done

echo "\nRunning real-context benchmark..."
python "${CMD_ARGS[@]}"
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
  echo "❌ Benchmark failed with status ${STATUS}"
else
  echo "\n✅ Benchmark completed successfully"
  echo "Results directory: ${OUTPUT_DIR}"
  if [[ -f "${OUTPUT_DIR}/bench_results.json" ]]; then
    echo "Summary:"
    cat "${OUTPUT_DIR}/bench_results.json"
  fi
fi

exit $STATUS
