#!/bin/bash -l

#SBATCH --job-name="SmartKV-BucketTests"
#SBATCH --output=logs/bucket_tests_%j.txt
#SBATCH --error=logs/bucket_tests_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -eo pipefail

# Activate conda
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

CONDA_ENV=${CONDA_ENV:-smartkv}
conda activate "$CONDA_ENV"

# Ensure modern GCC toolchain for nvcc
conda install -y -c conda-forge gxx_linux-64=11 -q
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

cd /data/SalmanAsif/RobbyMoseley/SmartKV/SmartKV

# Rebuild CUDA extension
python setup.py build_ext --inplace

# Debug script
python tests/test_debug_bucket.py

# Targeted tests
pytest \
  tests/test_quant_cuda.py::test_bucketed_attention_matches_legacy \
  tests/test_quant_cuda.py::test_bucket_kernel_bug_fixes \
  -v -s --tb=short
