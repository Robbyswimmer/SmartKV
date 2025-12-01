#!/bin/bash -l

#SBATCH --job-name="smartkv-ppl-gpu"
#SBATCH --output=logs/ppl_gpu_%j.out
#SBATCH --error=logs/ppl_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -p gpu

set -eo pipefail

MODEL=${MODEL:-NousResearch/Meta-Llama-3.1-8B}
OUTPUT_DIR=${OUTPUT_DIR:-results/wikitext2_perplexity_llama3_gpu}
BLOCK_SIZE=${BLOCK_SIZE:-512}
MAX_EVAL_TOKENS=${MAX_EVAL_TOKENS:-8192}
BUDGETS=${BUDGETS:-"0.35 0.5 0.7"}
DEVICE=${DEVICE:-cuda}
CONDA_ENV=${CONDA_ENV:-smartkv}

# Activate environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

# Optional: rebuild CUDA extension (comment out if already built on this node)
# rm -rf build smartkv_cuda.*.so smartkv.egg-info dist
# CUDAHOSTCXX=${CXX:-$(which g++)} python setup.py build_ext --inplace --force

export PYTHONPATH="${PWD}:${PYTHONPATH}"
mkdir -p logs

echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Budgets: ${BUDGETS}"
echo "Block size: ${BLOCK_SIZE}"
echo "Max eval tokens: ${MAX_EVAL_TOKENS}"
echo "Device: ${DEVICE}"
echo "Conda env: ${CONDA_ENV}"

python -m smartkv.experiments.run_wikitext_perplexity \
  --model "${MODEL}" \
  --dataset wikitext --subset wikitext-2-raw-v1 --split test \
  --block-size "${BLOCK_SIZE}" \
  --max-eval-tokens "${MAX_EVAL_TOKENS}" \
  --budgets ${BUDGETS} \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}"
