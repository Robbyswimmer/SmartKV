# SmartKV GPU Testing Guide

Comprehensive testing framework for validating SmartKV GPU implementation with SLURM-based ablation studies.

## Quick Start

### 1. Build CUDA Extension

```bash
# One-time setup
pip install -e .
```

### 2. Run Tests Locally

```bash
# Unit tests (CUDA kernels)
pytest tests/test_cuda_kernels.py -v -s
pytest tests/test_bit_packing.py -v -s

# Integration tests
pytest tests/test_gpu_generation.py -v -s

# All GPU tests
pytest tests/test_*.py -k "cuda or gpu or bit_packing" -v
```

### 3. Run via SLURM

```bash
# Unit tests (~30 min, 1 GPU)
sbatch scripts/slurm/test_cuda_kernels.sh

# Integration tests (~2 hrs, 1 GPU)
sbatch scripts/slurm/test_integration.sh

# Ablation study (~24 hrs, 1 GPU) ⭐ KEY EXPERIMENT
sbatch scripts/slurm/ablate_memory.sh

# Benchmarks (~12 hrs, 1 GPU)
sbatch scripts/slurm/benchmark_gpu.sh
```

---

## Test Structure

### Unit Tests (Fast Validation)

**`tests/test_cuda_kernels.py`** - CUDA kernel correctness
- Quantization: CUDA vs CPU reference
- Fused attention: Numerical correctness
- Performance: GPU faster than CPU

**`tests/test_bit_packing.py`** - Bit-packing validation
- Pack/unpack roundtrip (2/3/4-bit)
- Compression ratios (2-bit: 4×, 3-bit: 2.67×, 4-bit: 2×)
- Memory savings vs FP16

**Usage:**
```bash
pytest tests/test_cuda_kernels.py::TestCUDAQuantization::test_quantization_cuda_vs_cpu_8bit -v
pytest tests/test_bit_packing.py::TestBitPacking::test_pack_unpack_roundtrip -v
```

### Integration Tests (End-to-End)

**`tests/test_gpu_generation.py`** - Full model inference
- GPU cache device placement
- CPU vs GPU consistency
- Bit-packing integration
- Long context (4K tokens)

**Usage:**
```bash
pytest tests/test_gpu_generation.py::TestGPUGeneration::test_gpu_cache_device_placement -v
pytest tests/test_gpu_generation.py::TestGPUGeneration::test_long_context_gpu -v -s
```

---

## Ablation Studies ⭐

### Memory Budget Sweep

**Script:** `scripts/run_ablation_study.py`

Tests a single configuration:
```bash
python scripts/run_ablation_study.py \
  --budget 0.5 \
  --bits 2,3,4,8 \
  --context-length 2048 \
  --use-bit-packing \
  --device cuda:0 \
  --output results/ablation_b50_2348_ctx2048.json
```

**Parameters:**
- `--budget`: Memory budget (0.2 - 0.7 of FP16)
- `--bits`: Bit widths (e.g., "2,3,4,8", "4,8")
- `--context-length`: Context length (512 - 8192)
- `--use-bit-packing`: Enable sub-byte storage
- `--num-samples`: Generation samples for metrics

**Output:**
```json
{
  "config": {
    "budget": 0.5,
    "bits": "2,3,4,8",
    "context_length": 2048,
    "use_bit_packing": true
  },
  "memory": {
    "memory_ratio": 0.48,
    "avg_bits": 4.2,
    "allocated_gb": 0.523,
    "precision_distribution": {
      "2-bit": 120,
      "3-bit": 340,
      "4-bit": 580,
      "8-bit": 1008
    }
  },
  "performance": {
    "avg_latency_ms": 15.3,
    "throughput_tps": 1245,
  },
  "compression": {
    "actual_vs_fp16": 0.45,
    "theoretical_ratio": 0.48
  }
}
```

### Full Ablation Sweep (SLURM)

**Script:** `scripts/slurm/ablate_memory.sh`

**Configuration:**
```bash
# In ablate_memory.sh
MEMORY_BUDGETS="0.2 0.3 0.4 0.5 0.6 0.7"               # 6 budgets
BIT_CONFIGS="2,3,4,8 4,8 2,4,8"                         # 3 configs
CONTEXT_LENGTHS="512 1024 2048 4096 8192"              # 5 contexts
USE_BIT_PACKING_MODES="true false"                     # 2 modes

# Total: 6 × 3 × 5 × 2 = 180 experiments (~8-12 hours)
```

**Submit:**
```bash
sbatch scripts/slurm/ablate_memory.sh
```

**Output Structure:**
```
results/ablations/<job_id>/
├── ablation_summary.csv              # Aggregated metrics
├── b0.2_bits2_3_4_8_ctx512_packed.json
├── b0.2_bits2_3_4_8_ctx512_unpacked.json
├── ...
└── b0.7_bits4_8_ctx8192_unpacked.json
```

**Summary CSV:**
```csv
experiment,budget,bits,context_length,use_packing,memory_ratio,avg_bits,allocated_gb,throughput_tps,avg_latency_ms
1,0.2,2,3,4,8,512,true,0.19,2.8,0.082,3241,0.158
2,0.2,2,3,4,8,512,false,0.47,2.8,0.195,3198,0.160
...
```

### Analyzing Results

```python
import pandas as pd

# Load summary
df = pd.read_csv('results/ablations/<job_id>/ablation_summary.csv')

# Budget vs Quality
budget_impact = df.groupby('budget').agg({
    'avg_bits': 'mean',
    'throughput_tps': 'mean',
    'memory_ratio': 'mean'
})

# Packing impact
packing_savings = df[df['use_packing'] == True]['allocated_gb'].mean() / \
                   df[df['use_packing'] == False]['allocated_gb'].mean()
print(f"Bit-packing saves: {(1 - packing_savings)*100:.1f}%")

# Context scaling
ctx_scaling = df.groupby('context_length')['throughput_tps'].mean()
```

---

## Performance Benchmarks

### Single Baseline

**Script:** `scripts/run_gpu_benchmark.py`

```bash
python scripts/run_gpu_benchmark.py \
  --baseline smartkv_mixed \
  --context-lengths 512,1024,2048,4096 \
  --device cuda:0 \
  --num-iterations 100 \
  --output results/benchmark_smartkv.csv
```

**Baselines:**
- `fp16`: No quantization (baseline)
- `uniform_8bit`: Uniform 8-bit quantization
- `uniform_4bit`: Uniform 4-bit quantization
- `smartkv_mixed`: SmartKV mixed precision (2/3/4/8-bit)

### Full Benchmark Suite (SLURM)

**Script:** `scripts/slurm/benchmark_gpu.sh`

```bash
# Runs all 4 baselines across 5 context lengths
sbatch scripts/slurm/benchmark_gpu.sh
```

**Output:**
```
results/benchmarks/<job_id>/
├── benchmark_comparison.csv          # Combined results
├── benchmark_fp16.csv
├── benchmark_uniform_8bit.csv
├── benchmark_uniform_4bit.csv
└── benchmark_smartkv_mixed.csv
```

**Comparison Table:**
```
Context Length: 2048
  Baseline                Latency (ms)      Throughput     Memory (GB)
  fp16                           12.50            1638           2.048
  uniform_8bit                    8.30            2469           1.024
  uniform_4bit                    6.10            3357           0.512
  smartkv_mixed                   7.20            2844           0.819
```

---

## SLURM Scripts Configuration

All SLURM scripts support environment variable configuration:

### Common Variables

```bash
# Environment
export CONDA_ENV=smartkv-gpu
export DEVICE=cuda:0

# Model config
export NUM_LAYERS=32
export NUM_HEADS=32
export HEAD_DIM=128

# Submit with custom config
MEMORY_BUDGETS="0.3 0.5 0.7" sbatch scripts/slurm/ablate_memory.sh
```

### Script-Specific Variables

**`test_cuda_kernels.sh`:**
```bash
export TEST_MODULES="test_cuda_kernels test_bit_packing"
```

**`test_integration.sh`:**
```bash
export CONTEXT_LENGTHS="512 1024 2048"
export USE_FUSED_GPU=true
```

**`ablate_memory.sh`:**
```bash
export MEMORY_BUDGETS="0.2 0.3 0.4 0.5"
export BIT_CONFIGS="2,3,4,8 4,8"
export CONTEXT_LENGTHS="1024 2048"
export USE_BIT_PACKING_MODES="true false"
```

**`benchmark_gpu.sh`:**
```bash
export BASELINES="fp16 smartkv_mixed"
export CONTEXT_LENGTHS="512,1024,2048"
export NUM_ITERATIONS=100
```

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/ablation_<job_id>.txt

# Check specific experiment
grep "Experiment 50" logs/ablation_<job_id>.txt

# Monitor GPU usage
ssh <compute-node> 'watch -n 1 nvidia-smi'
```

---

## Expected Results

### Unit Tests
- **All tests pass**: CUDA kernels match CPU reference
- **Bit-packing ratios**: 2-bit: 3.5-4×, 3-bit: 2.3-2.67×, 4-bit: 1.9-2×
- **Time**: ~5-10 minutes

### Integration Tests
- **Cache consistency**: CPU and GPU produce same results (rtol=1e-5)
- **Long context**: 4K tokens without errors
- **Time**: ~30-60 minutes

### Ablation Study
- **Budget vs bits**: Lower budget → lower avg bits
- **Packing savings**: 20-40% vs INT8 storage
- **Context scaling**: Throughput decreases with context length
- **Time**: ~8-12 hours for full sweep (180 experiments)

### Benchmarks
- **GPU speedup**: 2-3× faster than CPU
- **Memory savings**: 50-80% vs FP16 (depending on budget)
- **Latency**: <20ms per token @ 2K context
- **Time**: ~12 hours for all baselines

---

## Troubleshooting

### CUDA Extension Build Fails

```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Rebuild extension
pip uninstall smartkv
pip install -e . --force-reinstall
```

### Tests Fail with "CUDA not available"

```bash
# Verify GPU access
nvidia-smi

# Check SLURM GPU allocation
echo $CUDA_VISIBLE_DEVICES
```

### Out of Memory Errors

```bash
# Reduce context length or batch size
export CONTEXT_LENGTHS="512 1024"

# Or use smaller model
export NUM_LAYERS=8
export NUM_HEADS=8
```

### SLURM Job Pending

```bash
# Check queue
squeue -p gpu

# Check partition limits
scontrol show partition gpu
```

---

## Files Created

```
tests/
├── test_cuda_kernels.py          # CUDA kernel tests
├── test_bit_packing.py            # Bit-packing tests
└── test_gpu_generation.py         # Integration tests

scripts/
├── run_ablation_study.py          # Ablation runner
├── run_gpu_benchmark.py           # Benchmark runner
└── slurm/
    ├── test_cuda_kernels.sh       # Unit tests SLURM
    ├── test_integration.sh         # Integration tests SLURM
    ├── ablate_memory.sh           # Ablation study SLURM ⭐
    └── benchmark_gpu.sh           # Benchmarks SLURM
```

---

## Next Steps

1. **Build extension**: `pip install -e .`
2. **Run unit tests**: `pytest tests/test_cuda_kernels.py -v`
3. **Submit ablation**: `sbatch scripts/slurm/ablate_memory.sh`
4. **Analyze results**: Review `results/ablations/<job_id>/ablation_summary.csv`

---

## Key Metrics to Report

From ablation studies:

1. **Memory vs Quality Tradeoff**
   - Budget vs average bits
   - Budget vs perplexity/accuracy

2. **Bit-Packing Impact**
   - Memory savings: with vs without packing
   - Actual compression ratios achieved

3. **Context Scaling**
   - Throughput vs context length
   - Latency vs context length

4. **Optimal Configuration**
   - Best bit config for target budget
   - Recommended settings for production

5. **GPU Speedup**
   - Fused CUDA vs CPU streaming
   - SmartKV vs FP16 baseline
