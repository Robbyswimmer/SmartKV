# SmartKV Experiments

This directory contains experiment scripts for validating and evaluating SmartKV.

## Baseline Validation

### Quick Test (No Model Loading)

Test memory measurement and baseline functionality without loading large models:

```bash
python scripts/test_validation.py
```

This will:
- ✓ Test all baseline memory measurements (FP16, INT8, INT4, KIVI)
- ✓ Verify memory ratios are correct
- ✓ Test reconstruction error
- ✓ Benchmark latency

### Full Validation (With Model)

Run inference with a real model (uses GPT-2 by default for speed):

```bash
# Test with GPT-2 (small, fast)
python -m smartkv.experiments.validate_baselines --model gpt2 --num-prompts 3

# Test with larger model (requires more resources)
python -m smartkv.experiments.validate_baselines --model gpt2-medium --num-prompts 5

# Test specific baselines
python -m smartkv.experiments.validate_baselines --baselines FP16 INT8 INT4
```

### Memory-Only Mode

Test memory measurement without model loading:

```bash
python -m smartkv.experiments.validate_baselines --memory-only
```

## Expected Results

### Memory Ratios
- **FP16**: 1.00x (baseline, no compression)
- **Uniform-INT8**: 0.50x (50% of FP16)
- **Uniform-INT4**: 0.25x (25% of FP16)
- **KIVI**: 0.188x (18.8% of FP16, K=2bit V=4bit)
- **SmartKV**: 0.40-0.60x (adaptive, 40-60% of FP16)

### Reconstruction Error
- FP16 < INT8 < INT4 (lower is better)
- SmartKV should have error between INT8 and INT4

## Options

```
--model MODEL          Model name (default: gpt2)
--device DEVICE        Device to use (cpu/cuda)
--baselines NAMES      Baselines to test (space-separated)
--num-prompts N        Number of test prompts
--memory-only          Only test memory (no model loading)
```

## Experiment Logs

All experiments are logged to:
- Console: Colored output
- File: `experiments/validation/validation.log`
- JSON: `experiments/validation/{experiment_id}.json`

## Next Steps

After validation passes:
1. Proceed to Phase 11: Attention Analysis
2. Run full evaluation on LongBench (Phase 13)
3. Compare SmartKV against all baselines
