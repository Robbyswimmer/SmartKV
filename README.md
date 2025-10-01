# SmartKV: Attention-Guided Adaptive Precision for KV-Cache Compression

SmartKV is a research implementation exploring KV-cache compression through attention-guided adaptive precision allocation. The system dynamically assigns bit-widths (2, 3, 4, or 8 bits) to cached key-value pairs based on their importance in attention patterns, aiming to achieve better accuracy-memory tradeoffs than uniform quantization approaches.

## Core Concept

While uniform quantization methods apply the same bit-width to all tokens in the KV-cache, SmartKV allocates precision adaptively based on attention patterns. Tokens that receive higher attention scores are assigned higher precision, while less important tokens are compressed more aggressively. This approach is motivated by the observation that attention patterns are often sparse, with models focusing on a small subset of critical tokens.

## Algorithm Overview

SmartKV operates in three phases:

1. **Importance Tracking**: Maintains exponential moving average of attention scores for each token across layers
2. **Precision Allocation**: Every N tokens, performs greedy allocation of available bit-widths to maximize importance coverage within a memory budget
3. **Quantization**: Applies per-tensor symmetric quantization with allocated precision levels

Key parameters:
- `memory_budget`: Target memory usage as fraction of FP16 baseline (e.g., 0.35 = 35%)
- `decay`: EMA decay factor for importance tracking (default: 0.9)
- `realloc_freq`: Frequency of precision reallocation in tokens (default: 16)
- `available_bits`: Supported quantization levels (default: [2, 3, 4, 8])

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SmartKV.git
cd SmartKV

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Current Implementation Status

**Completed:**
- Core quantization infrastructure (2/3/4/8-bit symmetric quantization)
- Importance tracking system with EMA-based scoring
- Greedy precision allocation algorithm
- Llama model integration (tested with Llama 3.1 8B, TinyLlama 1.1B)
- Basic evaluation pipeline with latency and output quality metrics

**In Progress:**
- Comprehensive evaluation against uniform quantization baselines
- Perplexity-based quality assessment

**Planned:**
- GPU optimization (current implementation is CPU-only)
- Additional model architectures (Mistral, GPT-NeoX)
- Integration with standard benchmarks (LongBench, RULER)

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig
import torch

# Load model
model_name = "NousResearch/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure SmartKV
config = SmartKVConfig(
    enabled=True,
    memory_budget=0.35,  # 35% of FP16 memory
    decay=0.9,
    realloc_freq=16,
    available_bits=[2, 3, 4, 8],
    device="cpu"
)

# Wrap model with SmartKV
smartkv_model = LlamaSmartKV(model, config)

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Running Evaluations

```bash
# Quick compatibility test (3 prompts)
python -m smartkv.experiments.run_smartkv_test \
  --model "NousResearch/Meta-Llama-3.1-8B" \
  --max-tokens 20 \
  --output-dir experiments/quick_test

# Comprehensive evaluation (30 prompts across 6 categories)
python -m smartkv.experiments.comprehensive_evaluation \
  --model "NousResearch/Meta-Llama-3.1-8B" \
  --max-tokens 50 \
  --budget 0.35 \
  --output-dir experiments/comprehensive_eval

# Analyze results
python -m smartkv.experiments.analyze_comprehensive_results \
  experiments/comprehensive_eval/comprehensive_results.json
```

## Preliminary Results

**Llama 3.1 8B (CPU, 3 prompts, 20 tokens each):**
- Memory budget: 50% of FP16
- Latency: 20.3% faster than FP16 baseline (21.8s vs 27.3s)
- Precision distribution: 25 tokens at 8-bit, 1 token at 2-bit
- Output quality: Qualitatively similar to baseline

**Note:** Performance improvements on CPU are primarily due to memory bandwidth savings. GPU performance characteristics may differ significantly. Current results compare against FP16 baseline; comparison against uniform INT8/INT4 quantization baselines is ongoing.

## Project Structure

```
SmartKV/
├── smartkv/
│   ├── core/
│   │   ├── quantizers.py      # Quantization implementations
│   │   ├── importance.py      # Attention importance tracking
│   │   └── allocation.py      # Precision allocation logic
│   ├── models/
│   │   ├── llama_smartkv.py   # Llama integration
│   │   └── attention.py       # Modified attention layers
│   └── experiments/
│       ├── run_smartkv_test.py
│       ├── comprehensive_evaluation.py
│       └── analyze_comprehensive_results.py
├── experiments/               # Experiment outputs
├── requirements.txt
└── README.md
```

## Known Limitations

1. **CPU-only implementation**: Current version is not optimized for GPU inference
2. **Limited baseline comparisons**: Needs evaluation against uniform INT8/INT4 quantization
3. **No perplexity metrics**: Current evaluation uses output similarity; perplexity-based evaluation needed
4. **Single architecture**: Only tested with Llama-family models
5. **No custom CUDA kernels**: Would be required for competitive GPU performance

## Research Context

This implementation is designed for research exploration of attention-guided quantization strategies. For production use cases, consider:
- Established uniform quantization methods (INT8/INT4) with hardware acceleration
- Methods with optimized GPU kernels (e.g., GPTQ, AWQ for weights; uniform quantization for KV-cache)
- Trade-off between adaptive allocation complexity and potential accuracy gains

## Testing

```bash
# Run tests (when available)
pytest tests/

# Run specific test module
pytest tests/test_quantizers.py
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{smartkv2024,
  title={SmartKV: Attention-Guided Adaptive Precision for KV-Cache Compression},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/SmartKV}
}
```

## Acknowledgments

Built with PyTorch and Hugging Face Transformers. Inspired by research in KV-cache compression including KIVI, GEAR, and related work on adaptive quantization.
