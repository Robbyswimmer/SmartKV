"""
Comprehensive evaluation of SmartKV across diverse prompts and tasks.

Tests SmartKV performance on:
- Factual knowledge
- Reasoning
- Code generation
- Creative writing
- Conversational responses
- Question answering
"""

import torch
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import time
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig
from smartkv.utils.logger import create_logger
from smartkv.utils.metrics import MetricsTracker


# Comprehensive test prompts across different domains
TEST_PROMPTS = {
    "factual_knowledge": [
        "The capital of France is",
        "The first person to walk on the moon was",
        "The largest ocean on Earth is",
        "Python is a programming language created by",
        "The speed of light is approximately",
    ],
    "reasoning": [
        "If all cats are animals, and some animals are pets, then",
        "To solve 2x + 5 = 15, first we",
        "The main difference between a virus and a bacteria is",
        "If it takes 5 machines 5 minutes to make 5 widgets, then 100 machines would take",
        "The reason seasons occur on Earth is because",
    ],
    "code": [
        "def fibonacci(n):\n    # Calculate the nth Fibonacci number",
        "# Function to reverse a string in Python\ndef reverse_string(s):",
        "Here's a Python function to check if a number is prime:",
        "To sort a list in Python, you can use",
        "A simple HTTP server in Python can be created using",
    ],
    "creative": [
        "Once upon a time, in a land far away,",
        "The old mansion on the hill was known for",
        "She opened the mysterious letter and read:",
        "In the year 2150, humanity had finally",
        "The robot looked at its creator and said,",
    ],
    "conversation": [
        "Q: How are you doing today?\nA:",
        "Q: What's the weather like?\nA:",
        "Q: Can you help me with my homework?\nA:",
        "Q: What's your favorite color?\nA:",
        "Q: Tell me a joke.\nA:",
    ],
    "completion": [
        "In machine learning, attention mechanisms are used to",
        "The quick brown fox jumps over",
        "To be or not to be,",
        "Four score and seven years ago,",
        "When in the course of human events,",
    ],
}


class ComprehensiveEvaluator:
    """Run comprehensive evaluation of SmartKV."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "cpu",
        output_dir: str = "experiments/comprehensive_eval"
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(
            name="comprehensive_eval",
            log_file=str(self.output_dir / "eval.log")
        )

        self.tokenizer = None
        self.base_model = None
        self.smartkv_model = None
        self.metrics = MetricsTracker()

    def load_models(self) -> bool:
        """Load models."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("transformers not available")
            return False

        try:
            self.logger.info(f"Loading model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
            self.base_model = self.base_model.to(self.device)
            self.base_model.eval()

            self.logger.info("Models loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_smartkv_model(
        self,
        memory_budget: float = 0.5,
        decay: float = 0.9,
        realloc_freq: int = 16
    ) -> bool:
        """Create SmartKV model."""
        try:
            self.logger.info("Creating SmartKV model...")

            config = SmartKVConfig(
                enabled=True,
                memory_budget=memory_budget,
                decay=decay,
                realloc_freq=realloc_freq,
                available_bits=[2, 3, 4, 8],
                device=self.device
            )

            self.smartkv_model = LlamaSmartKV(self.base_model, config)
            self.logger.info("SmartKV model created successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create SmartKV model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_inference(
        self,
        prompt: str,
        max_new_tokens: int,
        use_smartkv: bool
    ) -> Dict[str, Any]:
        """Run inference on a single prompt."""
        model = self.smartkv_model.model if use_smartkv else self.base_model

        # Clear cache if using SmartKV
        if use_smartkv and self.smartkv_model is not None:
            self.smartkv_model.reset_cache()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Measure latency
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            'prompt': prompt,
            'generated': generated_text,
            'latency_ms': latency_ms,
            'num_tokens': len(outputs[0]) - len(inputs['input_ids'][0])
        }

    def evaluate_category(
        self,
        category: str,
        prompts: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """Evaluate on a category of prompts."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Evaluating category: {category.upper()}")
        self.logger.info(f"{'='*80}")

        fp16_results = []
        smartkv_results = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"[{category}] Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

            # FP16 baseline
            fp16_result = self.run_inference(prompt, max_new_tokens, use_smartkv=False)
            fp16_results.append(fp16_result)

            # SmartKV
            smartkv_result = self.run_inference(prompt, max_new_tokens, use_smartkv=True)
            smartkv_results.append(smartkv_result)

        # Compute statistics
        fp16_latencies = [r['latency_ms'] for r in fp16_results]
        smartkv_latencies = [r['latency_ms'] for r in smartkv_results]

        return {
            'category': category,
            'num_prompts': len(prompts),
            'fp16_results': fp16_results,
            'smartkv_results': smartkv_results,
            'fp16_avg_latency': np.mean(fp16_latencies),
            'smartkv_avg_latency': np.mean(smartkv_latencies),
            'latency_overhead_pct': (np.mean(smartkv_latencies) / np.mean(fp16_latencies) - 1) * 100,
        }

    def compare_outputs(self, fp16_text: str, smartkv_text: str, prompt: str) -> Dict[str, Any]:
        """Compare FP16 and SmartKV outputs."""
        # Strip prompt from generated text
        fp16_gen = fp16_text[len(prompt):].strip()
        smartkv_gen = smartkv_text[len(prompt):].strip()

        # Exact match
        exact_match = fp16_gen == smartkv_gen

        # Token-level similarity (simple word overlap)
        fp16_words = set(fp16_gen.lower().split())
        smartkv_words = set(smartkv_gen.lower().split())

        if fp16_words or smartkv_words:
            word_overlap = len(fp16_words & smartkv_words) / len(fp16_words | smartkv_words)
        else:
            word_overlap = 1.0

        # Prefix match (how many starting words are the same)
        fp16_tokens = fp16_gen.split()
        smartkv_tokens = smartkv_gen.split()

        prefix_match_len = 0
        for w1, w2 in zip(fp16_tokens, smartkv_tokens):
            if w1 == w2:
                prefix_match_len += 1
            else:
                break

        return {
            'exact_match': exact_match,
            'word_overlap': word_overlap,
            'prefix_match_tokens': prefix_match_len,
            'fp16_length': len(fp16_tokens),
            'smartkv_length': len(smartkv_tokens),
        }

    def run_comprehensive_evaluation(
        self,
        max_new_tokens: int = 50,
        memory_budget: float = 0.5,
        categories: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        from smartkv.experiments._thread_utils import configure_threads
        threads = configure_threads()
        self.logger.info(f"Configured inference threads: {threads}")
        self.logger.info("="*80)
        self.logger.info("COMPREHENSIVE SMARTKV EVALUATION")
        self.logger.info("="*80)

        # Load models
        if not self.load_models():
            return {'error': 'Failed to load models'}

        # Create SmartKV model
        if not self.create_smartkv_model(memory_budget=memory_budget):
            return {'error': 'Failed to create SmartKV model'}

        # Select categories to evaluate
        if categories is None:
            categories = list(TEST_PROMPTS.keys())

        # Run evaluation on each category
        category_results = {}
        all_comparisons = []

        for category in categories:
            if category not in TEST_PROMPTS:
                self.logger.warning(f"Unknown category: {category}")
                continue

            prompts = TEST_PROMPTS[category]
            result = self.evaluate_category(category, prompts, max_new_tokens)
            category_results[category] = result

            # Compare outputs
            for fp16_res, smartkv_res in zip(result['fp16_results'], result['smartkv_results']):
                comparison = self.compare_outputs(
                    fp16_res['generated'],
                    smartkv_res['generated'],
                    fp16_res['prompt']
                )
                comparison['category'] = category
                comparison['prompt'] = fp16_res['prompt']
                all_comparisons.append(comparison)

        # Aggregate statistics
        all_fp16_latencies = []
        all_smartkv_latencies = []

        for cat_result in category_results.values():
            all_fp16_latencies.extend([r['latency_ms'] for r in cat_result['fp16_results']])
            all_smartkv_latencies.extend([r['latency_ms'] for r in cat_result['smartkv_results']])

        # Overall quality metrics
        exact_matches = sum(1 for c in all_comparisons if c['exact_match'])
        avg_word_overlap = np.mean([c['word_overlap'] for c in all_comparisons])
        avg_prefix_match = np.mean([c['prefix_match_tokens'] for c in all_comparisons])

        results = {
            'model': self.model_name,
            'memory_budget': memory_budget,
            'max_new_tokens': max_new_tokens,
            'total_prompts': len(all_comparisons),
            'categories_tested': list(category_results.keys()),
            'category_results': category_results,
            'comparisons': all_comparisons,
            'overall_stats': {
                'fp16_avg_latency_ms': np.mean(all_fp16_latencies),
                'smartkv_avg_latency_ms': np.mean(all_smartkv_latencies),
                'latency_overhead_pct': (np.mean(all_smartkv_latencies) / np.mean(all_fp16_latencies) - 1) * 100,
                'exact_match_rate': exact_matches / len(all_comparisons),
                'avg_word_overlap': avg_word_overlap,
                'avg_prefix_match_tokens': avg_prefix_match,
            }
        }

        # Save results
        results_file = self.output_dir / "comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nResults saved to {results_file}")

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        stats = results['overall_stats']

        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*80)

        self.logger.info(f"\nTotal Prompts: {results['total_prompts']}")
        self.logger.info(f"Categories: {', '.join(results['categories_tested'])}")
        self.logger.info(f"Memory Budget: {results['memory_budget']:.1%}")

        self.logger.info("\n--- Performance ---")
        self.logger.info(f"FP16 Avg Latency:     {stats['fp16_avg_latency_ms']:.2f} ms")
        self.logger.info(f"SmartKV Avg Latency:  {stats['smartkv_avg_latency_ms']:.2f} ms")
        self.logger.info(f"Latency Overhead:     {stats['latency_overhead_pct']:+.1f}%")

        self.logger.info("\n--- Quality ---")
        self.logger.info(f"Exact Match Rate:     {stats['exact_match_rate']:.1%}")
        self.logger.info(f"Avg Word Overlap:     {stats['avg_word_overlap']:.1%}")
        self.logger.info(f"Avg Prefix Match:     {stats['avg_prefix_match_tokens']:.1f} tokens")

        self.logger.info("\n--- Per-Category Performance ---")
        for category, cat_result in results['category_results'].items():
            self.logger.info(f"\n{category.upper()}:")
            self.logger.info(f"  Prompts: {cat_result['num_prompts']}")
            self.logger.info(f"  FP16 Latency:     {cat_result['fp16_avg_latency']:.2f} ms")
            self.logger.info(f"  SmartKV Latency:  {cat_result['smartkv_avg_latency']:.2f} ms")
            self.logger.info(f"  Overhead:         {cat_result['latency_overhead_pct']:+.1f}%")

        self.logger.info("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive SmartKV evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.5,
        help="Memory budget (fraction of FP16)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to test (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/comprehensive_eval",
        help="Output directory"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = ComprehensiveEvaluator(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir
    )

    results = evaluator.run_comprehensive_evaluation(
        max_new_tokens=args.max_tokens,
        memory_budget=args.budget,
        categories=args.categories
    )

    if 'error' not in results:
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {args.output_dir}/comprehensive_results.json")

        stats = results['overall_stats']
        print(f"\n📊 Key Metrics:")
        print(f"  Total Prompts:      {results['total_prompts']}")
        print(f"  Exact Match Rate:   {stats['exact_match_rate']:.1%}")
        print(f"  Word Overlap:       {stats['avg_word_overlap']:.1%}")
        print(f"  Latency Overhead:   {stats['latency_overhead_pct']:+.1f}%")
    else:
        print(f"\n❌ Evaluation failed: {results['error']}")


if __name__ == "__main__":
    main()
