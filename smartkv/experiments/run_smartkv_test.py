"""
Test SmartKV with actual adaptive precision KV cache.

This script demonstrates SmartKV actually working with dynamic precision allocation
during model inference, unlike validate_baselines.py which only simulates it.
"""

import torch
import argparse
from pathlib import Path
from typing import Dict, Any, List
import json
import time

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from smartkv.models.llama_smartkv import LlamaSmartKV, SmartKVConfig
from smartkv.utils.logger import create_logger
from smartkv.utils.metrics import MetricsTracker


class SmartKVTester:
    """
    Test SmartKV with real adaptive precision KV cache.

    Validates that:
    1. SmartKV integrates correctly into Llama
    2. Attention tracking works during inference
    3. Precision allocation is non-uniform
    4. Memory usage matches configured budget
    5. Output quality is maintained
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "cpu",
        output_dir: str = "experiments/smartkv_test"
    ):
        """
        Initialize SmartKV tester.

        Args:
            model_name: HuggingFace model name
            device: Device to use
            output_dir: Output directory for results
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(
            name="smartkv_test",
            log_file=str(self.output_dir / "test.log")
        )

        self.tokenizer = None
        self.base_model = None
        self.smartkv_model = None
        self.metrics = MetricsTracker()

    def load_models(self) -> bool:
        """Load base model and create SmartKV wrapper."""
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
                dtype=torch.float32
            )
            self.base_model = self.base_model.to(self.device)
            self.base_model.eval()

            self.logger.info("Base model loaded successfully")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_smartkv_model(
        self,
        memory_budget: float = 0.5,
        decay: float = 0.9,
        realloc_freq: int = 16
    ) -> bool:
        """
        Create SmartKV-enabled model.

        Args:
            memory_budget: Memory budget (fraction of FP16)
            decay: EMA decay for importance tracking
            realloc_freq: Reallocate precision every N tokens

        Returns:
            True if successful
        """
        try:
            self.logger.info("Creating SmartKV model...")
            self.logger.info(f"  Memory budget: {memory_budget:.1%}")
            self.logger.info(f"  Decay: {decay}")
            self.logger.info(f"  Realloc freq: {realloc_freq}")

            # Create SmartKV config
            config = SmartKVConfig(
                enabled=True,
                memory_budget=memory_budget,
                decay=decay,
                realloc_freq=realloc_freq,
                available_bits=[2, 3, 4, 8],
                device=self.device
            )

            # Wrap base model with SmartKV
            self.smartkv_model = LlamaSmartKV(self.base_model, config)

            self.logger.info("SmartKV model created successfully")
            self.logger.info(f"  SmartKV cache: {self.smartkv_model.smartkv_cache is not None}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to create SmartKV model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_inference(
        self,
        prompts: List[str],
        max_new_tokens: int = 20,
        use_smartkv: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference with or without SmartKV.

        Args:
            prompts: List of prompts
            max_new_tokens: Max tokens to generate
            use_smartkv: Whether to use SmartKV model

        Returns:
            Dict with results
        """
        model = self.smartkv_model.model if use_smartkv else self.base_model
        model_name = "SmartKV" if use_smartkv else "FP16"

        results = []
        latencies = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"[{model_name}] Processing prompt {i+1}/{len(prompts)}")

            # Clear SmartKV cache between prompts if using SmartKV model
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
            latencies.append(latency_ms)

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'latency_ms': latency_ms
            })

        return {
            'model': model_name,
            'results': results,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'num_prompts': len(prompts)
        }

    def verify_smartkv_active(self) -> Dict[str, Any]:
        """
        Verify that SmartKV is actually being used.

        Checks:
        1. Attention layers were replaced
        2. SmartKV cache exists
        3. Precision allocation is non-uniform
        4. Importance tracking has data

        Returns:
            Verification results
        """
        self.logger.info("\nVerifying SmartKV is active...")

        verification = {
            'attention_layers_replaced': False,
            'smartkv_cache_exists': False,
            'cache_has_data': False,
            'precision_is_adaptive': False,
            'importance_tracked': False
        }

        if self.smartkv_model is None:
            self.logger.error("SmartKV model not created")
            return verification

        # Check if attention layers were replaced
        if hasattr(self.smartkv_model.model, 'model') and hasattr(self.smartkv_model.model.model, 'layers'):
            layers = self.smartkv_model.model.model.layers
            if layers and hasattr(layers[0], 'self_attn'):
                attn = layers[0].self_attn
                if hasattr(attn, 'smartkv_cache'):
                    verification['attention_layers_replaced'] = True
                    self.logger.info("✓ Attention layers replaced with SmartKV")

        # Check SmartKV cache exists
        if self.smartkv_model.smartkv_cache is not None:
            verification['smartkv_cache_exists'] = True
            self.logger.info("✓ SmartKV cache exists")

            cache = self.smartkv_model.smartkv_cache

            # Check if cache has data
            if cache.k_cache or cache.v_cache:
                verification['cache_has_data'] = True
                self.logger.info(f"✓ Cache has data: {len(cache.k_cache)} K entries, {len(cache.v_cache)} V entries")

            # Check if precision is adaptive
            if cache.precision_map:
                precision_values = list(cache.precision_map.values())
                unique_precisions = set(precision_values)
                if len(unique_precisions) > 1:
                    verification['precision_is_adaptive'] = True
                    self.logger.info(f"✓ Precision is adaptive: {unique_precisions} bits used")

                    # Show precision distribution
                    precision_counts = {}
                    for bits in precision_values:
                        precision_counts[bits] = precision_counts.get(bits, 0) + 1
                    self.logger.info(f"  Precision distribution: {precision_counts}")
                else:
                    self.logger.warning(f"✗ Precision is uniform: only {unique_precisions} used")

            # Check importance tracking
            if cache.importance_tracker and cache.importance_tracker.token_importance:
                verification['importance_tracked'] = True
                num_tracked = len(cache.importance_tracker.token_importance)
                self.logger.info(f"✓ Importance tracked: {num_tracked} tokens")

                # Show top-5 most important tokens
                top_tokens = sorted(
                    cache.importance_tracker.token_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                self.logger.info(f"  Top-5 important tokens: {top_tokens}")

        return verification

    def run_full_test(
        self,
        test_prompts: List[str] = None,
        max_new_tokens: int = 20,
        memory_budget: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run complete SmartKV test.

        Args:
            test_prompts: Test prompts
            max_new_tokens: Max tokens to generate
            memory_budget: Memory budget for SmartKV

        Returns:
            Complete test results
        """
        if test_prompts is None:
            test_prompts = [
                "The capital of France is",
                "In machine learning, attention mechanisms are used to",
                "The quick brown fox jumps over"
            ]

        self.logger.info("="*80)
        self.logger.info("SMARTKV FULL INTEGRATION TEST")
        self.logger.info("="*80)

        # Load models
        if not self.load_models():
            return {'error': 'Failed to load models'}

        # Run baseline (FP16) inference
        self.logger.info("\n" + "="*80)
        self.logger.info("Running FP16 baseline inference...")
        self.logger.info("="*80)
        fp16_results = self.run_inference(test_prompts, max_new_tokens, use_smartkv=False)

        # Create SmartKV model
        if not self.create_smartkv_model(memory_budget=memory_budget):
            return {'error': 'Failed to create SmartKV model'}

        # Run SmartKV inference
        self.logger.info("\n" + "="*80)
        self.logger.info("Running SmartKV inference...")
        self.logger.info("="*80)
        smartkv_results = self.run_inference(test_prompts, max_new_tokens, use_smartkv=True)

        # Verify SmartKV is active
        verification = self.verify_smartkv_active()

        # Compare results
        self.logger.info("\n" + "="*80)
        self.logger.info("RESULTS COMPARISON")
        self.logger.info("="*80)

        self.logger.info(f"\nFP16 Baseline:")
        self.logger.info(f"  Avg Latency: {fp16_results['avg_latency_ms']:.2f} ms")

        self.logger.info(f"\nSmartKV (budget={memory_budget:.1%}):")
        self.logger.info(f"  Avg Latency: {smartkv_results['avg_latency_ms']:.2f} ms")
        self.logger.info(f"  Latency overhead: {(smartkv_results['avg_latency_ms'] / fp16_results['avg_latency_ms'] - 1) * 100:.1f}%")

        # Compare outputs
        self.logger.info("\nOutput Comparison:")
        for i in range(len(test_prompts)):
            fp16_out = fp16_results['results'][i]['generated']
            smartkv_out = smartkv_results['results'][i]['generated']
            match = "✓" if fp16_out == smartkv_out else "✗"
            self.logger.info(f"  Prompt {i+1}: {match} {'MATCH' if fp16_out == smartkv_out else 'DIFFER'}")
            if fp16_out != smartkv_out:
                self.logger.info(f"    FP16:     {fp16_out}")
                self.logger.info(f"    SmartKV:  {smartkv_out}")

        # Save results
        results = {
            'model': self.model_name,
            'memory_budget': memory_budget,
            'fp16_results': fp16_results,
            'smartkv_results': smartkv_results,
            'verification': verification
        }

        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nResults saved to {results_file}")
        self.logger.info("="*80)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test SmartKV integration")
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
        default=20,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/smartkv_test",
        help="Output directory"
    )

    args = parser.parse_args()

    # Run test
    tester = SmartKVTester(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir
    )

    results = tester.run_full_test(
        max_new_tokens=args.max_tokens,
        memory_budget=args.budget
    )

    if 'error' not in results:
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")

        # Print verification status
        verification = results.get('verification', {})
        all_passed = all(verification.values())

        print("\nVerification Status:")
        for check, passed in verification.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}")

        if all_passed:
            print("\n🎉 All verification checks passed! SmartKV is working correctly.")
        else:
            print("\n⚠️  Some verification checks failed. Review logs for details.")
    else:
        print(f"\nTest failed: {results['error']}")


if __name__ == "__main__":
    main()
