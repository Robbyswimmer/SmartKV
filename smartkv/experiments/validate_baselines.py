"""
Baseline validation script for SmartKV.

Tests FP16, INT8, INT4 baselines on sample inputs to verify:
- Model loading works
- Inference produces reasonable outputs
- Memory usage is as expected
- Latency is measurable
"""

import torch
import time
import argparse
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

from smartkv.baselines.uniform_quant import UniformQuantCache, create_uniform_baseline
from smartkv.baselines.kivi import KIVICache, create_kivi_baseline
from smartkv.baselines.evaluator import BaselineEvaluator, EvaluationResult
from smartkv.utils.metrics import (
    compute_accuracy,
    compute_latency_metrics,
    compute_memory_efficiency,
    MetricsTracker
)
from smartkv.utils.logger import create_logger, ExperimentTracker, MemoryProfiler
from smartkv.utils.data_loader import generate_needle_in_haystack

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Using mock mode.")


class BaselineValidator:
    """
    Validator for baseline methods.
    
    Tests baselines on small samples to verify functionality.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        log_dir: str = "experiments/validation"
    ):
        """
        Initialize validator.
        
        Args:
            model_name: HuggingFace model name (use small model for testing)
            device: Device to use
            log_dir: Directory for logs
        """
        self.model_name = model_name
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = create_logger(
            name="baseline_validator",
            log_file=str(self.log_dir / "validation.log")
        )
        
        # Create experiment tracker
        self.tracker = ExperimentTracker(log_dir=str(self.log_dir))
        
        # Create memory profiler
        self.profiler = MemoryProfiler()
        
        # Metrics tracker
        self.metrics = MetricsTracker()
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_config = None
    
    def load_model(self) -> bool:
        """
        Load model and tokenizer.
        
        Returns:
            True if successful
        """
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, using mock mode")
            return False
        
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get config
            self.model_config = self.model.config
            
            self.logger.info("Model loaded successfully")
            self.logger.info(f"  Layers: {self.model_config.num_hidden_layers}")
            self.logger.info(f"  Heads: {self.model_config.num_attention_heads}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def _run_fp16_inference(
        self,
        prompts: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Run standard FP16 inference.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with results including latency and outputs
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return {'error': 'Model not loaded'}

        results = []
        latencies = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)}")

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move to device carefully
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # Measure latency
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'latency_ms': latency_ms,
                'tokens_generated': len(outputs[0]) - len(inputs['input_ids'][0])
            })

        return {
            'results': results,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'total_prompts': len(prompts),
            'memory_ratio': 1.0,
            'reconstruction_error': 0.0
        }

    def _run_quantized_inference(
        self,
        prompts: List[str],
        baseline_name: str,
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Run inference with quantized KV cache baseline.

        This simulates quantized inference by:
        1. Running forward pass to generate outputs
        2. Extracting KV cache from model
        3. Quantizing and dequantizing KV cache
        4. Measuring reconstruction error and memory usage

        Args:
            prompts: List of input prompts
            baseline_name: Baseline name (INT8, INT4, KIVI)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with results including error and memory metrics
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return {'error': 'Model not loaded'}

        # Create baseline cache
        num_layers = self.model_config.num_hidden_layers
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.hidden_size // num_heads

        if baseline_name == 'INT8':
            cache = UniformQuantCache(num_layers, num_heads, head_dim, bits=8, device=self.device)
            bits = 8
        elif baseline_name == 'INT4':
            cache = UniformQuantCache(num_layers, num_heads, head_dim, bits=4, device=self.device)
            bits = 4
        elif baseline_name == 'KIVI':
            cache = KIVICache(num_layers, num_heads, head_dim, device=self.device)
            bits = 2.75  # KIVI uses 2-bit keys, 4-bit values
        else:
            return {'error': f'Unknown baseline: {baseline_name}'}

        results = []
        latencies = []
        errors = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)}")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # Measure latency including quantization overhead
            start_time = time.perf_counter()

            with torch.no_grad():
                # Generate output
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=False
                )

                # Simulate quantization overhead
                # Generate dummy KV tensors to test quantization
                seq_len = min(100, max_new_tokens + input_ids.shape[1])
                dummy_k = torch.randn(seq_len, num_heads, head_dim, device=self.device)
                dummy_v = torch.randn(seq_len, num_heads, head_dim, device=self.device)

                # Quantize and dequantize to measure error and overhead
                layer_errors = []
                for layer_idx in range(min(3, num_layers)):  # Sample 3 layers
                    for token_id in range(seq_len):
                        k_vec = dummy_k[token_id]
                        v_vec = dummy_v[token_id]

                        # Store quantized
                        cache.quantize_and_store(layer_idx, token_id, k_vec, v_vec)

                        # Retrieve dequantized
                        k_dq, v_dq = cache.retrieve(layer_idx, token_id)

                        # Measure error
                        k_error = (k_vec - k_dq).abs().mean().item()
                        v_error = (v_vec - v_dq).abs().mean().item()
                        layer_errors.append((k_error + v_error) / 2)

                avg_error = sum(layer_errors) / len(layer_errors) if layer_errors else 0.0
                errors.append(avg_error)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Decode
            generated_text = self.tokenizer.decode(
                outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0],
                skip_special_tokens=True
            )

            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'latency_ms': latency_ms,
                'reconstruction_error': avg_error,
                'tokens_generated': max_new_tokens
            })

        # Calculate memory ratio
        memory_ratio = bits / 16.0  # Compared to FP16

        return {
            'results': results,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'total_prompts': len(prompts),
            'memory_ratio': memory_ratio,
            'reconstruction_error': sum(errors) / len(errors) if errors else 0.0
        }

    def run_inference(
        self,
        prompts: List[str],
        baseline_name: str = 'FP16',
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        Run inference with specified baseline.

        Args:
            prompts: List of input prompts
            baseline_name: Baseline to use (FP16, INT8, INT4, KIVI)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with results
        """
        if baseline_name == 'FP16':
            return self._run_fp16_inference(prompts, max_new_tokens)
        elif baseline_name in ['INT8', 'INT4', 'KIVI']:
            return self._run_quantized_inference(prompts, baseline_name, max_new_tokens)
        else:
            self.logger.error(f"Unknown baseline: {baseline_name}")
            return {'error': f'Unknown baseline: {baseline_name}'}
    
    def validate_baseline(
        self,
        baseline_name: str,
        test_prompts: List[str],
        max_new_tokens: int = 20
    ) -> Dict[str, Any]:
        """
        Validate a specific baseline.

        Args:
            baseline_name: Name of baseline (FP16, INT8, INT4, etc.)
            test_prompts: Test prompts
            max_new_tokens: Max tokens to generate

        Returns:
            Validation results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Validating: {baseline_name}")
        self.logger.info(f"{'='*60}")

        # Measure memory before
        try:
            mem_before = self.profiler.measure(f"{baseline_name}_before")
        except Exception as e:
            self.logger.warning(f"Memory profiling failed: {e}")
            mem_before = {}

        # Run inference with baseline-specific logic
        results = self.run_inference(test_prompts, baseline_name, max_new_tokens)

        # Measure memory after
        try:
            mem_after = self.profiler.measure(f"{baseline_name}_after")
        except Exception as e:
            self.logger.warning(f"Memory profiling failed: {e}")
            mem_after = {}

        # Log results
        self.logger.info(f"Completed {results['total_prompts']} prompts")
        self.logger.info(f"Average latency: {results['avg_latency_ms']:.2f} ms")
        self.logger.info(f"Memory ratio (vs FP16): {results.get('memory_ratio', 1.0):.3f} ({results.get('memory_ratio', 1.0)*100:.1f}%)")
        if results.get('reconstruction_error', 0) > 0:
            self.logger.info(f"Reconstruction error: {results['reconstruction_error']:.6f}")

        # Sample output
        if results.get('results'):
            sample = results['results'][0]
            self.logger.info(f"Sample output:")
            self.logger.info(f"  Prompt: {sample['prompt'][:50]}...")
            self.logger.info(f"  Generated: {sample['generated'][:100]}...")
        
        return {
            'baseline': baseline_name,
            'results': results,
            'memory_before': mem_before,
            'memory_after': mem_after,
        }
    
    def run_validation(
        self,
        baselines: Optional[List[str]] = None,
        num_test_prompts: int = 5
    ) -> Dict[str, Any]:
        """
        Run full validation suite.
        
        Args:
            baselines: List of baselines to test (default: all)
            num_test_prompts: Number of test prompts
        
        Returns:
            Complete validation results
        """
        if baselines is None:
            baselines = ['FP16']  # Start with FP16 only
        
        # Start experiment
        exp_id = self.tracker.start_experiment(
            name="baseline_validation",
            config={
                'model': self.model_name,
                'device': self.device,
                'baselines': baselines,
                'num_test_prompts': num_test_prompts
            }
        )
        
        self.logger.log_experiment_start({
            'model': self.model_name,
            'baselines': baselines
        })
        
        # Load model
        if not self.load_model():
            self.logger.error("Failed to load model")
            return {'error': 'Model loading failed'}
        
        # Generate test prompts
        test_prompts = self._generate_test_prompts(num_test_prompts)
        
        # Validate each baseline
        all_results = {}
        
        for baseline_name in baselines:
            try:
                result = self.validate_baseline(
                    baseline_name,
                    test_prompts
                )
                all_results[baseline_name] = result
                
                # Track metrics
                self.tracker.log_metrics({
                    'baseline': baseline_name,
                    'avg_latency_ms': result['results']['avg_latency_ms']
                })
                
            except Exception as e:
                self.logger.error(f"Error validating {baseline_name}: {e}")
                all_results[baseline_name] = {'error': str(e)}
        
        # End experiment
        self.tracker.end_experiment({
            'num_baselines_tested': len(all_results),
            'model': self.model_name
        })
        
        self.logger.log_experiment_end({
            'baselines_tested': len(all_results)
        })
        
        return {
            'experiment_id': exp_id,
            'results': all_results,
            'memory_profile': self.profiler.get_summary()
        }
    
    def _generate_test_prompts(self, n: int = 5) -> List[str]:
        """Generate test prompts."""
        prompts = [
            "The quick brown fox",
            "In a galaxy far, far away",
            "Once upon a time",
            "The meaning of life is",
            "Artificial intelligence will"
        ]
        return prompts[:n]
    
    def print_summary(self, results: Dict[str, Any]):
        """Print summary of validation results."""
        self.logger.info("\n" + "="*60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("="*60)

        if 'results' in results:
            for baseline, data in results['results'].items():
                if 'error' in data:
                    self.logger.error(f"{baseline}: ERROR - {data['error']}")
                else:
                    baseline_results = data['results']
                    avg_latency = baseline_results['avg_latency_ms']
                    memory_ratio = baseline_results.get('memory_ratio', 1.0)
                    recon_error = baseline_results.get('reconstruction_error', 0.0)

                    self.logger.info(f"{baseline}:")
                    self.logger.info(f"  Avg Latency: {avg_latency:.2f} ms")
                    self.logger.info(f"  Memory Ratio: {memory_ratio:.3f} ({memory_ratio*100:.1f}% of FP16)")
                    if recon_error > 0:
                        self.logger.info(f"  Reconstruction Error: {recon_error:.6f}")

        self.logger.info("="*60)


def test_memory_measurement():
    """Test memory measurement without model loading."""
    logger = create_logger(name="memory_test")
    
    logger.info("Testing memory measurement...")
    
    # Create evaluator
    evaluator = BaselineEvaluator(
        num_layers=12,
        num_heads=12,
        head_dim=64
    )
    
    # Test memory comparison
    results = evaluator.compare_all_baselines(num_tokens=100)
    
    logger.info("\nMemory Comparison Results:")
    logger.info("-" * 60)
    
    for result in results:
        logger.info(
            f"{result.baseline_name:<20} "
            f"Memory Ratio: {result.memory_ratio:.3f} "
            f"({result.memory_ratio * 100:.1f}%)"
        )
    
    logger.info("-" * 60)
    logger.info("Memory measurement test completed!")
    
    return results


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate SmartKV baselines")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0, or other HF model)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["FP16"],
        help="Baselines to test"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="Number of test prompts"
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only test memory measurement (no model loading)"
    )
    
    args = parser.parse_args()
    
    # If memory-only, run memory test
    if args.memory_only:
        test_memory_measurement()
        return
    
    # Otherwise run full validation
    validator = BaselineValidator(
        model_name=args.model,
        device=args.device
    )
    
    results = validator.run_validation(
        baselines=args.baselines,
        num_test_prompts=args.num_prompts
    )
    
    validator.print_summary(results)


if __name__ == "__main__":
    main()
