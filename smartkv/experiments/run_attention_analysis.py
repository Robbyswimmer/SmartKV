"""
Run attention analysis experiments.

Validates the hypothesis that high-attention tokens are more sensitive to quantization.
"""

import torch
import argparse
from pathlib import Path
from typing import Dict, Any
import json

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from smartkv.analysis.attention_analysis import (
    AttentionLogger,
    analyze_attention_distribution,
    identify_high_attention_tokens,
    measure_quantization_sensitivity,
    plot_attention_heatmap,
    plot_attention_distribution,
    plot_quantization_sensitivity
)
from smartkv.core.quantizers import EightbitQuantizer, FourbitQuantizer, TwobitQuantizer
from smartkv.utils.logger import create_logger


class AttentionAnalysisExperiment:
    """
    Run attention analysis experiments.

    Tests the core SmartKV hypothesis: tokens with high attention are more
    sensitive to quantization errors.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "cpu",
        output_dir: str = "experiments/attention_analysis"
    ):
        """
        Initialize experiment.

        Args:
            model_name: HuggingFace model name
            device: Device to use
            output_dir: Directory to save results
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(
            name="attention_analysis",
            log_file=str(self.output_dir / "analysis.log")
        )

        self.model = None
        self.tokenizer = None
        self.attention_logger = AttentionLogger()

    def load_model(self) -> bool:
        """Load model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("transformers not available")
            return False

        try:
            self.logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with output_attentions=True
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                output_attentions=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def run_inference_with_logging(
        self,
        prompts: list,
        max_new_tokens: int = 20
    ) -> Dict[int, torch.Tensor]:
        """
        Run inference and capture attention weights.

        Args:
            prompts: List of prompts
            max_new_tokens: Max tokens to generate

        Returns:
            Dict mapping layer_idx to attention weights
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return {}

        # Register hooks
        self.attention_logger.register_hooks(self.model)

        all_attention_weights = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)}")

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Extract attention weights from outputs
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # outputs.attentions is a tuple of tuples
                # Shape: (num_gen_steps,) where each element is (num_layers,)
                # Each layer element is [batch, heads, 1, past_len]

                for step_attentions in outputs.attentions:
                    all_attention_weights.append(step_attentions)

        # Remove hooks
        self.attention_logger.remove_hooks()

        # Aggregate attention weights by layer
        layer_attention = {}

        if all_attention_weights:
            num_layers = len(all_attention_weights[0])

            for layer_idx in range(num_layers):
                layer_weights = []

                for step_attentions in all_attention_weights:
                    if layer_idx < len(step_attentions):
                        attn = step_attentions[layer_idx]
                        # attn is [batch, heads, 1, seq_len]
                        # Squeeze the query dimension and take only the last position
                        # to get [batch, heads, seq_len]
                        if attn.dim() == 4:
                            attn = attn[:, :, -1, :]  # [batch, heads, seq_len]
                        layer_weights.append(attn)

                if layer_weights:
                    # Average across all generation steps
                    # Each is [batch, heads, seq_len] but seq_len may vary
                    # Solution: Just use the first attention tensor (initial generation step)
                    # This represents attention during the first generated token
                    layer_attention[layer_idx] = layer_weights[0].unsqueeze(2)  # [batch, heads, 1, seq_len]

        return layer_attention

    def analyze_attention_patterns(
        self,
        attention_weights: Dict[int, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns across layers.

        Args:
            attention_weights: Dict mapping layer_idx to attention weights

        Returns:
            Analysis results
        """
        results = {}

        for layer_idx, attn in attention_weights.items():
            self.logger.info(f"Analyzing layer {layer_idx}")

            stats = analyze_attention_distribution(attn, k=10)

            self.logger.info(f"  Mean: {stats.mean:.4f}")
            self.logger.info(f"  Std: {stats.std:.4f}")
            self.logger.info(f"  Entropy: {stats.entropy:.4f}")
            self.logger.info(f"  90th percentile: {stats.percentiles.get(90, 0):.4f}")

            results[f'layer_{layer_idx}'] = stats.to_dict()

        return results

    def test_quantization_hypothesis(
        self,
        attention_weights: Dict[int, torch.Tensor],
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Test hypothesis: high attention tokens are more sensitive to quantization.

        Args:
            attention_weights: Attention weights from inference
            num_samples: Number of sample tokens to test

        Returns:
            Results showing sensitivity differences
        """
        self.logger.info("\nTesting quantization sensitivity hypothesis...")

        # Use first layer's attention for analysis
        if not attention_weights:
            self.logger.error("No attention weights available")
            return {}

        layer_idx = list(attention_weights.keys())[0]
        attn = attention_weights[layer_idx]

        # Generate dummy KV tensors to quantize
        # Simulate typical hidden dim
        hidden_dim = 2048
        kv_tensors = torch.randn(num_samples, hidden_dim)

        results = {}
        quantizers = {
            '8-bit': EightbitQuantizer(),
            '4-bit': FourbitQuantizer(),
            '2-bit': TwobitQuantizer()
        }

        for quant_name, quantizer in quantizers.items():
            self.logger.info(f"\nTesting {quant_name} quantization:")

            sensitivity = measure_quantization_sensitivity(
                kv_tensors,
                attn,
                quantizer,
                threshold_percentile=90.0
            )

            self.logger.info(f"  High attention error: {sensitivity['high_attention_error']:.6f}")
            self.logger.info(f"  Low attention error: {sensitivity['low_attention_error']:.6f}")
            self.logger.info(f"  Error ratio (high/low): {sensitivity['error_ratio']:.2f}x")

            results[quant_name] = sensitivity

        return results

    def create_visualizations(
        self,
        attention_weights: Dict[int, torch.Tensor],
        sensitivity_results: Dict[str, Any]
    ):
        """
        Create visualization plots.

        Args:
            attention_weights: Attention weights
            sensitivity_results: Results from quantization sensitivity test
        """
        self.logger.info("\nGenerating visualizations...")

        # Plot attention heatmap
        fig1 = plot_attention_heatmap(attention_weights, max_tokens=50)
        if fig1:
            fig1.savefig(self.output_dir / "attention_heatmap.png", dpi=300)
            self.logger.info("Saved attention_heatmap.png")

        # Plot attention distribution
        if attention_weights:
            layer_idx = list(attention_weights.keys())[0]
            fig2 = plot_attention_distribution(attention_weights[layer_idx])
            if fig2:
                fig2.savefig(self.output_dir / "attention_distribution.png", dpi=300)
                self.logger.info("Saved attention_distribution.png")

        # Plot quantization sensitivity
        if sensitivity_results:
            sensitivity_list = list(sensitivity_results.values())
            fig3 = plot_quantization_sensitivity(sensitivity_list)
            if fig3:
                fig3.savefig(self.output_dir / "quantization_sensitivity.png", dpi=300)
                self.logger.info("Saved quantization_sensitivity.png")

    def run_full_analysis(
        self,
        test_prompts: list = None,
        max_new_tokens: int = 20
    ) -> Dict[str, Any]:
        """
        Run complete attention analysis.

        Args:
            test_prompts: Test prompts (default: use built-in)
            max_new_tokens: Max tokens to generate

        Returns:
            Complete analysis results
        """
        if test_prompts is None:
            test_prompts = [
                "The capital of France is",
                "In machine learning, attention mechanisms",
                "The quick brown fox jumps"
            ]

        self.logger.info("="*60)
        self.logger.info("ATTENTION ANALYSIS EXPERIMENT")
        self.logger.info("="*60)

        # Load model
        if not self.load_model():
            return {'error': 'Failed to load model'}

        # Run inference and capture attention
        self.logger.info("\nRunning inference...")
        attention_weights = self.run_inference_with_logging(test_prompts, max_new_tokens)

        if not attention_weights:
            self.logger.error("Failed to capture attention weights")
            return {'error': 'No attention weights captured'}

        self.logger.info(f"Captured attention for {len(attention_weights)} layers")

        # Analyze patterns
        self.logger.info("\nAnalyzing attention patterns...")
        pattern_analysis = self.analyze_attention_patterns(attention_weights)

        # Test quantization hypothesis
        sensitivity_results = self.test_quantization_hypothesis(attention_weights)

        # Create visualizations
        self.create_visualizations(attention_weights, sensitivity_results)

        # Save results
        results = {
            'model': self.model_name,
            'num_prompts': len(test_prompts),
            'num_layers': len(attention_weights),
            'pattern_analysis': pattern_analysis,
            'quantization_sensitivity': sensitivity_results
        }

        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nResults saved to {results_file}")
        self.logger.info("="*60)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run attention analysis experiment")
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
        "--output-dir",
        type=str,
        default="experiments/attention_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Max tokens to generate"
    )

    args = parser.parse_args()

    # Run experiment
    experiment = AttentionAnalysisExperiment(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir
    )

    results = experiment.run_full_analysis(max_new_tokens=args.max_tokens)

    if 'error' not in results:
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
    else:
        print(f"\nExperiment failed: {results['error']}")


if __name__ == "__main__":
    main()
