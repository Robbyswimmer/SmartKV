"""
Evaluation wrapper for all baseline methods.

Provides unified interface for evaluating SmartKV against baselines:
- FP16 (no quantization)
- Uniform INT8
- Uniform INT4
- KIVI
- SmartKV (our method)
"""

import torch
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time

from smartkv.baselines.uniform_quant import (
    UniformQuantCache,
    UniformQuantAttention,
    create_uniform_baseline
)
from smartkv.baselines.kivi import (
    KIVICache,
    KIVIAttention,
    create_kivi_baseline
)
from smartkv.core.cache import SmartKVCache
from smartkv.models.attention import SmartKVAttention


@dataclass
class EvaluationResult:
    """Results from baseline evaluation."""
    
    baseline_name: str
    memory_ratio: float  # Memory usage compared to FP16
    avg_bits: Optional[float] = None  # Average bits (for SmartKV)
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'baseline': self.baseline_name,
            'memory_ratio': self.memory_ratio,
            'avg_bits': self.avg_bits,
            'accuracy': self.accuracy,
            'perplexity': self.perplexity,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput_tokens_per_sec,
            'cache_stats': self.cache_stats,
        }


class BaselineEvaluator:
    """
    Unified evaluator for all baseline methods.
    
    Provides consistent interface for comparing different quantization approaches.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: str = "cpu"
    ):
        """
        Initialize baseline evaluator.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            device: Device to use
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Store baseline configurations
        self.baselines: Dict[str, Dict[str, Any]] = {}
        
        # Register default baselines
        self._register_default_baselines()
    
    def _register_default_baselines(self):
        """Register default baseline configurations."""
        # FP16 baseline
        self.register_baseline(create_uniform_baseline(bits=16, name="FP16"))
        
        # Uniform INT8
        self.register_baseline(create_uniform_baseline(bits=8, name="Uniform-INT8"))
        
        # Uniform INT4
        self.register_baseline(create_uniform_baseline(bits=4, name="Uniform-INT4"))
        
        # KIVI
        self.register_baseline(create_kivi_baseline(key_bits=2, value_bits=4, name="KIVI"))
    
    def register_baseline(self, baseline_config: Dict[str, Any]):
        """
        Register a new baseline.
        
        Args:
            baseline_config: Baseline configuration dict
        """
        name = baseline_config['name']
        self.baselines[name] = baseline_config
    
    def register_smartkv(
        self,
        memory_budget: float = 0.5,
        decay: float = 0.9,
        realloc_freq: int = 16,
        available_bits: List[int] = [2, 3, 4, 8]
    ):
        """
        Register SmartKV as a baseline.
        
        Args:
            memory_budget: Memory budget for SmartKV
            decay: EMA decay
            realloc_freq: Reallocation frequency
            available_bits: Available bit-widths
        """
        self.baselines['SmartKV'] = {
            'name': 'SmartKV',
            'type': 'smartkv',
            'cache_class': SmartKVCache,
            'attention_class': SmartKVAttention,
            'config': {
                'memory_budget': memory_budget,
                'decay': decay,
                'realloc_freq': realloc_freq,
                'available_bits': available_bits,
            }
        }
    
    def create_cache(self, baseline_name: str) -> Any:
        """
        Create cache for a specific baseline.
        
        Args:
            baseline_name: Name of the baseline
        
        Returns:
            Cache instance
        """
        if baseline_name not in self.baselines:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        
        baseline = self.baselines[baseline_name]
        cache_class = baseline['cache_class']
        
        if baseline['type'] == 'uniform':
            return cache_class(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                bits=baseline['bits'],
                device=self.device
            )
        elif baseline['type'] == 'kivi':
            return cache_class(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                key_bits=baseline['key_bits'],
                value_bits=baseline['value_bits'],
                device=self.device
            )
        elif baseline['type'] == 'smartkv':
            config = baseline['config']
            return cache_class(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                memory_budget=config['memory_budget'],
                decay=config['decay'],
                realloc_freq=config['realloc_freq'],
                available_bits=config['available_bits'],
                device=self.device
            )
        else:
            raise ValueError(f"Unknown baseline type: {baseline['type']}")
    
    def evaluate_memory(
        self,
        baseline_name: str,
        num_tokens: int
    ) -> EvaluationResult:
        """
        Evaluate memory usage for a baseline.
        
        Args:
            baseline_name: Name of the baseline
            num_tokens: Number of tokens to simulate
        
        Returns:
            Evaluation result
        """
        cache = self.create_cache(baseline_name)
        
        # Simulate storing tokens
        for layer_idx in range(self.num_layers):
            for token_id in range(num_tokens):
                k = torch.randn(self.num_heads, self.head_dim)
                v = torch.randn(self.num_heads, self.head_dim)
                cache.quantize_and_store(layer_idx, token_id, k, v)
        
        # Get statistics
        stats = cache.get_memory_stats()
        
        result = EvaluationResult(
            baseline_name=baseline_name,
            memory_ratio=stats.get('memory_ratio', 1.0),
            avg_bits=stats.get('avg_bits'),
            cache_stats=stats
        )
        
        return result
    
    def evaluate_reconstruction_error(
        self,
        baseline_name: str,
        num_samples: int = 100
    ) -> float:
        """
        Evaluate reconstruction error for a baseline.
        
        Args:
            baseline_name: Name of the baseline
            num_samples: Number of samples to test
        
        Returns:
            Mean squared error
        """
        cache = self.create_cache(baseline_name)
        
        errors = []
        
        for i in range(num_samples):
            # Generate random KV
            k_orig = torch.randn(self.num_heads, self.head_dim)
            v_orig = torch.randn(self.num_heads, self.head_dim)
            
            # Store and retrieve
            cache.quantize_and_store(0, i, k_orig, v_orig)
            k_recon, v_recon = cache.retrieve(0, i)
            
            # Compute error
            k_error = torch.mean((k_orig - k_recon) ** 2).item()
            v_error = torch.mean((v_orig - v_recon) ** 2).item()
            errors.append((k_error + v_error) / 2)
        
        return sum(errors) / len(errors)
    
    def compare_all_baselines(
        self,
        num_tokens: int = 1000,
        include_smartkv: bool = False
    ) -> List[EvaluationResult]:
        """
        Compare all registered baselines.
        
        Args:
            num_tokens: Number of tokens to simulate
            include_smartkv: Whether to include SmartKV
        
        Returns:
            List of evaluation results
        """
        results = []
        
        for baseline_name in self.baselines.keys():
            if baseline_name == 'SmartKV' and not include_smartkv:
                continue
            
            try:
                result = self.evaluate_memory(baseline_name, num_tokens)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {baseline_name}: {e}")
        
        return results
    
    def print_comparison_table(
        self,
        results: List[EvaluationResult]
    ):
        """
        Print comparison table of results.
        
        Args:
            results: List of evaluation results
        """
        print("\n" + "="*80)
        print("Baseline Comparison")
        print("="*80)
        print(f"{'Baseline':<20} {'Memory Ratio':<15} {'Avg Bits':<12} {'MSE':<12}")
        print("-"*80)
        
        for result in results:
            avg_bits_str = f"{result.avg_bits:.2f}" if result.avg_bits else "N/A"
            print(
                f"{result.baseline_name:<20} "
                f"{result.memory_ratio:<15.3f} "
                f"{avg_bits_str:<12} "
            )
        
        print("="*80 + "\n")
    
    def get_baseline_names(self) -> List[str]:
        """Get list of registered baseline names."""
        return list(self.baselines.keys())
    
    def benchmark_latency(
        self,
        baseline_name: str,
        num_tokens: int = 100,
        num_iterations: int = 10
    ) -> float:
        """
        Benchmark latency for a baseline.
        
        Args:
            baseline_name: Name of the baseline
            num_tokens: Number of tokens to process
            num_iterations: Number of iterations for averaging
        
        Returns:
            Average latency in milliseconds
        """
        cache = self.create_cache(baseline_name)
        
        latencies = []
        
        for _ in range(num_iterations):
            # Generate data
            tokens_data = [
                (torch.randn(self.num_heads, self.head_dim),
                 torch.randn(self.num_heads, self.head_dim))
                for _ in range(num_tokens)
            ]
            
            # Measure store time
            start_time = time.perf_counter()
            for token_id, (k, v) in enumerate(tokens_data):
                cache.quantize_and_store(0, token_id, k, v)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            cache.clear()
        
        return sum(latencies) / len(latencies)


def get_default_baselines() -> List[str]:
    """
    Get list of default baseline names.
    
    Returns:
        List of baseline names
    """
    return ['FP16', 'Uniform-INT8', 'Uniform-INT4', 'KIVI']


def create_evaluator(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = "cpu"
) -> BaselineEvaluator:
    """
    Create a baseline evaluator with default configuration.
    
    Args:
        num_layers: Number of layers (default: 32 for Llama-7B)
        num_heads: Number of heads (default: 32 for Llama-7B)
        head_dim: Head dimension (default: 128 for Llama-7B)
        device: Device to use
    
    Returns:
        BaselineEvaluator instance
    """
    return BaselineEvaluator(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device
    )
