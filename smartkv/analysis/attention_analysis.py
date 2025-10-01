"""
Attention pattern analysis for SmartKV.

Analyzes attention distributions across layers to validate the core hypothesis:
tokens receiving high attention are more sensitive to quantization errors.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class AttentionStats:
    """Statistics about attention patterns."""

    mean: float
    std: float
    min: float
    max: float
    entropy: float
    top_k_tokens: List[Tuple[int, float]] = field(default_factory=list)
    percentiles: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'entropy': self.entropy,
            'top_k_tokens': self.top_k_tokens,
            'percentiles': self.percentiles
        }


class AttentionLogger:
    """
    Logs attention weights during model inference.

    Uses forward hooks to capture attention patterns without modifying model code.
    """

    def __init__(self):
        """Initialize attention logger."""
        self.attention_weights: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.hooks: List[Any] = []
        self.layer_names: Dict[str, int] = {}
        self.current_step = 0

    def register_hooks(self, model: nn.Module):
        """
        Register forward hooks on attention layers.

        Args:
            model: PyTorch model to monitor
        """
        layer_idx = 0

        for name, module in model.named_modules():
            # Match Llama attention layers
            if 'self_attn' in name or isinstance(module, type(module)) and 'Attention' in type(module).__name__:
                self.layer_names[name] = layer_idx
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx, name)
                )
                self.hooks.append(hook)
                layer_idx += 1

    def _make_hook(self, layer_idx: int, layer_name: str):
        """Create hook function for specific layer."""
        def hook(module, input, output):
            # Try to extract attention weights from output
            # Different models have different output formats
            if isinstance(output, tuple):
                # Check if attention weights are in output
                for item in output:
                    if isinstance(item, torch.Tensor):
                        # Attention weights typically have shape [batch, heads, seq, seq]
                        if item.dim() == 4:
                            self.attention_weights[layer_idx].append(item.detach().cpu())
                            return
            elif isinstance(output, dict) and 'attentions' in output:
                attn = output['attentions']
                if isinstance(attn, torch.Tensor):
                    self.attention_weights[layer_idx].append(attn.detach().cpu())

        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_attention_weights(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Get captured attention weights for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Attention weights tensor or None if not captured
        """
        if layer_idx not in self.attention_weights:
            return None

        weights = self.attention_weights[layer_idx]
        if not weights:
            return None

        # Concatenate all captured weights
        return torch.cat(weights, dim=0)

    def get_all_attention_weights(self) -> Dict[int, torch.Tensor]:
        """Get all captured attention weights."""
        result = {}
        for layer_idx in self.attention_weights.keys():
            weights = self.get_attention_weights(layer_idx)
            if weights is not None:
                result[layer_idx] = weights
        return result

    def clear(self):
        """Clear all captured attention weights."""
        self.attention_weights.clear()
        self.current_step = 0


def compute_attention_entropy(attention_weights: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute entropy of attention distribution.

    Higher entropy = more uniform attention
    Lower entropy = more concentrated attention

    Args:
        attention_weights: Attention weights [batch, heads, queries, keys]
        eps: Small value for numerical stability

    Returns:
        Average entropy across all attention distributions
    """
    # Ensure weights are normalized
    attn = attention_weights + eps
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Compute entropy: -sum(p * log(p))
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)

    # Average across batch and heads
    return entropy.mean().item()


def analyze_attention_distribution(
    attention_weights: torch.Tensor,
    k: int = 10
) -> AttentionStats:
    """
    Analyze attention weight distribution.

    Args:
        attention_weights: Attention weights [batch, heads, queries, keys]
        k: Number of top tokens to return

    Returns:
        AttentionStats with distribution statistics
    """
    # Sum across batch, heads, and queries to get per-key importance
    # Shape: [num_keys]
    key_importance = attention_weights.sum(dim=(0, 1, 2))

    # Compute statistics
    mean_val = key_importance.mean().item()
    std_val = key_importance.std().item()
    min_val = key_importance.min().item()
    max_val = key_importance.max().item()

    # Compute entropy
    entropy = compute_attention_entropy(attention_weights)

    # Find top-k tokens
    top_k_values, top_k_indices = torch.topk(key_importance, min(k, len(key_importance)))
    top_k_tokens = [
        (idx.item(), val.item())
        for idx, val in zip(top_k_indices, top_k_values)
    ]

    # Compute percentiles
    percentiles = {}
    for p in [10, 25, 50, 75, 90, 95, 99]:
        percentiles[p] = torch.quantile(key_importance, p / 100.0).item()

    return AttentionStats(
        mean=mean_val,
        std=std_val,
        min=min_val,
        max=max_val,
        entropy=entropy,
        top_k_tokens=top_k_tokens,
        percentiles=percentiles
    )


def identify_high_attention_tokens(
    attention_weights: torch.Tensor,
    threshold_percentile: float = 90.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identify tokens receiving high attention.

    Args:
        attention_weights: Attention weights [batch, heads, queries, keys]
        threshold_percentile: Percentile threshold for "high attention"

    Returns:
        (high_attention_indices, low_attention_indices)
    """
    # Sum across batch, heads, and queries
    key_importance = attention_weights.sum(dim=(0, 1, 2))

    # Compute threshold
    threshold = torch.quantile(key_importance, threshold_percentile / 100.0)

    # Find high and low attention tokens
    high_mask = key_importance >= threshold
    low_mask = key_importance < threshold

    high_indices = torch.where(high_mask)[0]
    low_indices = torch.where(low_mask)[0]

    return high_indices, low_indices


def measure_quantization_sensitivity(
    kv_tensors: torch.Tensor,
    attention_weights: torch.Tensor,
    quantizer,
    threshold_percentile: float = 90.0
) -> Dict[str, float]:
    """
    Measure how quantization affects high vs low attention tokens.

    Args:
        kv_tensors: Key or value tensors to quantize [seq_len, hidden_dim]
        attention_weights: Attention weights [batch, heads, queries, keys]
        quantizer: Quantizer object with quantize/dequantize methods
        threshold_percentile: Percentile for high attention threshold

    Returns:
        Dict with error metrics for high and low attention tokens
    """
    # Identify high/low attention tokens
    high_indices, low_indices = identify_high_attention_tokens(
        attention_weights,
        threshold_percentile
    )

    # Quantize all tensors
    quantized = quantizer.quantize(kv_tensors)
    dequantized = quantizer.dequantize(quantized)

    # Compute reconstruction error
    error = (kv_tensors - dequantized).abs()

    # Separate errors for high/low attention tokens
    high_error = error[high_indices].mean().item() if len(high_indices) > 0 else 0.0
    low_error = error[low_indices].mean().item() if len(low_indices) > 0 else 0.0

    return {
        'high_attention_error': high_error,
        'low_attention_error': low_error,
        'error_ratio': high_error / (low_error + 1e-10),
        'num_high_tokens': len(high_indices),
        'num_low_tokens': len(low_indices)
    }


def plot_attention_heatmap(
    attention_weights: Dict[int, torch.Tensor],
    token_texts: Optional[List[str]] = None,
    max_tokens: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[Any]:
    """
    Plot attention heatmap across layers.

    Args:
        attention_weights: Dict mapping layer_idx to attention weights
        token_texts: Optional token text labels
        max_tokens: Maximum tokens to show
        figsize: Figure size

    Returns:
        Matplotlib figure or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib seaborn")
        return None

    num_layers = len(attention_weights)

    # Aggregate attention across layers
    # For each layer, sum attention received by each key token
    attention_matrix = []

    for layer_idx in sorted(attention_weights.keys()):
        attn = attention_weights[layer_idx]
        # Sum across batch, heads, and queries: [num_keys]
        key_importance = attn.sum(dim=(0, 1, 2))

        # Truncate to max_tokens
        key_importance = key_importance[:max_tokens]
        attention_matrix.append(key_importance.numpy())

    # Convert to numpy array [num_layers, num_tokens]
    attention_matrix = np.array(attention_matrix)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        attention_matrix,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Attention Score'}
    )

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title('Attention Heatmap Across Layers')

    # Add token labels if provided
    if token_texts and len(token_texts) >= max_tokens:
        ax.set_xticklabels(token_texts[:max_tokens], rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_attention_distribution(
    attention_weights: torch.Tensor,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[Any]:
    """
    Plot histogram of attention score distribution.

    Args:
        attention_weights: Attention weights [batch, heads, queries, keys]
        figsize: Figure size

    Returns:
        Matplotlib figure or None
    """
    if not PLOTTING_AVAILABLE:
        return None

    # Sum to get per-key importance
    key_importance = attention_weights.sum(dim=(0, 1, 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(key_importance.numpy(), bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Attention Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Attention Score Distribution')
    ax1.axvline(key_importance.mean().item(), color='red', linestyle='--', label='Mean')
    ax1.legend()

    # CDF
    sorted_importance = torch.sort(key_importance)[0]
    cdf = torch.arange(1, len(sorted_importance) + 1) / len(sorted_importance)
    ax2.plot(sorted_importance.numpy(), cdf.numpy())
    ax2.set_xlabel('Attention Score')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Attention Score CDF')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_quantization_sensitivity(
    results: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[Any]:
    """
    Plot quantization sensitivity for high vs low attention tokens.

    Args:
        results: List of dicts from measure_quantization_sensitivity
        figsize: Figure size

    Returns:
        Matplotlib figure or None
    """
    if not PLOTTING_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    high_errors = [r['high_attention_error'] for r in results]
    low_errors = [r['low_attention_error'] for r in results]
    x = np.arange(len(results))

    width = 0.35
    ax.bar(x - width/2, high_errors, width, label='High Attention Tokens', alpha=0.8)
    ax.bar(x + width/2, low_errors, width, label='Low Attention Tokens', alpha=0.8)

    ax.set_xlabel('Quantization Level')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Quantization Sensitivity: High vs Low Attention Tokens')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig
