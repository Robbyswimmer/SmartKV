"""
Importance tracking for attention-guided precision allocation.

Tracks attention patterns across layers and computes token importance scores
to guide dynamic precision allocation in the SmartKV cache.
"""

import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


class ImportanceTracker:
    """
    Tracks token importance based on attention patterns.
    
    Maintains both per-layer and cumulative importance scores using
    Exponential Moving Average (EMA) for temporal smoothing.
    """
    
    def __init__(self, num_layers: int, decay: float = 0.9):
        """
        Initialize importance tracker.
        
        Args:
            num_layers: Number of transformer layers
            decay: EMA decay factor (0 < decay < 1)
                  Higher values give more weight to history
        """
        self.num_layers = num_layers
        self.decay = decay
        
        # Token importance scores
        self.token_importance: Dict[int, float] = {}  # token_id -> cumulative score
        self.layer_importance: Dict[Tuple[int, int], float] = {}  # (layer, token_id) -> score
        
        # History tracking for analysis
        self.importance_history: List[Dict[int, float]] = []
        self.update_count = 0
        
    def update_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        token_ids: List[int]
    ) -> None:
        """
        Update importance scores based on attention weights.
        
        Args:
            layer_idx: Layer index (0 to num_layers-1)
            attention_weights: Attention weights tensor
                Shape: [batch, num_heads, num_queries, num_keys]
            token_ids: List of token IDs corresponding to keys
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer_idx {layer_idx}, must be in [0, {self.num_layers})")
        
        # Sum attention weights across batch, heads, and queries to get per-key importance
        # Shape: [num_keys]
        if attention_weights.dim() == 4:
            # [batch, num_heads, num_queries, num_keys] -> [num_keys]
            key_importance = attention_weights.sum(dim=(0, 1, 2))
        elif attention_weights.dim() == 3:
            # [num_heads, num_queries, num_keys] -> [num_keys]
            key_importance = attention_weights.sum(dim=(0, 1))
        else:
            raise ValueError(f"Expected attention_weights to have 3 or 4 dims, got {attention_weights.dim()}")
        
        # Convert to float and update scores
        key_importance = key_importance.detach().cpu()
        
        for token_idx, token_id in enumerate(token_ids):
            if token_idx >= len(key_importance):
                break
                
            score = float(key_importance[token_idx])
            
            # Update layer-specific importance with EMA
            layer_key = (layer_idx, token_id)
            if layer_key not in self.layer_importance:
                self.layer_importance[layer_key] = score
            else:
                self.layer_importance[layer_key] = (
                    self.decay * self.layer_importance[layer_key] + 
                    (1 - self.decay) * score
                )
            
            # Update global token importance (cumulative across layers)
            if token_id not in self.token_importance:
                self.token_importance[token_id] = score
            else:
                # Accumulate scores across layers
                self.token_importance[token_id] += score
        
        self.update_count += 1
    
    def get_importance(self, token_id: int) -> float:
        """
        Get cumulative importance score for a token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Cumulative importance score (0.0 if token not seen)
        """
        return self.token_importance.get(token_id, 0.0)
    
    def get_layer_importance(self, layer_idx: int, token_id: int) -> float:
        """
        Get layer-specific importance score for a token.
        
        Args:
            layer_idx: Layer index
            token_id: Token ID
            
        Returns:
            Layer-specific importance score (0.0 if not seen)
        """
        return self.layer_importance.get((layer_idx, token_id), 0.0)
    
    def get_top_k_tokens(self, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k tokens by importance.
        
        Args:
            k: Number of top tokens to return
            
        Returns:
            List of (token_id, importance) tuples, sorted by importance (descending)
        """
        sorted_tokens = sorted(
            self.token_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_tokens[:k]
    
    def get_bottom_k_tokens(self, k: int) -> List[Tuple[int, float]]:
        """
        Get bottom-k tokens by importance.
        
        Args:
            k: Number of bottom tokens to return
            
        Returns:
            List of (token_id, importance) tuples, sorted by importance (ascending)
        """
        sorted_tokens = sorted(
            self.token_importance.items(),
            key=lambda x: x[1]
        )
        return sorted_tokens[:k]
    
    def get_importance_statistics(self) -> Dict[str, float]:
        """
        Compute statistics on importance scores.
        
        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        if not self.token_importance:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'total': 0.0,
                'num_tokens': 0
            }
        
        scores = np.array(list(self.token_importance.values()))
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'total': float(np.sum(scores)),
            'num_tokens': len(scores)
        }
    
    def get_layer_statistics(self, layer_idx: int) -> Dict[str, float]:
        """
        Compute statistics for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Dictionary with layer-specific statistics
        """
        layer_scores = [
            score for (layer, token_id), score in self.layer_importance.items()
            if layer == layer_idx
        ]
        
        if not layer_scores:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'num_tokens': 0
            }
        
        scores = np.array(layer_scores)
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'num_tokens': len(scores)
        }
    
    def reset(self) -> None:
        """Reset all importance scores."""
        self.token_importance.clear()
        self.layer_importance.clear()
        self.importance_history.clear()
        self.update_count = 0
    
    def snapshot(self) -> Dict[int, float]:
        """
        Take a snapshot of current token importance.
        
        Returns:
            Copy of current token importance dictionary
        """
        return self.token_importance.copy()
    
    def save_snapshot(self) -> None:
        """Save current importance scores to history."""
        self.importance_history.append(self.snapshot())


class AttentionAnalyzer:
    """
    Analyzes attention patterns for debugging and visualization.
    
    Provides utilities to understand which tokens receive high attention
    and how attention patterns correlate with task performance.
    """
    
    def __init__(self):
        """Initialize attention analyzer."""
        self.attention_logs: List[Dict] = []
    
    def log_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        token_ids: List[int],
        tokens: Optional[List[str]] = None
    ) -> None:
        """
        Log attention weights for later analysis.
        
        Args:
            layer_idx: Layer index
            attention_weights: Attention weights tensor
            token_ids: Token IDs
            tokens: Optional token strings for visualization
        """
        self.attention_logs.append({
            'layer': layer_idx,
            'weights': attention_weights.detach().cpu().numpy(),
            'token_ids': token_ids,
            'tokens': tokens
        })
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """
        Compute entropy of attention distribution.
        
        High entropy = uniform attention (harder to compress)
        Low entropy = focused attention (easier to compress)
        
        Args:
            attention_weights: Attention weights [batch, heads, queries, keys]
            
        Returns:
            Average attention entropy
        """
        # Average across batch and heads: [queries, keys]
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=(0, 1))
        elif attention_weights.dim() == 3:
            attn = attention_weights.mean(dim=0)
        else:
            attn = attention_weights
        
        # Compute entropy for each query
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        attn = attn + eps
        entropy = -(attn * torch.log(attn)).sum(dim=-1)
        
        return float(entropy.mean())
    
    def identify_attention_patterns(self) -> Dict[str, any]:
        """
        Identify common attention patterns in logged data.
        
        Returns:
            Dictionary describing attention patterns
        """
        if not self.attention_logs:
            return {}
        
        patterns = {
            'total_logs': len(self.attention_logs),
            'layers': set(log['layer'] for log in self.attention_logs),
            'avg_entropy_per_layer': {}
        }
        
        # Compute average entropy per layer
        layer_entropies = defaultdict(list)
        for log in self.attention_logs:
            weights_tensor = torch.from_numpy(log['weights'])
            entropy = self.compute_attention_entropy(weights_tensor)
            layer_entropies[log['layer']].append(entropy)
        
        for layer, entropies in layer_entropies.items():
            patterns['avg_entropy_per_layer'][layer] = np.mean(entropies)
        
        return patterns
    
    def get_high_attention_tokens(
        self,
        threshold: float = 0.1,
        layer_idx: Optional[int] = None
    ) -> List[int]:
        """
        Get tokens that receive high attention.
        
        Args:
            threshold: Minimum attention weight to consider "high"
            layer_idx: Optional layer to filter by
            
        Returns:
            List of token IDs with high attention
        """
        high_attention_tokens = set()
        
        for log in self.attention_logs:
            if layer_idx is not None and log['layer'] != layer_idx:
                continue
            
            weights = log['weights']
            token_ids = log['token_ids']
            
            # Average attention per key
            avg_attn = weights.mean(axis=(0, 1, 2)) if weights.ndim == 4 else weights.mean()
            
            for tid, attn in zip(token_ids, avg_attn.flatten()):
                if attn > threshold:
                    high_attention_tokens.add(tid)
        
        return list(high_attention_tokens)
    
    def clear_logs(self) -> None:
        """Clear all logged attention data."""
        self.attention_logs.clear()


def compute_ema_importance(
    current_scores: Dict[int, float],
    new_scores: Dict[int, float],
    decay: float = 0.9
) -> Dict[int, float]:
    """
    Update importance scores using Exponential Moving Average.
    
    Args:
        current_scores: Current importance scores
        new_scores: New importance scores to incorporate
        decay: EMA decay factor
        
    Returns:
        Updated importance scores
    """
    updated_scores = current_scores.copy()
    
    for token_id, new_score in new_scores.items():
        if token_id in updated_scores:
            updated_scores[token_id] = (
                decay * updated_scores[token_id] + 
                (1 - decay) * new_score
            )
        else:
            updated_scores[token_id] = new_score
    
    return updated_scores


def aggregate_layer_importance(
    layer_importance: Dict[Tuple[int, int], float],
    num_layers: int
) -> Dict[int, float]:
    """
    Aggregate layer-specific importance into token-level importance.
    
    Args:
        layer_importance: Dict mapping (layer_idx, token_id) to importance
        num_layers: Total number of layers
        
    Returns:
        Dict mapping token_id to aggregated importance
    """
    token_scores = defaultdict(float)
    
    for (layer_idx, token_id), score in layer_importance.items():
        token_scores[token_id] += score
    
    return dict(token_scores)
