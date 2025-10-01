"""
SmartKV Cache: Attention-guided adaptive precision KV-cache.

Main class that integrates quantizers, importance tracking, and precision
allocation to provide mixed-precision KV-cache compression.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
import psutil
import os

from smartkv.core.quantizers import get_quantizer, QuantizerBase
from smartkv.core.importance import ImportanceTracker
from smartkv.core.allocation import greedy_allocation, compute_memory_usage


class SmartKVCache:
    """
    SmartKV cache with attention-guided adaptive precision.
    
    Dynamically allocates 2-8 bit precision to tokens based on their attention
    patterns, achieving better accuracy-memory tradeoffs than uniform quantization.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        memory_budget: float = 0.5,
        decay: float = 0.9,
        realloc_freq: int = 16,
        available_bits: List[int] = [2, 3, 4, 8],
        device: str = "cpu"
    ):
        """
        Initialize SmartKV cache.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            memory_budget: Fraction of FP16 memory to use (e.g., 0.5 = 50%)
            decay: EMA decay for importance tracking
            realloc_freq: Reallocate precision every N tokens
            available_bits: Available bit-widths for quantization
            device: Device to store cache on ("cpu" or "cuda")
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.memory_budget = memory_budget
        self.decay = decay
        self.realloc_freq = realloc_freq
        self.available_bits = sorted(available_bits, reverse=True)
        self.device = device
        
        # Importance tracking
        self.importance_tracker = ImportanceTracker(
            num_layers=num_layers,
            decay=decay
        )
        
        # Precision mapping: token_id -> bits
        self.precision_map: Dict[int, int] = {}
        
        # Quantizers for each bit-width
        self.quantizers: Dict[int, QuantizerBase] = {
            bits: get_quantizer(bits) for bits in available_bits
        }
        
        # Cache storage: (layer, token_id) -> quantized data
        self.k_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.v_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        # Metadata
        self.token_counter = 0
        self.realloc_counter = 0
        self.total_tokens_stored = 0
        
    def update_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
        token_ids: List[int]
    ) -> None:
        """
        Update importance scores based on attention weights.
        
        Should be called after each attention computation.
        
        Args:
            layer_idx: Layer index
            attention_weights: Attention weights [batch, heads, queries, keys]
            token_ids: Token IDs corresponding to keys
        """
        self.importance_tracker.update_attention(
            layer_idx,
            attention_weights,
            token_ids
        )
        
        # Periodically reallocate precision
        if self.token_counter % self.realloc_freq == 0:
            self.allocate_precision(token_ids)
            self.realloc_counter += 1
    
    def allocate_precision(self, token_ids: List[int]) -> Dict[int, int]:
        """
        Allocate precision to tokens based on importance.
        
        Args:
            token_ids: List of token IDs to allocate precision for
        
        Returns:
            Dict mapping token_id -> allocated bits
        """
        # Get importance scores for tokens
        importance_scores = {
            tid: self.importance_tracker.get_importance(tid)
            for tid in token_ids
        }
        
        # Use greedy allocation
        allocation = greedy_allocation(
            importance_scores,
            memory_budget=self.memory_budget,
            available_bits=self.available_bits
        )

        # Encourage mixed precision when possible (avoid uniform allocation)
        if allocation and len(set(allocation.values())) == 1 and len(allocation) > 1:
            # Downgrade least important tokens until multiple precisions are used
            sorted_tokens = sorted(
                importance_scores.items(),
                key=lambda x: x[1]
            )
            min_bits = min(self.available_bits)
            for token_id, _ in sorted_tokens:
                if allocation[token_id] > min_bits:
                    allocation[token_id] = min_bits
                    if len(set(allocation.values())) > 1:
                        break

        # Update precision map
        self.precision_map.update(allocation)

        return allocation
    
    def quantize_and_store(
        self,
        layer_idx: int,
        token_id: int,
        k_vec: torch.Tensor,
        v_vec: torch.Tensor
    ) -> None:
        """
        Quantize and store key and value vectors.
        
        Args:
            layer_idx: Layer index
            token_id: Token ID
            k_vec: Key vector to store [num_heads, head_dim] or [head_dim]
            v_vec: Value vector to store [num_heads, head_dim] or [head_dim]
        """
        # Get allocated precision (default to 4-bit if not allocated yet)
        bits = self.precision_map.get(token_id, 4)
        quantizer = self.quantizers[bits]

        # Ensure tensors are on CPU for storage and use consistent dtype
        k_cpu = k_vec.detach().to(torch.float32).cpu()
        v_cpu = v_vec.detach().to(torch.float32).cpu()

        # Quantize key/value vectors
        k_qdata = quantizer.quantize(k_cpu)
        v_qdata = quantizer.quantize(v_cpu)
        k_qdata['bits'] = bits
        v_qdata['bits'] = bits

        cache_key = (layer_idx, token_id)
        self.k_cache[cache_key] = k_qdata
        self.v_cache[cache_key] = v_qdata

        # Track precision for reporting even if allocation has not yet run
        if token_id not in self.precision_map:
            self.precision_map[token_id] = bits

        self.total_tokens_stored += 1
        self.token_counter += 1
    
    def retrieve(
        self,
        layer_idx: int,
        token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve and dequantize key and value vectors.
        
        Args:
            layer_idx: Layer index
            token_id: Token ID
        
        Returns:
            Tuple of (key, value) tensors
        
        Raises:
            KeyError: If token not found in cache
        """
        cache_key = (layer_idx, token_id)

        if cache_key not in self.k_cache:
            raise KeyError(f"Token {token_id} not found in cache for layer {layer_idx}")

        k_entry = self.k_cache[cache_key]
        v_entry = self.v_cache[cache_key]

        k_bits = k_entry.get('bits', 4)
        v_bits = v_entry.get('bits', 4)

        k_quantizer = self.quantizers[k_bits]
        v_quantizer = self.quantizers[v_bits]

        # Dequantize back to floating point
        k = k_quantizer.dequantize(k_entry).to(k_entry.get('dtype', torch.float32))
        v = v_quantizer.dequantize(v_entry).to(v_entry.get('dtype', torch.float32))

        # Move to device if needed
        if self.device != "cpu":
            k = k.to(self.device)
            v = v.to(self.device)

        return k, v
    
    def retrieve_all(
        self,
        layer_idx: int,
        token_ids: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve all cached KV for a layer.
        
        Args:
            layer_idx: Layer index
            token_ids: Optional list of token IDs to retrieve.
                      If None, retrieves all tokens in cache for this layer.
        
        Returns:
            Tuple of (keys, values) tensors
            Shape: [num_tokens, num_heads, head_dim] or [num_tokens, head_dim]
        """
        if token_ids is None:
            # Get all token IDs for this layer
            token_ids = sorted([
                tid for (l, tid) in self.k_cache.keys() if l == layer_idx
            ])
        
        if not token_ids:
            # Return empty tensors
            return torch.empty(0), torch.empty(0)
        
        # Retrieve each token
        keys = []
        values = []
        for token_id in token_ids:
            try:
                k, v = self.retrieve(layer_idx, token_id)
                keys.append(k)
                values.append(v)
            except KeyError:
                # Token not in cache, skip
                continue
        
        if not keys:
            return torch.empty(0), torch.empty(0)
        
        # Stack into tensors
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        
        return keys, values
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory statistics
        """
        # Compute memory usage from precision map
        if self.precision_map:
            allocation_stats = compute_memory_usage(self.precision_map)
        else:
            allocation_stats = {
                'total_bits': 0,
                'avg_bits': 0.0,
                'memory_ratio': 0.0,
                'num_tokens': 0
            }
        
        # Count cache entries
        num_cache_entries = len(self.k_cache)
        
        # Compute precision distribution
        precision_distribution = {}
        for bits in self.available_bits:
            count = sum(1 for b in self.precision_map.values() if b == bits)
            precision_distribution[f"{bits}-bit"] = count
        
        # Get system memory info (if available)
        try:
            process = psutil.Process(os.getpid())
            system_memory_mb = process.memory_info().rss / 1024 / 1024
        except:
            system_memory_mb = None
        
        return {
            'memory_budget': self.memory_budget,
            'memory_ratio': allocation_stats['memory_ratio'],
            'avg_bits': allocation_stats['avg_bits'],
            'num_tokens': allocation_stats['num_tokens'],
            'num_cache_entries': num_cache_entries,
            'precision_distribution': precision_distribution,
            'total_tokens_stored': self.total_tokens_stored,
            'realloc_count': self.realloc_counter,
            'system_memory_mb': system_memory_mb,
        }
    
    def get_precision(self, token_id: int) -> int:
        """
        Get allocated precision for a token.
        
        Args:
            token_id: Token ID
        
        Returns:
            Allocated bits (defaults to 4 if not allocated)
        """
        return self.precision_map.get(token_id, 4)
    
    def get_importance(self, token_id: int) -> float:
        """
        Get importance score for a token.
        
        Args:
            token_id: Token ID
        
        Returns:
            Importance score
        """
        return self.importance_tracker.get_importance(token_id)
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.k_cache.clear()
        self.v_cache.clear()
        self.precision_map.clear()
        self.importance_tracker.reset()
        self.token_counter = 0
        self.realloc_counter = 0
        self.total_tokens_stored = 0
    
    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()
    
    def get_top_k_important_tokens(self, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k most important tokens.
        
        Args:
            k: Number of top tokens to return
        
        Returns:
            List of (token_id, importance) tuples
        """
        return self.importance_tracker.get_top_k_tokens(k)
    
    def get_bottom_k_important_tokens(self, k: int) -> List[Tuple[int, float]]:
        """
        Get bottom-k least important tokens.
        
        Args:
            k: Number of bottom tokens to return
        
        Returns:
            List of (token_id, importance) tuples
        """
        return self.importance_tracker.get_bottom_k_tokens(k)
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export cache state for checkpointing.
        
        Returns:
            Dict with cache state
        """
        return {
            'config': {
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'head_dim': self.head_dim,
                'memory_budget': self.memory_budget,
                'decay': self.decay,
                'realloc_freq': self.realloc_freq,
                'available_bits': self.available_bits,
            },
            'precision_map': self.precision_map.copy(),
            'token_counter': self.token_counter,
            'realloc_counter': self.realloc_counter,
            'total_tokens_stored': self.total_tokens_stored,
            # Note: We don't export the actual cached tensors as they're large
            # In practice, cache is regenerated during inference
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SmartKVCache(layers={self.num_layers}, "
            f"budget={self.memory_budget:.1%}, "
            f"tokens={len(self.precision_map)}, "
            f"cached={len(self.k_cache)})"
        )
    
    def __len__(self) -> int:
        """Number of tokens with allocated precision."""
        return len(self.precision_map)
