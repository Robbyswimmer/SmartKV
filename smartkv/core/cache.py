"""
SmartKV Cache: Attention-guided adaptive precision KV-cache.

Main class that integrates quantizers, importance tracking, and precision
allocation to provide mixed-precision KV-cache compression.
"""

import math
import torch
from typing import Dict, List, Tuple, Optional, Any
import psutil
import os

from smartkv.core._quant_cpu import quantize_per_head

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
        device: str = "cpu",
        special_token_ids: Optional[List[int]] = None
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
            special_token_ids: Token IDs to always keep at 8-bit (BOS, EOS, etc.)
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay = decay
        self.realloc_freq = realloc_freq
        self.available_bits = sorted(available_bits, reverse=True)
        self.device = device

        # Storage configuration
        self.use_packing = False  # Bit-packing not yet implemented
        self.scale_dtype = "fp32"  # FP32 scales (can be changed to "fp16")

        # Validate and adjust memory budget
        from smartkv.core.allocation import compute_minimum_budget
        self.min_budget = compute_minimum_budget(
            num_heads=num_heads,
            head_dim=head_dim,
            scale_dtype=self.scale_dtype,
            use_packing=self.use_packing
        )

        if memory_budget < self.min_budget:
            import warnings
            warnings.warn(
                f"Requested budget {memory_budget:.4f} is below minimum achievable "
                f"budget {self.min_budget:.4f} with current storage "
                f"(INT8 + {self.scale_dtype.upper()} scales, packing={self.use_packing}). "
                f"Clamping to minimum budget.",
                UserWarning
            )
            self.memory_budget = self.min_budget
        else:
            self.memory_budget = memory_budget

        # Special tokens always get 8-bit precision
        self.special_token_ids = set(special_token_ids or [])
        # Common special tokens that should never be low-precision
        self.protected_tokens = set([0, 1, 2])  # BOS, EOS, PAD typically

        self.head_importance = torch.ones(self.num_heads, dtype=torch.float32)
        self.head_importance_decay = 0.9
        self.recency_temperature = 512.0
        self.global_step = 0
        self.last_seen: Dict[int, int] = {}
        self.critical_token_window = 4
        self.critical_min_bits = max(self.available_bits)
        self.min_general_bits = min((b for b in self.available_bits if b >= 3), default=self.available_bits[-1])
        
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
        
        # Cache storage: (layer, token_id) -> quantized data (metadata view)
        self.k_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.v_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Per-layer quantized buffers
        self.layer_store: List[Dict[str, Any]] = [
            self._init_layer_store() for _ in range(num_layers)
        ]
        
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
        if attention_weights is not None:
            head_dim = attention_weights.size(1)
            if self.head_importance.numel() != head_dim:
                self.head_importance = torch.ones(head_dim, dtype=torch.float32)

            if attention_weights.device.type != 'cpu':
                head_stats = attention_weights.detach().to('cpu')
            else:
                head_stats = attention_weights.detach()

            head_mean = head_stats.mean(dim=(0, 2, 3))
            self.head_importance = (
                self.head_importance * self.head_importance_decay
                + head_mean * (1.0 - self.head_importance_decay)
            )

            normalized = self.head_importance / (self.head_importance.mean() + 1e-6)
            # NOTE: This rescales attention weights by head importance without renormalizing.
            # The result is NOT a probability distribution (doesn't sum to 1), but serves
            # as a heuristic importance signal for precision allocation. If probability
            # semantics are needed, apply softmax after this operation.
            attention_weights = attention_weights * normalized.view(1, -1, 1, 1).to(attention_weights.device)

        self.importance_tracker.update_attention(
            layer_idx,
            attention_weights,
            token_ids
        )

        self.global_step += 1
        current_step = self.global_step
        for token_id in token_ids:
            self.last_seen[int(token_id)] = current_step

        # Adaptive reallocation frequency based on context length
        # Scale frequency with number of tokens to reduce overhead at long context
        num_tokens = len(token_ids)
        total_tokens = len(self.precision_map) + num_tokens
        layer_factor = 1 + layer_idx // 4
        adaptive_freq = max(self.realloc_freq * layer_factor, max(32, total_tokens // 8))

        # Periodically reallocate precision
        if self.token_counter % adaptive_freq == 0:
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
        if not token_ids:
            return {}

        # Get base importance scores
        importance_scores = {
            tid: self.importance_tracker.get_importance(tid)
            for tid in token_ids
        }

        # Recency weighting
        if self.recency_temperature > 0:
            for tid in token_ids:
                age = self.global_step - self.last_seen.get(tid, self.global_step)
                importance_scores[tid] *= math.exp(-age / self.recency_temperature)

        protected_set = self.protected_tokens | self.special_token_ids
        for tid in token_ids:
            if tid in protected_set or tid < self.critical_token_window:
                importance_scores[tid] = float('inf')

        sorted_tokens = sorted(token_ids, key=lambda tid: importance_scores.get(tid, 0.0), reverse=True)
        if not sorted_tokens:
            return {}

        tiers = [0.1, 0.3, 0.7, 1.0]
        bits_order = self.available_bits + [self.available_bits[-1]] * 4
        tier_alloc: Dict[int, int] = {}
        total = len(sorted_tokens)
        for idx, tid in enumerate(sorted_tokens):
            frac = (idx + 1) / total
            tier_idx = next((i for i, edge in enumerate(tiers) if frac <= edge), len(self.available_bits) - 1)
            tier_idx = min(tier_idx, len(bits_order) - 1)
            tier_bits = max(bits_order[tier_idx], self.min_general_bits)
            tier_alloc[tid] = tier_bits

        # Ensure protected / early tokens have maximum precision
        for tid in sorted_tokens:
            if tid in protected_set or tid < self.critical_token_window:
                tier_alloc[tid] = self.critical_min_bits

        # Top 5% tokens also get maximum bits
        top_count = max(1, int(0.05 * len(sorted_tokens)))
        for tid in sorted_tokens[:top_count]:
            tier_alloc[tid] = max(tier_alloc[tid], self.available_bits[0])

        def compute_ratio(mapping: Dict[int, int]) -> tuple:
            stats = compute_memory_usage(
                mapping,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                scale_dtype=self.scale_dtype,
                use_packing=self.use_packing
            )
            # Return both ratios for logging
            return stats.get('memory_ratio_true', 0.0), stats

        def lower_bits(bits: int) -> int:
            if bits not in self.available_bits:
                return self.available_bits[-1]
            idx = self.available_bits.index(bits)
            next_bits = bits if idx == len(self.available_bits) - 1 else self.available_bits[idx + 1]
            return max(next_bits, self.min_general_bits)

        # Global budget enforcement: check merged map before committing
        merged_map = {**self.precision_map, **tier_alloc}
        budget_ratio, stats = compute_ratio(merged_map)

        if budget_ratio > self.memory_budget:
            while budget_ratio > self.memory_budget:
                changed = False
                for tid in reversed(sorted_tokens):
                    if tid in protected_set or tid < self.critical_token_window:
                        continue
                    old_bits = tier_alloc[tid]
                    new_bits = lower_bits(old_bits)
                    if new_bits == old_bits:
                        continue
                    tier_alloc[tid] = new_bits
                    # Recompute on merged map
                    merged_map = {**self.precision_map, **tier_alloc}
                    budget_ratio, stats = compute_ratio(merged_map)
                    changed = True
                    if budget_ratio <= self.memory_budget:
                        break
                if not changed:
                    break

        self.precision_map.update(tier_alloc)

        # Log dual budget reporting
        if hasattr(self, '_logger') and self._logger:
            self._logger.debug(
                f"Budget enforcement: "
                f"Theoretical={stats.get('memory_ratio', 0.0):.3f} "
                f"(avg {stats.get('avg_bits', 0.0):.2f} bits), "
                f"True={budget_ratio:.3f} "
                f"({stats.get('storage_mode', 'unknown')} + {stats.get('scale_dtype', 'unknown')} scales), "
                f"Target={self.memory_budget:.3f}, "
                f"Tokens={stats.get('num_tokens', 0)}"
            )

        return tier_alloc

    def _init_layer_store(self) -> Dict[str, Any]:
        """Create empty storage buffers for a layer."""
        return {
            'capacity': 0,
            'size': 0,
            'token_ids': None,
            'k_qx': None,
            'v_qx': None,
            'k_scale': None,
            'v_scale': None,
            'bits': None,
        }

    def _ensure_layer_capacity(self, layer_store: Dict[str, Any], min_capacity: int) -> None:
        """Ensure buffers can hold at least `min_capacity` tokens."""
        if layer_store['capacity'] >= min_capacity:
            return

        new_capacity = max(64, layer_store['capacity'] * 2 if layer_store['capacity'] else 64)
        while new_capacity < min_capacity:
            new_capacity *= 2

        def _grow_tensor(tensor: Optional[torch.Tensor], shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            if tensor is None:
                return torch.empty(shape, dtype=dtype)
            new_tensor = torch.empty(shape, dtype=dtype)
            prev = tensor.shape[0]
            if prev > 0:
                new_tensor[:prev] = tensor
            return new_tensor

        layer_store['token_ids'] = _grow_tensor(layer_store['token_ids'], (new_capacity,), torch.long)
        layer_store['k_qx'] = _grow_tensor(
            layer_store['k_qx'],
            (new_capacity, self.num_heads, self.head_dim),
            torch.int8,
        )
        layer_store['v_qx'] = _grow_tensor(
            layer_store['v_qx'],
            (new_capacity, self.num_heads, self.head_dim),
            torch.int8,
        )
        layer_store['k_scale'] = _grow_tensor(layer_store['k_scale'], (new_capacity, self.num_heads), torch.float32)
        layer_store['v_scale'] = _grow_tensor(layer_store['v_scale'], (new_capacity, self.num_heads), torch.float32)
        layer_store['bits'] = _grow_tensor(layer_store['bits'], (new_capacity,), torch.uint8)
        layer_store['capacity'] = new_capacity
    
    def quantize_and_store(
        self,
        layer_idx: int,
        token_id: int,
        k_vec: torch.Tensor,
        v_vec: torch.Tensor
    ) -> None:
        """Quantize and store a single KV vector."""
        if k_vec.dim() == 1:
            k_vec = k_vec.unsqueeze(0)
        if v_vec.dim() == 1:
            v_vec = v_vec.unsqueeze(0)
        if k_vec.dim() == 2:
            k_vec = k_vec.unsqueeze(0)
        if v_vec.dim() == 2:
            v_vec = v_vec.unsqueeze(0)

        self.quantize_and_store_batch(
            layer_idx,
            [int(token_id)],
            k_vec,
            v_vec,
        )

    def quantize_and_store_batch(
        self,
        layer_idx: int,
        token_ids: List[int],
        k_batch: torch.Tensor,
        v_batch: torch.Tensor
    ) -> None:
        """Quantize and store a batch of KV vectors for a layer."""
        if not token_ids:
            return

        layer_store = self.layer_store[layer_idx]

        k_cpu = k_batch.detach().to(torch.float32)
        v_cpu = v_batch.detach().to(torch.float32)
        if k_cpu.device.type != "cpu":
            k_cpu = k_cpu.cpu()
        if v_cpu.device.type != "cpu":
            v_cpu = v_cpu.cpu()

        seq_len = k_cpu.shape[0]
        assert seq_len == len(token_ids), "Mismatched KV batch length"

        bits_list = [self.precision_map.get(token_id, 4) for token_id in token_ids]
        bits_tensor = torch.tensor(bits_list, dtype=torch.uint8)

        k_qx = torch.empty((seq_len, self.num_heads, self.head_dim), dtype=torch.int8)
        v_qx = torch.empty((seq_len, self.num_heads, self.head_dim), dtype=torch.int8)
        k_scale = torch.empty((seq_len, self.num_heads), dtype=torch.float32)
        v_scale = torch.empty((seq_len, self.num_heads), dtype=torch.float32)

        unique_bits = bits_tensor.unique(sorted=True)
        for bits in unique_bits.tolist():
            mask = bits_tensor == bits
            if not bool(mask.any()):
                continue

            k_subset = k_cpu[mask]
            v_subset = v_cpu[mask]

            max_val = 2 ** (bits - 1) - 1
            min_val = -2 ** (bits - 1)
            if bits == 1:
                max_val = 0
                min_val = 0

            # Keys
            k_q, v_q, k_scale_subset, v_scale_subset = quantize_per_head(k_subset, v_subset, bits)

            k_qx[mask] = k_q
            v_qx[mask] = v_q
            k_scale[mask] = k_scale_subset
            v_scale[mask] = v_scale_subset

        for idx, token_id in enumerate(token_ids):
            token_int = int(token_id)
            if token_int not in self.precision_map:
                self.precision_map[token_int] = int(bits_tensor[idx])

        start = layer_store['size']
        new_size = start + seq_len
        self._ensure_layer_capacity(layer_store, new_size)

        token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
        # NOTE: We append tokens sequentially, assuming token_ids arrive in
        # monotonically increasing order (typical for autoregressive generation).
        # If token_ids are not sorted, torch.searchsorted in retrieve/retrieve_all
        # will fail. For non-sequential access patterns, use a dict-based index.
        layer_store['token_ids'][start:new_size] = token_id_tensor
        layer_store['k_qx'][start:new_size] = k_qx
        layer_store['v_qx'][start:new_size] = v_qx
        layer_store['k_scale'][start:new_size] = k_scale
        layer_store['v_scale'][start:new_size] = v_scale
        layer_store['bits'][start:new_size] = bits_tensor

        for offset, token_id in enumerate(token_ids):
            slot = start + offset
            token_int = int(token_id)

            cache_key = (layer_idx, token_int)
            self.k_cache[cache_key] = {
                'qx': layer_store['k_qx'][slot],
                'scale': layer_store['k_scale'][slot],
                'bits': int(bits_tensor[offset]),
                'dtype': torch.float32,
                'index': slot,
                'layer': layer_idx,
            }
            self.v_cache[cache_key] = {
                'qx': layer_store['v_qx'][slot],
                'scale': layer_store['v_scale'][slot],
                'bits': int(bits_tensor[offset]),
                'dtype': torch.float32,
                'index': slot,
                'layer': layer_idx,
            }

        layer_store['size'] = new_size
        self.total_tokens_stored += seq_len
        self.token_counter += seq_len
    
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
        layer_store = self.layer_store[layer_idx]
        size = layer_store['size']
        if size == 0:
            raise KeyError(f"Token {token_id} not found in cache for layer {layer_idx}")

        token_ids = layer_store['token_ids'][:size]
        query = torch.tensor([token_id], dtype=torch.long)
        pos = torch.searchsorted(token_ids, query)
        idx = pos.item()
        if idx >= size or token_ids[idx] != token_id:
            raise KeyError(f"Token {token_id} not found in cache for layer {layer_idx}")

        indices = torch.tensor([idx], dtype=torch.long)
        keys, values = self._dequantize_indices(layer_store, indices)

        k = keys[0]
        v = values[0]

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
        Retrieve all cached KV for a layer with batched dequantization.

        Args:
            layer_idx: Layer index
            token_ids: Optional list of token IDs to retrieve.
                      If None, retrieves all tokens in cache for this layer.

        Returns:
            Tuple of (keys, values) tensors
            Shape: [num_tokens, num_heads, head_dim]
        """
        layer_store = self.layer_store[layer_idx]
        size = layer_store['size']

        if size == 0:
            empty = torch.empty(0, self.num_heads, self.head_dim)
            return empty, empty

        token_ids_tensor = layer_store['token_ids'][:size]
        if token_ids is None:
            indices = torch.arange(size, dtype=torch.long)
        else:
            query = torch.tensor(token_ids, dtype=torch.long)
            pos = torch.searchsorted(token_ids_tensor, query)
            valid = (pos < size) & (token_ids_tensor[pos] == query)
            if not bool(valid.any()):
                empty = torch.empty(0, self.num_heads, self.head_dim)
                return empty, empty
            indices = pos[valid]
            # preserve original query ordering
            order = torch.arange(len(token_ids))[valid]
            sorted_idx = torch.argsort(order)
            indices = indices[sorted_idx]

        keys, values = self._dequantize_indices(layer_store, indices)

        if self.device != "cpu":
            keys = keys.to(self.device)
            values = values.to(self.device)

        return keys, values

    def retrieve_all_quantized(
        self,
        layer_idx: int,
        token_ids: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Return raw quantized buffers and metadata for fused attention."""
        layer_store = self.layer_store[layer_idx]
        size = layer_store['size']
        if size == 0:
            empty_ids = torch.empty(0, dtype=torch.long)
            empty_k = torch.empty((0, self.num_heads, self.head_dim), dtype=torch.int8)
            empty_scale = torch.empty((0, self.num_heads), dtype=torch.float32)
            return {
                'token_ids': empty_ids,
                'k_qx': empty_k,
                'k_scale': empty_scale,
                'v_qx': empty_k.clone(),
                'v_scale': empty_scale.clone(),
            }

        token_ids_tensor = layer_store['token_ids'][:size]
        if token_ids is None:
            indices = torch.arange(size, dtype=torch.long)
        else:
            query = torch.tensor(token_ids, dtype=torch.long)
            pos = torch.searchsorted(token_ids_tensor, query)
            valid = (pos < size) & (token_ids_tensor[pos] == query)
            if not bool(valid.any()):
                return {
                    'token_ids': torch.empty(0, dtype=torch.long),
                    'k_qx': torch.empty((0, self.num_heads, self.head_dim), dtype=torch.int8),
                    'k_scale': torch.empty((0, self.num_heads), dtype=torch.float32),
                    'v_qx': torch.empty((0, self.num_heads, self.head_dim), dtype=torch.int8),
                    'v_scale': torch.empty((0, self.num_heads), dtype=torch.float32),
                }
            indices = pos[valid]

        return {
            'token_ids': token_ids_tensor[indices].clone(),
            'k_qx': layer_store['k_qx'][indices].clone().to('cpu'),
            'k_scale': layer_store['k_scale'][indices].clone().to('cpu'),
            'v_qx': layer_store['v_qx'][indices].clone().to('cpu'),
            'v_scale': layer_store['v_scale'][indices].clone().to('cpu'),
        }

    def _dequantize_indices(
        self,
        layer_store: Dict[str, Any],
        indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized dequantization helper."""
        k_qx = layer_store['k_qx'][indices].to(torch.float32)
        v_qx = layer_store['v_qx'][indices].to(torch.float32)

        k_scale = layer_store['k_scale'][indices].unsqueeze(-1)
        v_scale = layer_store['v_scale'][indices].unsqueeze(-1)

        keys = k_qx * k_scale
        values = v_qx * v_scale

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
        num_cache_entries = sum(store['size'] for store in self.layer_store)
        
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
        self.layer_store = [self._init_layer_store() for _ in range(self.num_layers)]
        self.precision_map.clear()
        self.importance_tracker.reset()
        self.token_counter = 0
        self.realloc_counter = 0
        self.total_tokens_stored = 0
        self.global_step = 0
        self.last_seen.clear()
        self.head_importance = torch.ones(self.num_heads, dtype=torch.float32)
    
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
            f"cached={sum(store['size'] for store in self.layer_store)})"
        )
    
    def __len__(self) -> int:
        """Number of tokens with allocated precision."""
        return len(self.precision_map)
