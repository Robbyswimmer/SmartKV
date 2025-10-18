"""
SmartKV-enabled attention layer.

Modified attention mechanism that integrates SmartKV cache for
adaptive precision KV-cache compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from smartkv.core.cache import SmartKVCache


class SmartKVAttention(nn.Module):
    """
    Multi-head attention with SmartKV cache support.
    
    Supports both standard attention (without SmartKV) and adaptive precision
    attention (with SmartKV enabled). Backward compatible with standard attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_idx: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_smartkv: bool = False,
        smartkv_config: Optional[dict] = None
    ):
        """
        Initialize SmartKV attention layer.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            layer_idx: Layer index (for cache management)
            dropout: Dropout probability
            bias: Whether to use bias in projections
            use_smartkv: Whether to enable SmartKV cache
            smartkv_config: Configuration for SmartKV cache
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.use_smartkv = use_smartkv
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # SmartKV cache (created when first needed)
        self.smartkv_cache: Optional[SmartKVCache] = None
        self.smartkv_config = smartkv_config or {}
        
        # Token counter for tracking
        self.token_counter = 0
    
    def enable_smartkv(
        self,
        num_layers: int,
        memory_budget: float = 0.5,
        decay: float = 0.9,
        realloc_freq: int = 16,
        available_bits: list = [2, 3, 4, 8],
        device: str = "cpu"
    ):
        """
        Enable SmartKV cache.
        
        Args:
            num_layers: Total number of layers in model
            memory_budget: Fraction of FP16 memory to use
            decay: EMA decay for importance tracking
            realloc_freq: Reallocate precision every N tokens
            available_bits: Available bit-widths
            device: Device to store cache on
        """
        self.use_smartkv = True
        cfg_obj = self.smartkv_config or {}

        def cfg_get(key, default=None):
            if isinstance(cfg_obj, dict):
                return cfg_obj.get(key, default)
            return getattr(cfg_obj, key, default)

        forecast_kwargs = {}
        if cfg_get('enable_forecast', False):
            forecast_kwargs = {
                'enable_forecast': True,
                'forecast_history': cfg_get('forecast_history', 2048),
                'forecast_update_interval': cfg_get('forecast_update_interval', 32),
                'forecast_blend': cfg_get('forecast_blend', 0.5),
                'forecast_lr': cfg_get('forecast_lr', 0.05),
            }
        self.smartkv_cache = SmartKVCache(
            num_layers=num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            memory_budget=memory_budget,
            decay=decay,
            realloc_freq=realloc_freq,
            available_bits=available_bits,
            device=device,
            **forecast_kwargs,
        )
    
    def disable_smartkv(self):
        """Disable SmartKV cache and use standard attention."""
        self.use_smartkv = False
        if self.smartkv_cache is not None:
            self.smartkv_cache.clear()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[list] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional SmartKV cache.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask [batch, seq_len, seq_len]
            token_ids: Token IDs for cache management (required if use_smartkv)
            use_cache: Whether to use/return cache
            past_key_value: Past key-value cache (for generation)
        
        Returns:
            Tuple of (output, past_key_value)
            - output: [batch, seq_len, embed_dim]
            - past_key_value: Optional cached KV tensors
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)  # [batch, seq_len, embed_dim]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, head_dim]
        
        # Handle past key-values (for generation)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # Concatenate along seq_len
            v = torch.cat([past_v, v], dim=2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # Shape: [batch, num_heads, seq_len_q, seq_len_k]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # SmartKV: Track attention and store KV with adaptive precision
        if self.use_smartkv and self.smartkv_cache is not None and token_ids is not None:
            self._track_and_store_smartkv(
                attn_weights=attn_weights,
                k=k,
                v=v,
                token_ids=token_ids
            )
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: [batch, num_heads, seq_len, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Prepare cache for next step
        if use_cache:
            past_key_value = (k, v)
        else:
            past_key_value = None
        
        return output, past_key_value
    
    def _track_and_store_smartkv(
        self,
        attn_weights: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_ids: list
    ):
        """
        Track attention patterns and store KV with SmartKV cache.
        
        Args:
            attn_weights: Attention weights [batch, heads, queries, keys]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            token_ids: Token IDs for each position
        """
        # Update attention tracking
        self.smartkv_cache.update_attention(
            self.layer_idx,
            attn_weights,
            token_ids
        )
        
        # Store each token's KV with allocated precision
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        for token_idx, token_id in enumerate(token_ids):
            # Extract KV for this token across all heads
            k_token = k[:, :, token_idx, :]  # [batch, heads, head_dim]
            v_token = v[:, :, token_idx, :]
            
            # For now, store first batch item (extend for batched inference later)
            if batch_size > 0:
                k_token = k_token[0]  # [heads, head_dim]
                v_token = v_token[0]
                
                # Quantize and store
                self.smartkv_cache.quantize_and_store(
                    self.layer_idx,
                    token_id,
                    k_token,
                    v_token
                )
        
        self.token_counter += len(token_ids)
    
    def retrieve_from_smartkv(
        self,
        token_ids: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve KV from SmartKV cache.
        
        Args:
            token_ids: Optional list of specific token IDs to retrieve
        
        Returns:
            Tuple of (keys, values)
            Shape: [num_tokens, num_heads, head_dim]
        """
        if self.smartkv_cache is None:
            raise RuntimeError("SmartKV cache not initialized")
        
        keys, values = self.smartkv_cache.retrieve_all(
            self.layer_idx,
            token_ids
        )
        
        return keys, values
    
    def get_smartkv_stats(self) -> dict:
        """Get SmartKV cache statistics."""
        if self.smartkv_cache is None:
            return {}
        return self.smartkv_cache.get_memory_stats()
    
    def reset_cache(self):
        """Reset SmartKV cache."""
        if self.smartkv_cache is not None:
            self.smartkv_cache.clear()
        self.token_counter = 0
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"layer_idx={self.layer_idx}, "
            f"use_smartkv={self.use_smartkv}"
        )


class MultiQueryAttention(SmartKVAttention):
    """
    Multi-Query Attention with SmartKV support.
    
    Similar to SmartKVAttention but with shared key-value heads (MQA).
    Used in some efficient transformer variants.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_idx: int,
        num_kv_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        use_smartkv: bool = False,
        smartkv_config: Optional[dict] = None
    ):
        """
        Initialize Multi-Query Attention.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of query heads
            layer_idx: Layer index
            num_kv_heads: Number of key-value heads (typically 1 for MQA)
            dropout: Dropout probability
            bias: Whether to use bias
            use_smartkv: Whether to enable SmartKV
            smartkv_config: SmartKV configuration
        """
        # Initialize parent with query heads
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            dropout=dropout,
            bias=bias,
            use_smartkv=use_smartkv,
            smartkv_config=smartkv_config
        )
        
        self.num_kv_heads = num_kv_heads
        self.kv_head_dim = embed_dim // num_kv_heads
        
        # Override K, V projections for fewer heads
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[list] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with MQA and optional SmartKV.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            token_ids: Token IDs for cache
            use_cache: Whether to use cache
            past_key_value: Past KV cache
        
        Returns:
            Tuple of (output, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project Q with full heads, K/V with fewer heads
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand K, V to match Q heads (repeat for each query head group)
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Handle past KV
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Compute attention (same as parent)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # SmartKV tracking
        if self.use_smartkv and self.smartkv_cache is not None and token_ids is not None:
            self._track_and_store_smartkv(attn_weights, k, v, token_ids)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        if use_cache:
            past_key_value = (k, v)
        else:
            past_key_value = None
        
        return output, past_key_value
