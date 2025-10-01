"""
Uniform quantization baselines for KV-cache compression.

Implements uniform INT8, INT4, and FP16 baselines for comparison with SmartKV.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from smartkv.core.quantizers import EightbitQuantizer, FourbitQuantizer


@dataclass
class UniformQuantConfig:
    """Configuration for uniform quantization baseline."""
    
    bits: int = 8  # Quantization bits (4, 8, or 16 for FP16)
    enabled: bool = True
    device: str = "cpu"


class UniformQuantCache:
    """
    Uniform quantization KV-cache.
    
    Applies the same quantization level to all tokens, unlike SmartKV's
    adaptive approach. Used as a baseline for comparison.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        bits: int = 8,
        device: str = "cpu"
    ):
        """
        Initialize uniform quantization cache.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            bits: Quantization bits (4, 8, or 16 for FP16)
            device: Device to store cache
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        
        # Select quantizer
        if bits == 8:
            self.quantizer = EightbitQuantizer()
        elif bits == 4:
            self.quantizer = FourbitQuantizer()
        elif bits == 16:
            self.quantizer = None  # FP16, no quantization
        else:
            raise ValueError(f"Unsupported bits: {bits}. Use 4, 8, or 16.")
        
        # Cache storage: (layer, token_id) -> quantized/fp16 data
        self.k_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.v_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        # Metadata
        self.total_tokens_stored = 0
    
    def quantize_and_store(
        self,
        layer_idx: int,
        token_id: int,
        k_vec: torch.Tensor,
        v_vec: torch.Tensor
    ) -> None:
        """
        Quantize and store key-value vectors.
        
        Args:
            layer_idx: Layer index
            token_id: Token ID
            k_vec: Key vector [num_heads, head_dim]
            v_vec: Value vector [num_heads, head_dim]
        """
        # Move to CPU for storage
        if k_vec.device.type != "cpu":
            k_vec = k_vec.cpu()
        if v_vec.device.type != "cpu":
            v_vec = v_vec.cpu()
        
        # Quantize or store as FP16
        if self.quantizer is not None:
            k_quant = self.quantizer.quantize(k_vec)
            v_quant = self.quantizer.quantize(v_vec)
        else:
            # FP16 baseline - just convert to FP16
            k_quant = {'data': k_vec.half()}
            v_quant = {'data': v_vec.half()}
        
        # Store
        cache_key = (layer_idx, token_id)
        self.k_cache[cache_key] = k_quant
        self.v_cache[cache_key] = v_quant
        
        self.total_tokens_stored += 1
    
    def retrieve(
        self,
        layer_idx: int,
        token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve and dequantize key-value vectors.
        
        Args:
            layer_idx: Layer index
            token_id: Token ID
        
        Returns:
            Tuple of (key, value) tensors
        """
        cache_key = (layer_idx, token_id)
        
        if cache_key not in self.k_cache:
            raise KeyError(f"Token {token_id} not found in layer {layer_idx}")
        
        k_quant = self.k_cache[cache_key]
        v_quant = self.v_cache[cache_key]
        
        # Dequantize or convert from FP16
        if self.quantizer is not None:
            k = self.quantizer.dequantize(k_quant)
            v = self.quantizer.dequantize(v_quant)
        else:
            # FP16 baseline - convert back to float32
            k = k_quant['data'].float()
            v = v_quant['data'].float()
        
        # Move to device if needed
        if self.device != "cpu":
            k = k.to(self.device)
            v = v.to(self.device)
        
        return k, v
    
    def retrieve_all(
        self,
        layer_idx: int,
        token_ids: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve all cached KV for a layer.
        
        Args:
            layer_idx: Layer index
            token_ids: Optional list of specific token IDs
        
        Returns:
            Tuple of (keys, values) tensors
        """
        if token_ids is None:
            token_ids = sorted([
                tid for (l, tid) in self.k_cache.keys() if l == layer_idx
            ])
        
        if not token_ids:
            return torch.empty(0), torch.empty(0)
        
        keys = []
        values = []
        
        for token_id in token_ids:
            try:
                k, v = self.retrieve(layer_idx, token_id)
                keys.append(k)
                values.append(v)
            except KeyError:
                continue
        
        if not keys:
            return torch.empty(0), torch.empty(0)
        
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        
        return keys, values
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with memory statistics
        """
        num_tokens = len(set(tid for (_, tid) in self.k_cache.keys()))
        num_cache_entries = len(self.k_cache)
        
        # Calculate memory usage
        if self.bits == 16:
            # FP16: 2 bytes per value
            bytes_per_token = self.num_heads * self.head_dim * 2 * 2  # K and V
            memory_ratio = 1.0  # Baseline
        else:
            # Quantized
            bytes_per_token = self.num_heads * self.head_dim * self.bits / 8 * 2
            fp16_bytes = self.num_heads * self.head_dim * 2 * 2
            memory_ratio = bytes_per_token / fp16_bytes
        
        return {
            'bits': self.bits,
            'num_tokens': num_tokens,
            'num_cache_entries': num_cache_entries,
            'total_tokens_stored': self.total_tokens_stored,
            'memory_ratio': memory_ratio,
            'quantizer': 'fp16' if self.bits == 16 else f'int{self.bits}',
        }
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.k_cache.clear()
        self.v_cache.clear()
        self.total_tokens_stored = 0
    
    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UniformQuantCache(layers={self.num_layers}, "
            f"bits={self.bits}, "
            f"cached={len(self.k_cache)})"
        )
    
    def __len__(self) -> int:
        """Number of tokens cached."""
        return len(set(tid for (_, tid) in self.k_cache.keys()))


class UniformQuantAttention(nn.Module):
    """
    Attention layer with uniform quantization cache.
    
    Similar to SmartKVAttention but uses uniform quantization across all tokens.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_idx: int,
        bits: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize uniform quantization attention.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            layer_idx: Layer index
            bits: Quantization bits (4, 8, or 16)
            dropout: Dropout probability
            bias: Whether to use bias
            device: Device for cache
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.bits = bits
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Uniform quantization cache
        self.cache: Optional[UniformQuantCache] = None
        self.use_cache_quant = False
        self.device = device
    
    def enable_cache(
        self,
        shared_cache: Optional[UniformQuantCache] = None
    ):
        """Enable uniform quantization cache."""
        self.use_cache_quant = True
        if shared_cache is not None:
            self.cache = shared_cache
    
    def disable_cache(self):
        """Disable cache."""
        self.use_cache_quant = False
        if self.cache is not None:
            self.cache.clear()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[list] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional uniform quantization cache.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            token_ids: Token IDs (for cache storage)
            use_cache: Whether to use cache
            past_key_value: Past key-value cache
        
        Returns:
            Tuple of (output, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key-values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Store in quantized cache if enabled
        if self.use_cache_quant and self.cache is not None and token_ids is not None:
            for token_idx, token_id in enumerate(token_ids):
                if token_idx < seq_len and batch_size > 0:
                    k_token = k[0, :, token_idx, :]
                    v_token = v[0, :, token_idx, :]
                    self.cache.quantize_and_store(
                        self.layer_idx,
                        token_id,
                        k_token,
                        v_token
                    )
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attn_output)
        
        if use_cache:
            past_key_value = (k, v)
        else:
            past_key_value = None
        
        return output, past_key_value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache is None:
            return {}
        return self.cache.get_memory_stats()
    
    def reset_cache(self):
        """Reset cache."""
        if self.cache is not None:
            self.cache.clear()


def create_uniform_baseline(bits: int = 8, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a uniform quantization baseline configuration.
    
    Args:
        bits: Quantization bits (4, 8, or 16)
        name: Optional name for the baseline
    
    Returns:
        Configuration dict for the baseline
    """
    if name is None:
        if bits == 16:
            name = "FP16-Baseline"
        else:
            name = f"Uniform-INT{bits}"
    
    return {
        'name': name,
        'bits': bits,
        'type': 'uniform',
        'cache_class': UniformQuantCache,
        'attention_class': UniformQuantAttention,
        'config': UniformQuantConfig(bits=bits),
    }
