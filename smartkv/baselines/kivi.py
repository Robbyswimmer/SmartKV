"""
KIVI (Key-Value Quantization with Importance-based Variance) baseline.

Reproduction of the KIVI method for comparison:
- Keys: 2-bit residual quantization per channel
- Values: Per-token quantization (typically 4-bit)

Reference: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from smartkv.core.quantizers import TwobitQuantizer, FourbitQuantizer


@dataclass
class KIVIConfig:
    """Configuration for KIVI baseline."""
    
    key_bits: int = 2  # Bits for key quantization
    value_bits: int = 4  # Bits for value quantization
    enabled: bool = True
    device: str = "cpu"
    per_channel_keys: bool = True  # Per-channel quantization for keys


class KIVICache:
    """
    KIVI cache implementation.
    
    Implements asymmetric quantization:
    - Keys: 2-bit per-channel quantization
    - Values: 4-bit per-token quantization
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        key_bits: int = 2,
        value_bits: int = 4,
        per_channel_keys: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize KIVI cache.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            key_bits: Bits for key quantization (typically 2)
            value_bits: Bits for value quantization (typically 4)
            per_channel_keys: Use per-channel quantization for keys
            device: Device to store cache
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.per_channel_keys = per_channel_keys
        self.device = device
        
        # Quantizers
        if key_bits == 2:
            self.key_quantizer = TwobitQuantizer()
        elif key_bits == 4:
            self.key_quantizer = FourbitQuantizer()
        else:
            raise ValueError(f"Unsupported key_bits: {key_bits}")
        
        if value_bits == 2:
            self.value_quantizer = TwobitQuantizer()
        elif value_bits == 4:
            self.value_quantizer = FourbitQuantizer()
        else:
            raise ValueError(f"Unsupported value_bits: {value_bits}")
        
        # Cache storage
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
        Quantize and store key-value vectors with KIVI method.
        
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
        
        # KIVI: Keys use per-channel quantization
        if self.per_channel_keys:
            # Quantize each head dimension separately for keys
            k_quant = self._quantize_per_channel(k_vec, self.key_quantizer)
        else:
            k_quant = self.key_quantizer.quantize(k_vec)
        
        # Values use standard per-token quantization
        v_quant = self.value_quantizer.quantize(v_vec)
        
        # Store
        cache_key = (layer_idx, token_id)
        self.k_cache[cache_key] = k_quant
        self.v_cache[cache_key] = v_quant
        
        self.total_tokens_stored += 1
    
    def _quantize_per_channel(
        self,
        tensor: torch.Tensor,
        quantizer
    ) -> Dict[str, Any]:
        """
        Per-channel quantization (quantize each dimension separately).
        
        Args:
            tensor: Input tensor [num_heads, head_dim]
            quantizer: Quantizer to use
        
        Returns:
            Dict with quantized data
        """
        num_heads, head_dim = tensor.shape
        
        # Store per-channel quantized data
        quantized_channels = []
        scales = []
        zero_points = []
        
        for ch in range(head_dim):
            # Extract channel across all heads
            channel_data = tensor[:, ch]  # [num_heads]
            
            # Quantize this channel
            q_data = quantizer.quantize(channel_data)
            quantized_channels.append(q_data['qx'])
            scales.append(q_data['scale'])
            zero_points.append(q_data.get('zero_point', torch.zeros_like(q_data['scale'])))
        
        return {
            'data': quantized_channels,  # List of quantized channels
            'scale': scales,
            'zero_point': zero_points,
            'per_channel': True,
            'shape': tensor.shape
        }
    
    def _dequantize_per_channel(
        self,
        quant_data: Dict[str, Any],
        quantizer
    ) -> torch.Tensor:
        """
        Dequantize per-channel data.
        
        Args:
            quant_data: Quantized data dict
            quantizer: Quantizer to use
        
        Returns:
            Dequantized tensor
        """
        num_heads, head_dim = quant_data['shape']
        
        # Reconstruct tensor
        channels = []
        for ch in range(head_dim):
            ch_data = {
                'qx': quant_data['data'][ch],
                'scale': quant_data['scale'][ch],
                'zero_point': quant_data['zero_point'][ch]
            }
            dequant_channel = quantizer.dequantize(ch_data)
            channels.append(dequant_channel)
        
        # Stack channels back
        result = torch.stack(channels, dim=1)  # [num_heads, head_dim]
        return result
    
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
        
        # Dequantize
        if k_quant.get('per_channel', False):
            k = self._dequantize_per_channel(k_quant, self.key_quantizer)
        else:
            k = self.key_quantizer.dequantize(k_quant)
        
        v = self.value_quantizer.dequantize(v_quant)
        
        # Move to device
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
        # Keys: key_bits per value
        # Values: value_bits per value
        key_bytes = self.num_heads * self.head_dim * self.key_bits / 8
        value_bytes = self.num_heads * self.head_dim * self.value_bits / 8
        total_bytes = key_bytes + value_bytes
        
        # FP16 baseline
        fp16_bytes = self.num_heads * self.head_dim * 2 * 2  # K and V
        memory_ratio = total_bytes / fp16_bytes
        
        return {
            'key_bits': self.key_bits,
            'value_bits': self.value_bits,
            'num_tokens': num_tokens,
            'num_cache_entries': num_cache_entries,
            'total_tokens_stored': self.total_tokens_stored,
            'memory_ratio': memory_ratio,
            'per_channel_keys': self.per_channel_keys,
            'quantizer': 'kivi',
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
            f"KIVICache(layers={self.num_layers}, "
            f"key_bits={self.key_bits}, "
            f"value_bits={self.value_bits}, "
            f"cached={len(self.k_cache)})"
        )
    
    def __len__(self) -> int:
        """Number of tokens cached."""
        return len(set(tid for (_, tid) in self.k_cache.keys()))


class KIVIAttention(nn.Module):
    """
    Attention layer with KIVI cache.
    
    Uses asymmetric quantization for keys and values.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_idx: int,
        key_bits: int = 2,
        value_bits: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize KIVI attention.
        
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            layer_idx: Layer index
            key_bits: Bits for key quantization
            value_bits: Bits for value quantization
            dropout: Dropout probability
            bias: Whether to use bias
            device: Device for cache
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # KIVI cache
        self.cache: Optional[KIVICache] = None
        self.use_cache_quant = False
        self.device = device
    
    def enable_cache(
        self,
        shared_cache: Optional[KIVICache] = None
    ):
        """Enable KIVI cache."""
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
        Forward pass with KIVI cache.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            token_ids: Token IDs for cache
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
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past KV
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Store in KIVI cache
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


def create_kivi_baseline(
    key_bits: int = 2,
    value_bits: int = 4,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a KIVI baseline configuration.
    
    Args:
        key_bits: Bits for key quantization
        value_bits: Bits for value quantization
        name: Optional name for the baseline
    
    Returns:
        Configuration dict for the baseline
    """
    if name is None:
        name = f"KIVI-K{key_bits}V{value_bits}"
    
    return {
        'name': name,
        'key_bits': key_bits,
        'value_bits': value_bits,
        'type': 'kivi',
        'cache_class': KIVICache,
        'attention_class': KIVIAttention,
        'config': KIVIConfig(key_bits=key_bits, value_bits=value_bits),
    }
