"""
SmartKV-enabled Llama model.

Integrates SmartKV cache into Llama architecture by replacing standard
attention layers with SmartKVAttention.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    from transformers import LlamaForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaDecoderLayer,
        apply_rotary_pos_emb,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    LlamaForCausalLM = None
    LlamaConfig = None

from smartkv.core.cache import SmartKVCache
from smartkv.models.attention import SmartKVAttention


@dataclass
class SmartKVConfig:
    """Configuration for SmartKV cache."""
    
    enabled: bool = False
    memory_budget: float = 0.5
    decay: float = 0.9
    realloc_freq: int = 16
    available_bits: List[int] = None
    device: str = "cpu"
    
    def __post_init__(self):
        if self.available_bits is None:
            self.available_bits = [2, 3, 4, 8]


class LlamaSmartKVAttention(nn.Module):
    """
    Llama attention layer with SmartKV integration.
    
    Wraps LlamaAttention to use SmartKV cache when enabled.
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        layer_idx: int,
        smartkv_config: Optional[SmartKVConfig] = None
    ):
        """
        Initialize SmartKV-enabled Llama attention.
        
        Args:
            original_attention: Original Llama attention layer
            layer_idx: Layer index
            smartkv_config: SmartKV configuration
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for Llama integration")
        
        self.original_attention = original_attention
        self.layer_idx = layer_idx
        self.smartkv_config = smartkv_config or SmartKVConfig()

        # Copy attributes from original attention or config
        self.config = getattr(original_attention, 'config', None)

        # Get attributes safely
        self.hidden_size = getattr(original_attention, 'hidden_size',
                                   self.config.hidden_size if self.config else 2048)
        self.num_heads = getattr(original_attention, 'num_heads',
                                 self.config.num_attention_heads if self.config else 32)
        self.head_dim = getattr(original_attention, 'head_dim', 64)
        self.num_key_value_heads = getattr(original_attention, 'num_key_value_heads',
                                           self.config.num_key_value_heads if self.config else self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(original_attention, 'max_position_embeddings',
                                               self.config.max_position_embeddings if self.config else 2048)
        
        # Projections (share with original)
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        
        # RoPE
        if hasattr(original_attention, 'rotary_emb'):
            self.rotary_emb = original_attention.rotary_emb
        
        # SmartKV cache (created when enabled)
        self.smartkv_cache: Optional[SmartKVCache] = None
        self.use_smartkv = False
        self.token_counter = 0
    
    def enable_smartkv(self, shared_cache: Optional[SmartKVCache] = None):
        """
        Enable SmartKV cache.
        
        Args:
            shared_cache: Optional shared cache across layers
        """
        self.use_smartkv = True
        if shared_cache is not None:
            self.smartkv_cache = shared_cache
        # Cache will be created by model if not provided
    
    def disable_smartkv(self):
        """Disable SmartKV cache."""
        self.use_smartkv = False
        if self.smartkv_cache is not None:
            self.smartkv_cache.clear()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_ids: Optional[List[int]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with optional SmartKV.

        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Past KV cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cache
            token_ids: Token IDs for SmartKV tracking

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        # DEBUG
        debug = False  # Enable selectively for detailed tracing
        if debug:
            print(f"\n[DEBUG Layer {self.layer_idx}] Forward called")
            print(f"  use_smartkv: {self.use_smartkv}")
            print(f"  smartkv_cache: {self.smartkv_cache is not None}")
            print(f"  hidden_states shape: {hidden_states.shape}")
            print(f"  use_cache: {use_cache}")

        # If SmartKV not enabled, use original attention
        if not self.use_smartkv or self.smartkv_cache is None:
            if debug:
                print(f"  -> Using ORIGINAL attention")
            return self.original_attention(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )

        if debug:
            print(f"  -> Using SMARTKV attention")
        
        # Normalize interface differences across transformers versions
        if past_key_value is None and 'past_key_values' in kwargs:
            past_key_value = kwargs.pop('past_key_values')

        # SmartKV-enabled forward pass
        bsz, q_len, _ = hidden_states.size()

        if debug:
            print(f"  batch_size: {bsz}, query_len: {q_len}")

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if debug:
            print(f"  query_states: {query_states.shape}")
            print(f"  key_states: {key_states.shape}")
            print(f"  value_states: {value_states.shape}")
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings (RoPE)
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states,
            key_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )

        # For SmartKV: ALWAYS retrieve past KV from our cache (ignore use_cache parameter)
        # because transformers sets use_cache=False but we manage our own cache
        past_k = None
        past_v = None
        if debug:
            print(f"  k_cache size before retrieval: {len(self.smartkv_cache.k_cache)}")
        if len(self.smartkv_cache.k_cache) > 0:
            # Retrieve all cached KV for this layer
            past_k, past_v = self._retrieve_kv_smartkv(bsz)
            if past_k is not None and debug:
                print(f"  ✓ Retrieved past KV from SmartKV cache")
                print(f"  past_k shape: {past_k.shape}")
            elif past_k is None and debug:
                print(f"  ✗ Retrieval returned None")

        # Handle past KV from SmartKV cache
        if past_k is not None and past_v is not None:
            if debug:
                print(f"  Concatenating past KV: past={past_k.shape[-2]}, new={key_states.shape[-2]}")
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        # Save unrepeated KV states for storage (BEFORE repeat operation)
        # Only save the NEW tokens, not the concatenated past
        seq_offset = past_k.shape[-2] if past_k is not None else 0
        key_states_unrepeated = key_states[:, :, seq_offset:, :]  # Only new tokens
        value_states_unrepeated = value_states[:, :, seq_offset:, :]

        # Repeat KV for grouped-query attention
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        kv_len = attn_weights.shape[-1]
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            if debug:
                print(f"  attention_mask shape: {attention_mask.shape}")
        else:
            causal_mask = self._build_causal_mask(
                batch_size=bsz,
                query_len=q_len,
                key_len=kv_len,
                past_len=seq_offset,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
            attn_weights = attn_weights + causal_mask
            if debug:
                print(f"  applied causal mask shape: {causal_mask.shape}")

        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Track attention with SmartKV
        # Generate token IDs if not provided
        if token_ids is None:
            token_ids = list(range(self.token_counter, self.token_counter + q_len))
            self.token_counter += q_len

        if debug:
            print(f"  token_ids: {token_ids}")
            print(f"  attn_weights shape: {attn_weights.shape}")

        if len(token_ids) > 0:
            self.smartkv_cache.update_attention(
                self.layer_idx,
                attn_weights,
                token_ids
            )

            if debug:
                print(f"  Attention updated in cache")
                print(f"  Cache k_cache size: {len(self.smartkv_cache.k_cache)}")

            # Store KV with adaptive precision (use unrepeated states)
            self._store_kv_smartkv(key_states_unrepeated, value_states_unrepeated, token_ids, bsz)

            if debug:
                print(f"  After store - k_cache size: {len(self.smartkv_cache.k_cache)}")
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        # Prepare outputs - match Llama format
        # Llama decoder layer expects: (hidden_states, self_attn_weights)
        # The decoder layer signature is: hidden_states, _ = self.self_attn(...)
        if output_attentions:
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def _store_kv_smartkv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        token_ids: List[int],
        batch_size: int
    ):
        """Store KV with SmartKV cache."""
        debug = False  # Disable debug logging
        seq_len = key_states.shape[2]

        if debug:
            print(f"  [_store_kv_smartkv] seq_len: {seq_len}, num_token_ids: {len(token_ids)}")
            print(f"  [_store_kv_smartkv] key_states shape: {key_states.shape}")

        stored_count = 0
        for token_idx, token_id in enumerate(token_ids):
            if token_idx < seq_len and batch_size > 0:
                k_token = key_states[0, :, token_idx, :]  # [num_heads, head_dim]
                v_token = value_states[0, :, token_idx, :]

                if debug and token_idx < 2:  # Only print first 2
                    print(f"    Storing token_idx={token_idx}, token_id={token_id}")
                    print(f"    k_token shape: {k_token.shape}")

                self.smartkv_cache.quantize_and_store(
                    self.layer_idx,
                    token_id,
                    k_token,
                    v_token
                )
                stored_count += 1

        if debug:
            print(f"  [_store_kv_smartkv] Stored {stored_count} tokens")

    def _retrieve_kv_smartkv(self, batch_size: int):
        """
        Retrieve and dequantize all cached KV values for this layer.

        Returns:
            Tuple of (past_k, past_v) tensors or (None, None) if no cache
        """
        debug = self.layer_idx == 0  # Enable debug for layer 0

        # Get all token IDs stored in cache
        if not self.smartkv_cache.k_cache:
            return None, None

        # Collect all tokens for this layer
        # Cache keys are tuples: (layer_idx, token_id)
        layer_token_ids = []
        for cache_key in self.smartkv_cache.k_cache.keys():
            layer_idx, token_id = cache_key
            if layer_idx == self.layer_idx:
                layer_token_ids.append(token_id)

        if not layer_token_ids:
            return None, None

        # Sort token IDs to maintain order
        layer_token_ids.sort()

        if debug:
            print(f"  [_retrieve_kv_smartkv] Retrieving {len(layer_token_ids)} tokens for layer {self.layer_idx}")

        # Retrieve and dequantize each token
        k_tensors = []
        v_tensors = []

        for i, token_id in enumerate(layer_token_ids):
            k_token, v_token = self.smartkv_cache.retrieve(
                self.layer_idx,
                token_id
            )

            if k_token is not None and v_token is not None:
                if debug and i < 2:  # Print first 2 tokens
                    print(f"    Retrieved token_id={token_id}: k_token shape={k_token.shape}, mean={k_token.mean():.4f}, std={k_token.std():.4f}")
                k_tensors.append(k_token.unsqueeze(1))  # [num_heads, 1, head_dim]
                v_tensors.append(v_token.unsqueeze(1))

        if not k_tensors:
            return None, None

        # Concatenate along sequence dimension
        past_k = torch.cat(k_tensors, dim=1)  # [num_heads, seq_len, head_dim]
        past_v = torch.cat(v_tensors, dim=1)

        # Reshape to match expected format: [batch_size, num_heads, seq_len, head_dim]
        past_k = past_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
        past_v = past_v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if debug:
            print(f"  [_retrieve_kv_smartkv] Retrieved past_k shape: {past_k.shape}")

        return past_k, past_v

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for grouped-query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def _apply_rotary_pos_emb(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings (RoPE) in a version-agnostic way."""

        seq_len = query_states.shape[-2]

        # Preferred path: use rotary embedding module if available to compute using local position ids
        rotary_module = getattr(self, 'rotary_emb', None)
        if rotary_module is not None:
            position_ids = self._resolve_position_ids(seq_len, query_states.device, position_ids)
            cos, sin = rotary_module(key_states, position_ids)
            return apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids,
            )

        # Fallback path when only precomputed embeddings are provided
        if position_embeddings is not None:
            cos, sin = position_embeddings

            if cos.shape[1] != seq_len:
                cos = cos[:, -seq_len:, :]
                sin = sin[:, -seq_len:, :]

            return apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids,
            )

        # As a last resort, leave states unchanged
        return query_states, key_states

    def _resolve_position_ids(
        self,
        seq_len: int,
        device: torch.device,
        position_ids: Optional[torch.LongTensor]
    ) -> torch.LongTensor:
        """Resolve absolute position ids for the current token window."""
        if position_ids is not None and position_ids.size(-1) == seq_len:
            return position_ids

        start = self.token_counter
        return torch.arange(start, start + seq_len, device=device).unsqueeze(0)

    def _build_causal_mask(
        self,
        batch_size: int,
        query_len: int,
        key_len: int,
        past_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Construct a standard causal mask when none is provided by transformers."""
        if query_len == 0 or key_len == 0:
            return torch.zeros((batch_size, 1, query_len, key_len), device=device, dtype=dtype)

        mask_value = torch.finfo(dtype).min
        key_positions = torch.arange(key_len, device=device)
        allowed_positions = past_len + torch.arange(query_len, device=device)
        causal = key_positions.unsqueeze(0) <= allowed_positions.unsqueeze(1)

        mask = torch.full((query_len, key_len), mask_value, device=device, dtype=dtype)
        mask = mask.masked_fill(causal, 0.0)

        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_len, key_len)


class LlamaSmartKV(nn.Module):
    """
    Llama model with SmartKV cache integration.
    
    Wraps a pretrained Llama model and enables SmartKV on attention layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        smartkv_config: Optional[SmartKVConfig] = None
    ):
        """
        Initialize SmartKV-enabled Llama.
        
        Args:
            model: Pretrained Llama model (from transformers)
            smartkv_config: SmartKV configuration
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.model = model
        self.smartkv_config = smartkv_config or SmartKVConfig()
        self.smartkv_cache: Optional[SmartKVCache] = None
        
        # Replace attention layers
        if self.smartkv_config.enabled:
            self._replace_attention_layers()
    
    def _replace_attention_layers(self):
        """Replace standard attention layers with SmartKV attention."""
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            raise ValueError("Model structure not compatible with SmartKV")

        layers = self.model.model.layers
        num_layers = len(layers)

        # Create shared SmartKV cache
        if layers and hasattr(layers[0], 'self_attn'):
            first_attn = layers[0].self_attn

            # Get num_heads and head_dim from config or attention module
            if hasattr(first_attn, 'num_heads'):
                num_heads = first_attn.num_heads
            elif hasattr(self.model, 'config'):
                num_heads = self.model.config.num_attention_heads
            else:
                raise ValueError("Cannot determine num_heads")

            if hasattr(first_attn, 'head_dim'):
                head_dim = first_attn.head_dim
            elif hasattr(self.model, 'config'):
                head_dim = self.model.config.hidden_size // num_heads
            else:
                raise ValueError("Cannot determine head_dim")

            self.smartkv_cache = SmartKVCache(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                memory_budget=self.smartkv_config.memory_budget,
                decay=self.smartkv_config.decay,
                realloc_freq=self.smartkv_config.realloc_freq,
                available_bits=self.smartkv_config.available_bits,
                device=self.smartkv_config.device
            )
        
        # Replace attention in each layer
        shared_rotary_emb = getattr(self.model.model, 'rotary_emb', None)

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                original_attn = layer.self_attn
                smartkv_attn = LlamaSmartKVAttention(
                    original_attn,
                    layer_idx,
                    self.smartkv_config
                )
                if getattr(smartkv_attn, 'rotary_emb', None) is None and shared_rotary_emb is not None:
                    smartkv_attn.rotary_emb = shared_rotary_emb
                smartkv_attn.enable_smartkv(self.smartkv_cache)
                layer.self_attn = smartkv_attn
    
    def enable_smartkv(
        self,
        memory_budget: Optional[float] = None,
        decay: Optional[float] = None,
        realloc_freq: Optional[int] = None
    ):
        """
        Enable SmartKV cache.
        
        Args:
            memory_budget: Memory budget (fraction of FP16)
            decay: EMA decay factor
            realloc_freq: Reallocation frequency
        """
        if memory_budget is not None:
            self.smartkv_config.memory_budget = memory_budget
        if decay is not None:
            self.smartkv_config.decay = decay
        if realloc_freq is not None:
            self.smartkv_config.realloc_freq = realloc_freq
        
        self.smartkv_config.enabled = True
        self._replace_attention_layers()
    
    def disable_smartkv(self):
        """Disable SmartKV cache."""
        self.smartkv_config.enabled = False
        
        # Restore original attention layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                if isinstance(layer.self_attn, LlamaSmartKVAttention):
                    layer.self_attn = layer.self_attn.original_attention
    
    def forward(self, *args, **kwargs):
        """Forward pass through model."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text."""
        return self.model.generate(*args, **kwargs)
    
    def get_smartkv_stats(self) -> Dict[str, Any]:
        """Get SmartKV cache statistics."""
        if self.smartkv_cache is None:
            return {}
        return self.smartkv_cache.get_memory_stats()
    
    def reset_cache(self):
        """Reset SmartKV cache and token counters."""
        if self.smartkv_cache is not None:
            self.smartkv_cache.clear()

        # Reset token counters in all attention layers
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'token_counter'):
                layer.self_attn.token_counter = 0
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def load_llama_with_smartkv(
    model_name: str,
    smartkv_config: Optional[SmartKVConfig] = None,
    **model_kwargs
) -> LlamaSmartKV:
    """
    Load a Llama model with SmartKV enabled.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
        smartkv_config: SmartKV configuration
        **model_kwargs: Additional arguments for model loading
    
    Returns:
        LlamaSmartKV model
    
    Example:
        >>> model = load_llama_with_smartkv(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     smartkv_config=SmartKVConfig(
        ...         enabled=True,
        ...         memory_budget=0.5,
        ...         decay=0.9
        ...     ),
        ...     torch_dtype=torch.float16,
        ...     device_map="auto"
        ... )
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library required")
    
    # Load base model
    base_model = LlamaForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Wrap with SmartKV
    smartkv_model = LlamaSmartKV(base_model, smartkv_config)
    
    return smartkv_model
