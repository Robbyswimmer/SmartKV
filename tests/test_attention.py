"""
Unit tests for SmartKV attention layer.

Tests the SmartKVAttention and MultiQueryAttention classes with and without
SmartKV cache enabled (backward compatibility).
"""

import torch
import pytest
from smartkv.models.attention import SmartKVAttention, MultiQueryAttention
from smartkv.kernels import _quantized_attention_chunked, _quantized_attention_pytorch


class TestSmartKVAttentionBasic:
    """Test basic SmartKVAttention functionality."""
    
    def test_initialization(self):
        """Test basic initialization."""
        attn = SmartKVAttention(
            embed_dim=512,
            num_heads=8,
            layer_idx=0
        )
        
        assert attn.embed_dim == 512
        assert attn.num_heads == 8
        assert attn.head_dim == 64
        assert attn.layer_idx == 0
        assert attn.use_smartkv == False
    
    def test_forward_without_smartkv(self):
        """Test forward pass without SmartKV (standard attention)."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0,
            use_smartkv=False
        )
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 256)
        
        output, past_kv = attn(hidden_states, use_cache=False)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 256)
        assert past_kv is None
    
    def test_forward_with_cache(self):
        """Test forward pass with cache (for generation)."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        batch_size, seq_len = 1, 5
        hidden_states = torch.randn(batch_size, seq_len, 256)
        
        output, past_kv = attn(hidden_states, use_cache=True)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert past_kv is not None
        assert len(past_kv) == 2  # (keys, values)
    
    def test_forward_with_attention_mask(self):
        """Test forward with attention mask."""
        attn = SmartKVAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0
        )
        
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, 128)
        
        # Create causal mask
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        
        output, _ = attn(hidden_states, attention_mask=attention_mask)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_past_key_value_concatenation(self):
        """Test that past key-values are properly concatenated."""
        attn = SmartKVAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0
        )
        
        # First forward pass
        hidden_states_1 = torch.randn(1, 5, 128)
        output_1, past_kv = attn(hidden_states_1, use_cache=True)
        
        # Second forward pass with past
        hidden_states_2 = torch.randn(1, 3, 128)
        output_2, new_past_kv = attn(
            hidden_states_2,
            use_cache=True,
            past_key_value=past_kv
        )
        
        # Past KV should have grown
        assert new_past_kv[0].shape[2] == 8  # 5 + 3


class TestSmartKVAttentionWithCache:
    """Test SmartKVAttention with SmartKV cache enabled."""
    
    def test_enable_smartkv(self):
        """Test enabling SmartKV cache."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        attn.enable_smartkv(
            num_layers=12,
            memory_budget=0.5,
            decay=0.9
        )
        
        assert attn.use_smartkv == True
        assert attn.smartkv_cache is not None
    
    def test_disable_smartkv(self):
        """Test disabling SmartKV cache."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        attn.enable_smartkv(num_layers=12)
        attn.disable_smartkv()
        
        assert attn.use_smartkv == False
    
    def test_forward_with_smartkv(self):
        """Test forward pass with SmartKV enabled."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        attn.enable_smartkv(num_layers=12, memory_budget=0.5)
        
        batch_size, seq_len = 1, 10
        hidden_states = torch.randn(batch_size, seq_len, 256)
        token_ids = list(range(10))
        
        output, _ = attn(
            hidden_states,
            token_ids=token_ids,
            use_cache=False
        )
        
        assert output.shape == (batch_size, seq_len, 256)
    
    def test_attention_tracking(self):
        """Test that attention patterns are tracked."""
        attn = SmartKVAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0
        )
        
        attn.enable_smartkv(num_layers=4, memory_budget=0.5)
        
        hidden_states = torch.randn(1, 20, 128)
        token_ids = list(range(20))
        
        # Forward pass should track attention
        attn(hidden_states, token_ids=token_ids)
        
        # Check that importance is being tracked
        stats = attn.get_smartkv_stats()
        assert stats['num_tokens'] > 0
    
    def test_kv_storage_and_retrieval(self):
        """Test that KV are stored and can be retrieved."""
        attn = SmartKVAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0
        )
        
        attn.enable_smartkv(num_layers=4, memory_budget=0.5)
        
        hidden_states = torch.randn(1, 10, 128)
        token_ids = list(range(10))
        
        # Forward pass stores KV
        attn(hidden_states, token_ids=token_ids)
        
        # Retrieve stored KV
        keys, values = attn.retrieve_from_smartkv()
        
        assert keys.shape[0] == 10  # 10 tokens stored
        assert keys.shape[1] == 4   # 4 heads
        assert keys.shape[2] == 32  # head_dim


class TestMultiQueryAttention:
    """Test MultiQueryAttention variant."""
    
    def test_initialization(self):
        """Test MQA initialization."""
        mqa = MultiQueryAttention(
            embed_dim=512,
            num_heads=8,
            layer_idx=0,
            num_kv_heads=1
        )
        
        assert mqa.num_heads == 8
        assert mqa.num_kv_heads == 1
    
    def test_forward_mqa(self):
        """Test MQA forward pass."""
        mqa = MultiQueryAttention(
            embed_dim=256,
            num_heads=8,
            layer_idx=0,
            num_kv_heads=2
        )
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 256)
        
        output, _ = mqa(hidden_states)
        
        assert output.shape == (batch_size, seq_len, 256)
    
    def test_mqa_with_smartkv(self):
        """Test MQA with SmartKV enabled."""
        mqa = MultiQueryAttention(
            embed_dim=256,
            num_heads=8,
            layer_idx=0,
            num_kv_heads=1
        )
        
        mqa.enable_smartkv(num_layers=4, memory_budget=0.5)
        
        hidden_states = torch.randn(1, 15, 256)
        token_ids = list(range(15))
        
        output, _ = mqa(hidden_states, token_ids=token_ids)
        
        assert output.shape == (1, 15, 256)


class TestBackwardCompatibility:
    """Test backward compatibility (works without SmartKV)."""
    
    def test_standard_attention_mode(self):
        """Test that layer works in standard attention mode."""
        attn = SmartKVAttention(
            embed_dim=512,
            num_heads=8,
            layer_idx=0,
            use_smartkv=False
        )
        
        # Should work without token_ids
        hidden_states = torch.randn(2, 20, 512)
        output, _ = attn(hidden_states)
        
        assert output.shape == (2, 20, 512)
    
    def test_toggle_smartkv(self):
        """Test toggling SmartKV on and off."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        hidden_states = torch.randn(1, 10, 256)
        
        # Without SmartKV
        output_1, _ = attn(hidden_states)
        
        # Enable SmartKV
        attn.enable_smartkv(num_layers=4)
        output_2, _ = attn(hidden_states, token_ids=list(range(10)))
        
        # Disable SmartKV
        attn.disable_smartkv()
        output_3, _ = attn(hidden_states)
        
        # All should work
        assert output_1.shape == output_2.shape == output_3.shape
    
    def test_no_token_ids_without_smartkv(self):
        """Test that token_ids not required when SmartKV disabled."""
        attn = SmartKVAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0,
            use_smartkv=False
        )
        
        hidden_states = torch.randn(2, 15, 128)
        
        # Should work without token_ids when SmartKV is disabled
        output, _ = attn(hidden_states, token_ids=None)
        
        assert output.shape == (2, 15, 128)


class TestSmartKVStats:
    """Test SmartKV statistics and monitoring."""
    
    def test_get_stats(self):
        """Test getting SmartKV statistics."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        attn.enable_smartkv(num_layers=4, memory_budget=0.5)
        
        hidden_states = torch.randn(1, 20, 256)
        token_ids = list(range(20))
        
        attn(hidden_states, token_ids=token_ids)
        
        stats = attn.get_smartkv_stats()
        
        assert 'memory_budget' in stats
        assert 'num_tokens' in stats
        assert 'precision_distribution' in stats
    
    def test_reset_cache(self):
        """Test resetting cache."""
        attn = SmartKVAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0
        )
        
        attn.enable_smartkv(num_layers=4)
        
        hidden_states = torch.randn(1, 10, 128)
        token_ids = list(range(10))
        
        attn(hidden_states, token_ids=token_ids)
        
        # Cache should have data
        stats_before = attn.get_smartkv_stats()
        assert stats_before['num_tokens'] > 0
        
        # Reset
        attn.reset_cache()
        
        # Cache should be empty
        stats_after = attn.get_smartkv_stats()
        assert stats_after['num_tokens'] == 0


class TestAttentionShapes:
    """Test various input shapes and configurations."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [5, 10, 20])
    def test_various_batch_seq_sizes(self, batch_size, seq_len):
        """Test various batch and sequence lengths."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        hidden_states = torch.randn(batch_size, seq_len, 256)
        output, _ = attn(hidden_states)
        
        assert output.shape == (batch_size, seq_len, 256)
    
    @pytest.mark.parametrize("embed_dim,num_heads", [
        (128, 4),
        (256, 8),
        (512, 16),
        (1024, 32),
    ])
    def test_various_dimensions(self, embed_dim, num_heads):
        """Test various embedding dimensions and head counts."""
        attn = SmartKVAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layer_idx=0
        )
        
        hidden_states = torch.randn(2, 10, embed_dim)
        output, _ = attn(hidden_states)
        
        assert output.shape == (2, 10, embed_dim)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_token(self):
        """Test with single token."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0
        )
        
        hidden_states = torch.randn(1, 1, 256)
        output, _ = attn(hidden_states)
        
        assert output.shape == (1, 1, 256)
    
    def test_long_sequence(self):
        """Test with long sequence."""
        attn = SmartKVAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0
        )
        
        hidden_states = torch.randn(1, 1000, 128)
        output, _ = attn(hidden_states)
        
        assert output.shape == (1, 1000, 128)
    
    def test_dropout_training_mode(self):
        """Test dropout in training mode."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0,
            dropout=0.1
        )
        
        attn.train()
        
        hidden_states = torch.randn(2, 10, 256)
        output_1, _ = attn(hidden_states)
        output_2, _ = attn(hidden_states)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(output_1, output_2)
    
    def test_eval_mode(self):
        """Test in evaluation mode."""
        attn = SmartKVAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0,
            dropout=0.1
        )
        
        attn.eval()
        
        hidden_states = torch.randn(2, 10, 256)
        output_1, _ = attn(hidden_states)
        output_2, _ = attn(hidden_states)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output_1, output_2)


class TestIntegration:
    """Integration tests."""
    
    def test_multi_layer_workflow(self):
        """Test multi-layer workflow with SmartKV."""
        num_layers = 4
        layers = [
            SmartKVAttention(
                embed_dim=256,
                num_heads=4,
                layer_idx=i
            )
            for i in range(num_layers)
        ]
        
        # Enable SmartKV on all layers (shared cache)
        from smartkv.core.cache import SmartKVCache
        shared_cache = SmartKVCache(
            num_layers=num_layers,
            num_heads=4,
            head_dim=64,
            memory_budget=0.5
        )
        
        for layer in layers:
            layer.use_smartkv = True
            layer.smartkv_cache = shared_cache
        
        # Forward through all layers
        hidden_states = torch.randn(1, 20, 256)
        token_ids = list(range(20))
        
        for layer in layers:
            hidden_states, _ = layer(hidden_states, token_ids=token_ids)
        
        # Check final output
        assert hidden_states.shape == (1, 20, 256)
        
        # Check cache has data from all layers
        stats = shared_cache.get_memory_stats()
        assert stats['num_cache_entries'] > 0


class TestQuantizedAttentionFallbacks:
    """Tests for chunked quantized attention fallback."""

    def test_chunked_matches_pytorch_reference(self):
        """Chunked fallback should match full PyTorch attention."""
        torch.manual_seed(1234)

        batch, heads, q_len, k_len, head_dim = 1, 3, 2, 97, 16
        query = torch.randn(batch, heads, q_len, head_dim)
        key_int8 = torch.randint(-128, 127, (batch, heads, k_len, head_dim), dtype=torch.int8)
        value_int8 = torch.randint(-128, 127, (batch, heads, k_len, head_dim), dtype=torch.int8)
        key_scale = torch.rand(batch, heads, k_len) + 0.5
        value_scale = torch.rand(batch, heads, k_len) + 0.5

        attention_mask = torch.zeros(batch, 1, q_len, k_len)
        attention_mask[..., 40:] = float('-inf')

        reference = _quantized_attention_pytorch(
            query, key_int8, key_scale, value_int8, value_scale, attention_mask
        )
        chunked = _quantized_attention_chunked(
            query, key_int8, key_scale, value_int8, value_scale, attention_mask, chunk_size=17
        )

        assert torch.allclose(reference, chunked, rtol=1e-4, atol=1e-5)

    def test_chunked_handles_fully_masked_positions(self):
        """Chunked fallback should gracefully handle fully masked queries."""
        batch, heads, q_len, k_len, head_dim = 1, 2, 3, 65, 32
        query = torch.randn(batch, heads, q_len, head_dim)
        key_int8 = torch.randint(-128, 127, (batch, heads, k_len, head_dim), dtype=torch.int8)
        value_int8 = torch.randint(-128, 127, (batch, heads, k_len, head_dim), dtype=torch.int8)
        key_scale = torch.rand(batch, heads, k_len) + 0.25
        value_scale = torch.rand(batch, heads, k_len) + 0.25

        attention_mask = torch.zeros(batch, 1, q_len, k_len)
        attention_mask[:] = float('-inf')
        # Allow the first 10 keys for the first query to ensure mix of masked/unmasked
        attention_mask[:, :, 0, :10] = 0.0

        reference = _quantized_attention_pytorch(
            query, key_int8, key_scale, value_int8, value_scale, attention_mask
        )
        chunked = _quantized_attention_chunked(
            query, key_int8, key_scale, value_int8, value_scale, attention_mask, chunk_size=23
        )

        assert torch.allclose(reference, chunked, rtol=1e-4, atol=1e-5)
        # Fully masked queries should produce zeros (matching reference)
        masked_rows = attention_mask.squeeze(1).eq(float('-inf')).all(dim=-1)
        masked_positions = masked_rows.squeeze(0)
        assert torch.all(reference[:, :, masked_positions, :] == 0)
        assert torch.all(chunked[:, :, masked_positions, :] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
