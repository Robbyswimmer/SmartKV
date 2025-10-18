"""
Unit tests for SmartKV cache implementation.

Tests the main SmartKVCache class that integrates quantizers, importance
tracking, and precision allocation.
"""

import torch
import pytest
from smartkv.core.cache import SmartKVCache


class TestSmartKVCacheInitialization:
    """Test cache initialization."""
    
    def test_basic_init(self):
        """Test basic initialization."""
        cache = SmartKVCache(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            memory_budget=0.5
        )
        
        assert cache.num_layers == 12
        assert cache.num_heads == 8
        assert cache.head_dim == 64
        assert cache.memory_budget == 0.5
        assert cache.decay == 0.9
        assert cache.realloc_freq == 16
    
    def test_custom_config(self):
        """Test initialization with custom config."""
        cache = SmartKVCache(
            num_layers=24,
            num_heads=16,
            head_dim=128,
            memory_budget=0.3,
            decay=0.95,
            realloc_freq=32,
            available_bits=[2, 4, 8]
        )
        
        assert cache.decay == 0.95
        assert cache.realloc_freq == 32
        assert cache.available_bits == [8, 4, 2]  # Sorted descending
    
    def test_quantizers_initialized(self):
        """Test that quantizers are initialized."""
        cache = SmartKVCache(
            num_layers=4,
            num_heads=4,
            head_dim=32
        )
        
        # Should have quantizers for all bit-widths
        assert 2 in cache.quantizers
        assert 3 in cache.quantizers
        assert 4 in cache.quantizers
        assert 8 in cache.quantizers


class TestAttentionUpdating:
    """Test attention updating and importance tracking."""
    
    def test_update_attention(self):
        """Test updating attention scores."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Create synthetic attention weights
        attention_weights = torch.rand(1, 4, 10, 10)
        token_ids = list(range(10))
        
        cache.update_attention(0, attention_weights, token_ids)
        
        # Check that importance is tracked
        assert cache.get_importance(0) > 0
    
    def test_periodic_reallocation(self):
        """Test that precision is reallocated periodically."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            realloc_freq=5
        )
        
        token_ids = list(range(10))
        
        # First update should trigger reallocation (counter starts at 0)
        attention_weights = torch.rand(1, 4, 10, 10)
        cache.update_attention(0, attention_weights, token_ids)
        
        assert cache.realloc_counter == 1
        assert len(cache.precision_map) > 0


class TestQuantizeAndStore:
    """Test quantization and storage."""
    
    def test_store_single_token(self):
        """Test storing a single token."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Allocate precision first
        cache.precision_map[0] = 8
        
        # Create KV vectors
        k = torch.randn(4, 32)  # [num_heads, head_dim]
        v = torch.randn(4, 32)
        
        cache.quantize_and_store(0, 0, k, v)
        
        # Check stored
        assert (0, 0) in cache.k_cache
        assert (0, 0) in cache.v_cache
        assert cache.total_tokens_stored == 1
    
    def test_store_multiple_tokens(self):
        """Test storing multiple tokens."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Store 10 tokens in layer 0
        for token_id in range(10):
            cache.precision_map[token_id] = 4
            k = torch.randn(4, 32)
            v = torch.randn(4, 32)
            cache.quantize_and_store(0, token_id, k, v)
        
        assert cache.total_tokens_stored == 10
        assert len(cache.k_cache) == 10
    
    def test_store_across_layers(self):
        """Test storing tokens across multiple layers."""
        cache = SmartKVCache(
            num_layers=3,
            num_heads=4,
            head_dim=32
        )
        
        # Store same token in different layers
        cache.precision_map[0] = 8
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        
        for layer in range(3):
            cache.quantize_and_store(layer, 0, k, v)
        
        assert cache.total_tokens_stored == 3
        assert (0, 0) in cache.k_cache
        assert (1, 0) in cache.k_cache
        assert (2, 0) in cache.k_cache


class TestRetrieve:
    """Test retrieval and dequantization."""
    
    def test_retrieve_single_token(self):
        """Test retrieving a single token."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Store
        cache.precision_map[0] = 8
        k_orig = torch.randn(4, 32)
        v_orig = torch.randn(4, 32)
        cache.quantize_and_store(0, 0, k_orig, v_orig)
        
        # Retrieve
        k_retrieved, v_retrieved = cache.retrieve(0, 0)
        
        # Check shapes match
        assert k_retrieved.shape == k_orig.shape
        assert v_retrieved.shape == v_orig.shape
        
        # Check values are approximately correct (allow quantization error)
        assert torch.allclose(k_retrieved, k_orig, rtol=0.1, atol=0.1)
    
    def test_retrieve_nonexistent_token(self):
        """Test that retrieving non-existent token raises error."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        with pytest.raises(KeyError):
            cache.retrieve(0, 999)
    
    def test_retrieve_all_tokens(self):
        """Test retrieving all tokens in a layer."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Store 5 tokens
        for token_id in range(5):
            cache.precision_map[token_id] = 4
            k = torch.randn(4, 32)
            v = torch.randn(4, 32)
            cache.quantize_and_store(0, token_id, k, v)
        
        # Retrieve all
        keys, values = cache.retrieve_all(0)
        
        assert keys.shape == (5, 4, 32)
        assert values.shape == (5, 4, 32)
    
    def test_retrieve_all_empty_layer(self):
        """Test retrieving from empty layer."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        keys, values = cache.retrieve_all(0)
        
        assert keys.numel() == 0
        assert values.numel() == 0
    
    def test_retrieve_specific_tokens(self):
        """Test retrieving specific tokens."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Store 10 tokens
        for token_id in range(10):
            cache.precision_map[token_id] = 4
            k = torch.randn(4, 32)
            v = torch.randn(4, 32)
            cache.quantize_and_store(0, token_id, k, v)
        
        # Retrieve only tokens 2, 5, 7
        keys, values = cache.retrieve_all(0, token_ids=[2, 5, 7])
        
        assert keys.shape == (3, 4, 32)
        assert values.shape == (3, 4, 32)


class TestPrecisionAllocation:
    """Test precision allocation."""
    
    def test_allocate_precision(self):
        """Test precision allocation based on importance."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            memory_budget=0.5
        )
        
        # Simulate attention with non-uniform importance
        token_ids = list(range(20))
        attention_weights = torch.zeros(1, 4, 10, 20)
        
        # Token 0 gets most attention
        attention_weights[:, :, :, 0] = 0.5
        # Others get less
        attention_weights[:, :, :, 1:] = 0.5 / 19
        
        cache.update_attention(0, attention_weights, token_ids)
        
        # Token 0 should have higher precision
        precision_0 = cache.get_precision(0)
        precision_10 = cache.get_precision(10)
        
        assert precision_0 >= precision_10
    
    def test_precision_map_updated(self):
        """Test that precision map is updated."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        token_ids = list(range(10))
        cache.allocate_precision(0, token_ids)
        
        # All tokens should have precision allocated
        assert len(cache.precision_map) == 10


class TestForecasting:
    """Tests for forecast-guided precision allocation."""

    def test_forecast_predictor_updates(self):
        """Ensure forecast predictor receives training samples and updates."""
        cache = SmartKVCache(
            num_layers=1,
            num_heads=2,
            head_dim=16,
            memory_budget=0.5,
            enable_forecast=True,
            forecast_history=32,
            forecast_update_interval=1,
            forecast_blend=0.5,
        )

        token_ids = list(range(4, 8))
        torch.manual_seed(0)

        # Initial attention update to seed importance
        attn_weights = torch.rand(1, 2, 4, len(token_ids))
        cache.update_attention(0, attn_weights, token_ids)

        # Allocate precision to log features
        cache.allocate_precision(0, token_ids)

        # Second update provides targets for predictor
        attn_weights_2 = torch.rand(1, 2, 4, len(token_ids))
        cache.update_attention(0, attn_weights_2, token_ids)

        assert cache.enable_forecast is True
        assert cache.forecast_predictor is not None
        assert cache.forecast_last_loss is not None
        # Forecast should enqueue fresh pending features for the next step
        assert len(cache.forecast_pending) == len(token_ids)


class TestMemoryStats:
    """Test memory statistics."""
    
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            memory_budget=0.5
        )
        
        # Store some tokens
        for token_id in range(10):
            cache.precision_map[token_id] = 4
            k = torch.randn(4, 32)
            v = torch.randn(4, 32)
            cache.quantize_and_store(0, token_id, k, v)
        
        stats = cache.get_memory_stats()
        
        assert 'memory_budget' in stats
        assert 'memory_ratio' in stats
        assert 'num_tokens' in stats
        assert 'num_cache_entries' in stats
        assert stats['num_tokens'] == 10
        assert stats['num_cache_entries'] == 10
    
    def test_precision_distribution(self):
        """Test precision distribution in stats."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Manually set mixed precision
        cache.precision_map = {
            0: 8, 1: 8, 2: 8,  # 3 tokens at 8-bit
            3: 4, 4: 4,        # 2 tokens at 4-bit
            5: 2,              # 1 token at 2-bit
        }
        
        stats = cache.get_memory_stats()
        dist = stats['precision_distribution']
        
        assert dist['8-bit'] == 3
        assert dist['4-bit'] == 2
        assert dist['2-bit'] == 1


class TestCacheOperations:
    """Test cache operations."""
    
    def test_clear(self):
        """Test clearing cache."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Add some data
        for token_id in range(5):
            cache.precision_map[token_id] = 4
            k = torch.randn(4, 32)
            v = torch.randn(4, 32)
            cache.quantize_and_store(0, token_id, k, v)
        
        cache.clear()
        
        assert len(cache.k_cache) == 0
        assert len(cache.v_cache) == 0
        assert len(cache.precision_map) == 0
        assert cache.token_counter == 0
    
    def test_get_top_k_tokens(self):
        """Test getting top-k important tokens."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Manually set importance scores
        cache.importance_tracker.token_importance = {
            0: 10.0,
            1: 5.0,
            2: 8.0,
            3: 2.0,
        }
        
        top_2 = cache.get_top_k_important_tokens(2)
        
        assert len(top_2) == 2
        assert top_2[0][0] == 0  # Token 0 has highest importance
        assert top_2[1][0] == 2  # Token 2 has second highest
    
    def test_get_bottom_k_tokens(self):
        """Test getting bottom-k important tokens."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        cache.importance_tracker.token_importance = {
            0: 10.0,
            1: 5.0,
            2: 8.0,
            3: 2.0,
        }
        
        bottom_2 = cache.get_bottom_k_important_tokens(2)
        
        assert len(bottom_2) == 2
        assert bottom_2[0][0] == 3  # Token 3 has lowest importance


class TestExportState:
    """Test state export for checkpointing."""
    
    def test_export_state(self):
        """Test exporting cache state."""
        cache = SmartKVCache(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            memory_budget=0.5,
            decay=0.95
        )
        
        state = cache.export_state()
        
        assert 'config' in state
        assert 'precision_map' in state
        assert state['config']['num_layers'] == 12
        assert state['config']['memory_budget'] == 0.5
        assert state['config']['decay'] == 0.95


class TestStringRepresentation:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__."""
        cache = SmartKVCache(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            memory_budget=0.5
        )
        
        repr_str = repr(cache)
        
        assert 'SmartKVCache' in repr_str
        assert '12' in repr_str  # num_layers
        assert '50' in repr_str  # budget (50% or 50.0%)
    
    def test_len(self):
        """Test __len__."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        cache.precision_map = {0: 8, 1: 4, 2: 2}
        
        assert len(cache) == 3


class TestIntegration:
    """Integration tests combining multiple operations."""
    
    def test_full_workflow(self):
        """Test complete workflow: update -> allocate -> store -> retrieve."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            memory_budget=0.5,
            realloc_freq=1  # Reallocate every update
        )
        
        # Step 1: Update attention
        token_ids = list(range(10))
        attention_weights = torch.rand(1, 4, 10, 10)
        cache.update_attention(0, attention_weights, token_ids)
        
        # Step 2: Store tokens
        for token_id in token_ids:
            k = torch.randn(4, 32)
            v = torch.randn(4, 32)
            cache.quantize_and_store(0, token_id, k, v)
        
        # Step 3: Retrieve tokens
        keys, values = cache.retrieve_all(0)
        
        assert keys.shape[0] == 10
        assert values.shape[0] == 10
        
        # Step 4: Check memory stats
        stats = cache.get_memory_stats()
        assert stats['num_tokens'] == 10
        assert stats['memory_ratio'] <= 0.55  # Should respect budget
    
    def test_multiple_layers(self):
        """Test working with multiple layers."""
        cache = SmartKVCache(
            num_layers=4,
            num_heads=4,
            head_dim=32,
            memory_budget=0.5
        )
        
        token_ids = list(range(20))
        
        # Update attention and store for each layer
        for layer in range(4):
            attention_weights = torch.rand(1, 4, 20, 20)
            cache.update_attention(layer, attention_weights, token_ids)
            
            for token_id in token_ids:
                k = torch.randn(4, 32)
                v = torch.randn(4, 32)
                cache.quantize_and_store(layer, token_id, k, v)
        
        # Each layer should have 20 tokens
        for layer in range(4):
            keys, values = cache.retrieve_all(layer)
            assert keys.shape[0] == 20
    
    def test_adaptive_precision(self):
        """Test that precision adapts based on attention."""
        cache = SmartKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=32,
            memory_budget=0.4,
            realloc_freq=1
        )
        
        token_ids = list(range(50))
        
        # Create attention pattern where first 5 tokens get most attention
        attention_weights = torch.zeros(1, 4, 25, 50)
        attention_weights[:, :, :, :5] = 0.6 / 5  # First 5 tokens
        attention_weights[:, :, :, 5:] = 0.4 / 45  # Rest
        
        cache.update_attention(0, attention_weights, token_ids)
        
        # First tokens should have higher precision
        avg_precision_first_5 = sum(cache.get_precision(i) for i in range(5)) / 5
        avg_precision_last_5 = sum(cache.get_precision(i) for i in range(45, 50)) / 5
        
        assert avg_precision_first_5 >= avg_precision_last_5


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_cache_operations(self):
        """Test operations on empty cache."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Get stats on empty cache
        stats = cache.get_memory_stats()
        assert stats['num_tokens'] == 0
        
        # Retrieve from empty layer
        keys, values = cache.retrieve_all(0)
        assert keys.numel() == 0
    
    def test_single_token(self):
        """Test with single token."""
        cache = SmartKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=32
        )
        
        cache.precision_map[0] = 8
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        
        cache.quantize_and_store(0, 0, k, v)
        k_ret, v_ret = cache.retrieve(0, 0)
        
        assert k_ret.shape == k.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
