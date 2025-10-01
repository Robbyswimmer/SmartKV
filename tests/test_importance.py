"""
Unit tests for importance tracking system.

Tests attention score accumulation, EMA updates, layer-wise tracking,
and analysis utilities.
"""

import torch
import pytest
import numpy as np
from smartkv.core.importance import (
    ImportanceTracker,
    AttentionAnalyzer,
    compute_ema_importance,
    aggregate_layer_importance,
)


class TestImportanceTracker:
    """Test ImportanceTracker functionality."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ImportanceTracker(num_layers=32, decay=0.9)
        
        assert tracker.num_layers == 32
        assert tracker.decay == 0.9
        assert len(tracker.token_importance) == 0
        assert len(tracker.layer_importance) == 0
        assert tracker.update_count == 0
    
    def test_update_attention_basic(self):
        """Test basic attention update."""
        tracker = ImportanceTracker(num_layers=2, decay=0.9)
        
        # Create synthetic attention weights: [batch, heads, queries, keys]
        attention_weights = torch.rand(1, 8, 10, 10)
        token_ids = list(range(10))
        
        tracker.update_attention(0, attention_weights, token_ids)
        
        # Verify tokens have importance scores
        assert len(tracker.token_importance) == 10
        assert all(tracker.get_importance(tid) > 0 for tid in token_ids)
        assert tracker.update_count == 1
    
    def test_update_attention_3d_tensor(self):
        """Test attention update with 3D tensor (no batch dimension)."""
        tracker = ImportanceTracker(num_layers=2, decay=0.9)
        
        # [heads, queries, keys]
        attention_weights = torch.rand(8, 10, 10)
        token_ids = list(range(10))
        
        tracker.update_attention(0, attention_weights, token_ids)
        
        assert len(tracker.token_importance) == 10
        assert tracker.update_count == 1
    
    def test_cumulative_importance_across_layers(self):
        """Test that importance accumulates across layers."""
        tracker = ImportanceTracker(num_layers=4, decay=0.9)
        
        attention_weights = torch.ones(1, 8, 5, 5) * 0.2  # Uniform attention
        token_ids = list(range(5))
        
        # Update across multiple layers
        for layer in range(4):
            tracker.update_attention(layer, attention_weights, token_ids)
        
        # Importance should accumulate (be higher than single layer)
        # Each layer adds score, so cumulative should be ~4x single layer
        for tid in token_ids:
            importance = tracker.get_importance(tid)
            assert importance > 0
    
    def test_ema_temporal_smoothing(self):
        """Test that EMA smooths importance over time."""
        tracker = ImportanceTracker(num_layers=1, decay=0.9)
        
        token_ids = [0, 1, 2]
        
        # First update: high attention on token 0
        attn1 = torch.zeros(1, 1, 3, 3)
        attn1[:, :, :, 0] = 1.0  # All attention to token 0
        tracker.update_attention(0, attn1, token_ids)
        
        initial_importance_0 = tracker.get_layer_importance(0, 0)
        
        # Second update: high attention on token 1
        attn2 = torch.zeros(1, 1, 3, 3)
        attn2[:, :, :, 1] = 1.0  # All attention to token 1
        tracker.update_attention(0, attn2, token_ids)
        
        # Token 0's layer importance should decay (EMA effect)
        updated_importance_0 = tracker.get_layer_importance(0, 0)
        assert updated_importance_0 < initial_importance_0
    
    def test_layer_specific_importance(self):
        """Test layer-specific importance tracking."""
        tracker = ImportanceTracker(num_layers=3, decay=0.9)
        
        token_ids = [0, 1]
        
        # Different attention in different layers
        attn_layer0 = torch.zeros(1, 1, 2, 2)
        attn_layer0[:, :, :, 0] = 1.0  # Layer 0: attention to token 0
        
        attn_layer1 = torch.zeros(1, 1, 2, 2)
        attn_layer1[:, :, :, 1] = 1.0  # Layer 1: attention to token 1
        
        tracker.update_attention(0, attn_layer0, token_ids)
        tracker.update_attention(1, attn_layer1, token_ids)
        
        # Check layer-specific scores
        layer0_token0 = tracker.get_layer_importance(0, 0)
        layer0_token1 = tracker.get_layer_importance(0, 1)
        layer1_token0 = tracker.get_layer_importance(1, 0)
        layer1_token1 = tracker.get_layer_importance(1, 1)
        
        assert layer0_token0 > layer0_token1  # Token 0 has more attention in layer 0
        assert layer1_token1 > layer1_token0  # Token 1 has more attention in layer 1
    
    def test_get_top_k_tokens(self):
        """Test getting top-k tokens by importance."""
        tracker = ImportanceTracker(num_layers=1, decay=0.9)
        
        # Create attention with known distribution
        token_ids = list(range(10))
        attention_weights = torch.zeros(1, 1, 10, 10)
        
        # Give decreasing attention to tokens 0, 1, 2, ...
        for i in range(10):
            attention_weights[:, :, :, i] = 10 - i
        
        tracker.update_attention(0, attention_weights, token_ids)
        
        top_5 = tracker.get_top_k_tokens(5)
        
        assert len(top_5) == 5
        # Top tokens should be 0, 1, 2, 3, 4
        assert top_5[0][0] == 0  # Token 0 has highest importance
        assert top_5[4][0] == 4  # Token 4 is 5th
        
        # Check scores are descending
        for i in range(len(top_5) - 1):
            assert top_5[i][1] >= top_5[i + 1][1]
    
    def test_get_bottom_k_tokens(self):
        """Test getting bottom-k tokens by importance."""
        tracker = ImportanceTracker(num_layers=1, decay=0.9)
        
        token_ids = list(range(10))
        attention_weights = torch.zeros(1, 1, 10, 10)
        
        for i in range(10):
            attention_weights[:, :, :, i] = 10 - i
        
        tracker.update_attention(0, attention_weights, token_ids)
        
        bottom_5 = tracker.get_bottom_k_tokens(5)
        
        assert len(bottom_5) == 5
        # Bottom tokens should be 9, 8, 7, 6, 5
        assert bottom_5[0][0] == 9  # Token 9 has lowest importance
        
        # Check scores are ascending
        for i in range(len(bottom_5) - 1):
            assert bottom_5[i][1] <= bottom_5[i + 1][1]
    
    def test_get_importance_statistics(self):
        """Test importance statistics computation."""
        tracker = ImportanceTracker(num_layers=1, decay=0.9)
        
        token_ids = list(range(100))
        attention_weights = torch.rand(1, 8, 50, 100)
        
        tracker.update_attention(0, attention_weights, token_ids)
        
        stats = tracker.get_importance_statistics()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'total' in stats
        assert 'num_tokens' in stats
        
        assert stats['num_tokens'] == 100
        assert stats['min'] >= 0
        assert stats['max'] >= stats['min']
        assert stats['total'] > 0
    
    def test_get_layer_statistics(self):
        """Test layer-specific statistics."""
        tracker = ImportanceTracker(num_layers=3, decay=0.9)
        
        token_ids = list(range(50))
        attention_weights = torch.rand(1, 8, 25, 50)
        
        tracker.update_attention(0, attention_weights, token_ids)
        tracker.update_attention(1, attention_weights, token_ids)
        
        layer0_stats = tracker.get_layer_statistics(0)
        layer1_stats = tracker.get_layer_statistics(1)
        layer2_stats = tracker.get_layer_statistics(2)  # No updates
        
        assert layer0_stats['num_tokens'] == 50
        assert layer1_stats['num_tokens'] == 50
        assert layer2_stats['num_tokens'] == 0
        assert layer2_stats['mean'] == 0.0
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = ImportanceTracker(num_layers=2, decay=0.9)
        
        attention_weights = torch.rand(1, 8, 10, 10)
        token_ids = list(range(10))
        
        tracker.update_attention(0, attention_weights, token_ids)
        
        assert len(tracker.token_importance) > 0
        assert tracker.update_count > 0
        
        tracker.reset()
        
        assert len(tracker.token_importance) == 0
        assert len(tracker.layer_importance) == 0
        assert tracker.update_count == 0
    
    def test_snapshot(self):
        """Test taking snapshots of importance."""
        tracker = ImportanceTracker(num_layers=1, decay=0.9)
        
        attention_weights = torch.rand(1, 8, 10, 10)
        token_ids = list(range(10))
        
        tracker.update_attention(0, attention_weights, token_ids)
        
        snapshot = tracker.snapshot()
        
        assert isinstance(snapshot, dict)
        assert len(snapshot) == 10
        
        # Snapshot should be a copy, not a reference
        snapshot[0] = 999.0
        assert tracker.get_importance(0) != 999.0
    
    def test_invalid_layer_index(self):
        """Test that invalid layer index raises error."""
        tracker = ImportanceTracker(num_layers=2, decay=0.9)
        
        attention_weights = torch.rand(1, 8, 10, 10)
        token_ids = list(range(10))
        
        with pytest.raises(ValueError):
            tracker.update_attention(-1, attention_weights, token_ids)
        
        with pytest.raises(ValueError):
            tracker.update_attention(2, attention_weights, token_ids)
    
    def test_invalid_attention_shape(self):
        """Test that invalid attention shape raises error."""
        tracker = ImportanceTracker(num_layers=2, decay=0.9)
        
        # 2D tensor should fail
        attention_weights = torch.rand(10, 10)
        token_ids = list(range(10))
        
        with pytest.raises(ValueError):
            tracker.update_attention(0, attention_weights, token_ids)
    
    def test_empty_token_list(self):
        """Test handling of empty token list."""
        tracker = ImportanceTracker(num_layers=2, decay=0.9)
        
        attention_weights = torch.rand(1, 8, 10, 10)
        token_ids = []
        
        # Should not crash
        tracker.update_attention(0, attention_weights, token_ids)
        assert len(tracker.token_importance) == 0


class TestAttentionAnalyzer:
    """Test AttentionAnalyzer functionality."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = AttentionAnalyzer()
        assert len(analyzer.attention_logs) == 0
    
    def test_log_attention(self):
        """Test logging attention."""
        analyzer = AttentionAnalyzer()
        
        attention_weights = torch.rand(1, 8, 10, 10)
        token_ids = list(range(10))
        tokens = [f"token_{i}" for i in range(10)]
        
        analyzer.log_attention(0, attention_weights, token_ids, tokens)
        
        assert len(analyzer.attention_logs) == 1
        log = analyzer.attention_logs[0]
        assert log['layer'] == 0
        assert log['token_ids'] == token_ids
        assert log['tokens'] == tokens
    
    def test_compute_attention_entropy(self):
        """Test attention entropy computation."""
        analyzer = AttentionAnalyzer()
        
        # Uniform attention (high entropy)
        uniform_attn = torch.ones(1, 8, 10, 10) / 10
        uniform_entropy = analyzer.compute_attention_entropy(uniform_attn)
        
        # Focused attention (low entropy)
        focused_attn = torch.zeros(1, 8, 10, 10)
        focused_attn[:, :, :, 0] = 1.0  # All attention to first token
        focused_entropy = analyzer.compute_attention_entropy(focused_attn)
        
        # Uniform should have higher entropy than focused
        assert uniform_entropy > focused_entropy
        assert focused_entropy < 0.1  # Should be near zero
    
    def test_identify_attention_patterns(self):
        """Test pattern identification."""
        analyzer = AttentionAnalyzer()
        
        # Log attention for multiple layers
        for layer in range(3):
            attention_weights = torch.rand(1, 8, 10, 10)
            token_ids = list(range(10))
            analyzer.log_attention(layer, attention_weights, token_ids)
        
        patterns = analyzer.identify_attention_patterns()
        
        assert patterns['total_logs'] == 3
        assert len(patterns['layers']) == 3
        assert 'avg_entropy_per_layer' in patterns
    
    def test_get_high_attention_tokens(self):
        """Test identifying high attention tokens."""
        analyzer = AttentionAnalyzer()
        
        # Create attention where token 0 gets high attention
        attention_weights = torch.zeros(1, 1, 10, 10)
        attention_weights[:, :, :, 0] = 0.5  # Token 0 gets 0.5
        attention_weights[:, :, :, 1:] = 0.05  # Others get 0.05
        
        token_ids = list(range(10))
        analyzer.log_attention(0, attention_weights, token_ids)
        
        high_attn_tokens = analyzer.get_high_attention_tokens(threshold=0.1)
        
        assert 0 in high_attn_tokens  # Token 0 should be identified
        assert len(high_attn_tokens) >= 1
    
    def test_clear_logs(self):
        """Test clearing logs."""
        analyzer = AttentionAnalyzer()
        
        attention_weights = torch.rand(1, 8, 10, 10)
        token_ids = list(range(10))
        
        analyzer.log_attention(0, attention_weights, token_ids)
        assert len(analyzer.attention_logs) == 1
        
        analyzer.clear_logs()
        assert len(analyzer.attention_logs) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_ema_importance(self):
        """Test EMA computation."""
        current = {0: 10.0, 1: 5.0, 2: 3.0}
        new = {0: 2.0, 1: 8.0, 3: 5.0}  # Token 3 is new
        
        updated = compute_ema_importance(current, new, decay=0.9)
        
        # Token 0: 0.9 * 10 + 0.1 * 2 = 9.2
        assert abs(updated[0] - 9.2) < 0.01
        
        # Token 1: 0.9 * 5 + 0.1 * 8 = 5.3
        assert abs(updated[1] - 5.3) < 0.01
        
        # Token 2: no new score, should stay same
        assert updated[2] == 3.0
        
        # Token 3: new token, should be 5.0
        assert updated[3] == 5.0
    
    def test_aggregate_layer_importance(self):
        """Test aggregating layer importance to token level."""
        layer_importance = {
            (0, 0): 5.0,
            (0, 1): 3.0,
            (1, 0): 7.0,  # Token 0 in layer 1
            (1, 1): 2.0,
            (2, 0): 4.0,
        }
        
        aggregated = aggregate_layer_importance(layer_importance, num_layers=3)
        
        # Token 0: 5 + 7 + 4 = 16
        assert aggregated[0] == 16.0
        
        # Token 1: 3 + 2 = 5
        assert aggregated[1] == 5.0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_tracker_with_real_attention_pattern(self):
        """Test tracker with realistic attention patterns."""
        tracker = ImportanceTracker(num_layers=12, decay=0.9)
        
        seq_len = 50
        num_heads = 8
        
        # Simulate causal attention where queries only attend to previous tokens
        # This creates asymmetric pattern where early tokens get more attention
        for layer in range(12):
            attention_weights = torch.zeros(1, num_heads, seq_len, seq_len)
            
            # Causal mask: query at position q can only attend to positions <= q
            for q in range(seq_len):
                for k in range(q + 1):  # Only k <= q (causal)
                    # Attention decays with distance, but earlier tokens are visible to more queries
                    dist = q - k
                    attention_weights[0, :, q, k] = 1.0 / (1.0 + dist * 0.1)
            
            # Normalize each query's attention
            for q in range(seq_len):
                attn_sum = attention_weights[0, :, q, :].sum()
                if attn_sum > 0:
                    attention_weights[0, :, q, :] = attention_weights[0, :, q, :] / attn_sum
            
            token_ids = list(range(seq_len))
            tracker.update_attention(layer, attention_weights, token_ids)
        
        # Early tokens should have higher importance (they're attended to by more queries)
        early_importance = tracker.get_importance(0)
        mid_importance = tracker.get_importance(seq_len // 2)
        late_importance = tracker.get_importance(seq_len - 1)
        
        assert early_importance > mid_importance > late_importance
        
        # Top-k should include early tokens
        top_10 = tracker.get_top_k_tokens(10)
        top_token_ids = [tid for tid, _ in top_10]
        
        # Most top tokens should be from early positions
        early_tokens_in_top = sum(1 for tid in top_token_ids if tid < 10)
        assert early_tokens_in_top >= 7  # At least 7 of top 10 should be early tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
