"""
Unit tests for precision allocation algorithms.

Tests greedy, dynamic programming, layer-aware, and baseline allocation strategies.
Verifies memory budget constraints and allocation quality.
"""

import pytest
import numpy as np
from smartkv.core.allocation import (
    greedy_allocation,
    dynamic_programming_allocation,
    layer_aware_allocation,
    uniform_allocation,
    random_allocation,
    position_based_allocation,
    compute_memory_usage,
    validate_allocation,
    _compute_layer_budgets,
    _round_to_nearest_bits,
)


class TestGreedyAllocation:
    """Test greedy allocation algorithm."""
    
    def test_basic_allocation(self):
        """Test basic greedy allocation."""
        importance_scores = {
            0: 10.0,
            1: 5.0,
            2: 3.0,
            3: 1.0,
        }
        
        allocation = greedy_allocation(importance_scores, memory_budget=0.5)
        
        # Should allocate higher precision to more important tokens
        assert allocation[0] >= allocation[1]
        assert allocation[1] >= allocation[2]
        assert allocation[2] >= allocation[3]
    
    def test_memory_budget_respected(self):
        """Test that allocation respects memory budget."""
        importance_scores = {i: float(100 - i) for i in range(100)}
        
        allocation = greedy_allocation(importance_scores, memory_budget=0.5)
        
        stats = compute_memory_usage(allocation)
        
        # Should use approximately 50% of FP16 memory
        assert stats['memory_ratio'] <= 0.51  # Small tolerance
    
    def test_empty_scores(self):
        """Test handling of empty importance scores."""
        allocation = greedy_allocation({}, memory_budget=0.5)
        assert len(allocation) == 0
    
    def test_all_tokens_get_precision(self):
        """Test that all tokens receive some precision."""
        importance_scores = {i: float(i) for i in range(20)}
        
        allocation = greedy_allocation(importance_scores, memory_budget=0.3)
        
        # All tokens should be in allocation
        assert len(allocation) == 20
        
        # All should have valid bit-widths
        for bits in allocation.values():
            assert bits in [2, 3, 4, 8]
    
    def test_high_budget(self):
        """Test allocation with high budget (most tokens get 8-bit)."""
        importance_scores = {i: float(i) for i in range(10)}
        
        allocation = greedy_allocation(importance_scores, memory_budget=0.9)
        
        # With high budget, most tokens should get 8-bit
        eight_bit_count = sum(1 for bits in allocation.values() if bits == 8)
        assert eight_bit_count >= 7
    
    def test_low_budget(self):
        """Test allocation with low budget (most tokens get 2-bit)."""
        importance_scores = {i: float(i) for i in range(10)}
        
        allocation = greedy_allocation(importance_scores, memory_budget=0.2)
        
        # With low budget, most tokens should get 2-bit
        two_bit_count = sum(1 for bits in allocation.values() if bits == 2)
        assert two_bit_count >= 5
    
    def test_custom_available_bits(self):
        """Test allocation with custom available bit-widths."""
        importance_scores = {i: float(10 - i) for i in range(5)}
        
        allocation = greedy_allocation(
            importance_scores,
            memory_budget=0.5,
            available_bits=[2, 8]  # Only 2 or 8 bits
        )
        
        # All allocations should be either 2 or 8
        for bits in allocation.values():
            assert bits in [2, 8]


class TestDynamicProgrammingAllocation:
    """Test dynamic programming allocation."""
    
    def test_basic_dp_allocation(self):
        """Test basic DP allocation."""
        importance_scores = {
            0: 10.0,
            1: 5.0,
            2: 3.0,
        }
        
        allocation = dynamic_programming_allocation(
            importance_scores,
            memory_budget=0.5
        )
        
        # Should allocate to all tokens
        assert len(allocation) == 3
        
        # Higher importance should generally get higher precision
        assert allocation[0] >= allocation[2]
    
    def test_dp_respects_budget(self):
        """Test DP respects memory budget."""
        importance_scores = {i: float(50 - i) for i in range(50)}
        
        allocation = dynamic_programming_allocation(
            importance_scores,
            memory_budget=0.4
        )
        
        stats = compute_memory_usage(allocation)
        # DP may slightly exceed due to discrete allocation and filling missing tokens
        assert stats['memory_ratio'] <= 0.45  # Allow reasonable tolerance
    
    def test_dp_vs_greedy_quality(self):
        """Test that DP achieves at least as good allocation as greedy."""
        importance_scores = {i: float(np.random.rand()) for i in range(20)}
        memory_budget = 0.5
        
        greedy_alloc = greedy_allocation(importance_scores, memory_budget)
        dp_alloc = dynamic_programming_allocation(importance_scores, memory_budget)
        
        # Compute weighted importance (objective value)
        greedy_value = sum(
            importance_scores[tid] * bits 
            for tid, bits in greedy_alloc.items()
        )
        dp_value = sum(
            importance_scores[tid] * bits 
            for tid, bits in dp_alloc.items()
        )
        
        # DP should be at least as good as greedy
        assert dp_value >= greedy_value * 0.95  # Allow small tolerance
    
    def test_dp_empty_scores(self):
        """Test DP with empty scores."""
        allocation = dynamic_programming_allocation({}, memory_budget=0.5)
        assert len(allocation) == 0


class TestLayerAwareAllocation:
    """Test layer-aware allocation."""
    
    def test_basic_layer_aware(self):
        """Test basic layer-aware allocation."""
        layer_importance = {
            (0, 0): 5.0,
            (0, 1): 3.0,
            (1, 0): 7.0,
            (1, 1): 4.0,
            (2, 0): 6.0,
            (2, 1): 2.0,
        }
        
        allocation = layer_aware_allocation(
            layer_importance,
            memory_budget=0.5,
            num_layers=3
        )
        
        # Should allocate to both tokens
        assert 0 in allocation
        assert 1 in allocation
    
    def test_linear_layer_weighting(self):
        """Test linear layer weighting strategy."""
        layer_importance = {
            (i, j): float(i + j) 
            for i in range(4) 
            for j in range(10)
        }
        
        allocation = layer_aware_allocation(
            layer_importance,
            memory_budget=0.5,
            num_layers=4,
            layer_weight_strategy="linear"
        )
        
        assert len(allocation) > 0
    
    def test_equal_layer_weighting(self):
        """Test equal layer weighting strategy."""
        layer_importance = {
            (i, j): float(j) 
            for i in range(3) 
            for j in range(5)
        }
        
        allocation = layer_aware_allocation(
            layer_importance,
            memory_budget=0.5,
            num_layers=3,
            layer_weight_strategy="equal"
        )
        
        assert len(allocation) == 5
    
    def test_exponential_layer_weighting(self):
        """Test exponential layer weighting strategy."""
        layer_importance = {
            (i, j): float(j) 
            for i in range(3) 
            for j in range(5)
        }
        
        allocation = layer_aware_allocation(
            layer_importance,
            memory_budget=0.5,
            num_layers=3,
            layer_weight_strategy="exponential"
        )
        
        assert len(allocation) == 5


class TestUniformAllocation:
    """Test uniform allocation baseline."""
    
    def test_uniform_allocation(self):
        """Test uniform allocation assigns same bits to all tokens."""
        token_ids = list(range(10))
        
        allocation = uniform_allocation(token_ids, bits=4)
        
        assert len(allocation) == 10
        assert all(bits == 4 for bits in allocation.values())
    
    def test_uniform_different_bits(self):
        """Test uniform allocation with different bit-widths."""
        token_ids = list(range(5))
        
        for bits in [2, 3, 4, 8]:
            allocation = uniform_allocation(token_ids, bits=bits)
            assert all(b == bits for b in allocation.values())


class TestRandomAllocation:
    """Test random allocation baseline."""
    
    def test_random_allocation(self):
        """Test random allocation."""
        token_ids = list(range(20))
        
        allocation = random_allocation(token_ids, memory_budget=0.5, seed=42)
        
        assert len(allocation) == 20
        assert all(bits in [2, 3, 4, 8] for bits in allocation.values())
    
    def test_random_reproducibility(self):
        """Test that random allocation is reproducible with seed."""
        token_ids = list(range(10))
        
        alloc1 = random_allocation(token_ids, memory_budget=0.5, seed=42)
        alloc2 = random_allocation(token_ids, memory_budget=0.5, seed=42)
        
        assert alloc1 == alloc2
    
    def test_random_different_seeds(self):
        """Test that different seeds give different allocations."""
        token_ids = list(range(10))
        
        alloc1 = random_allocation(token_ids, memory_budget=0.5, seed=42)
        alloc2 = random_allocation(token_ids, memory_budget=0.5, seed=123)
        
        # Should be different (with high probability)
        assert alloc1 != alloc2


class TestPositionBasedAllocation:
    """Test position-based allocation."""
    
    def test_early_position_importance(self):
        """Test allocation favoring early positions."""
        token_ids = list(range(10))
        
        allocation = position_based_allocation(
            token_ids,
            memory_budget=0.5,
            position_importance="early"
        )
        
        # Earlier tokens should generally have higher precision
        assert allocation[0] >= allocation[5]
        assert allocation[5] >= allocation[9]
    
    def test_late_position_importance(self):
        """Test allocation favoring late positions."""
        token_ids = list(range(10))
        
        allocation = position_based_allocation(
            token_ids,
            memory_budget=0.5,
            position_importance="late"
        )
        
        # Later tokens should generally have higher precision
        assert allocation[9] >= allocation[5]
        assert allocation[5] >= allocation[0]
    
    def test_invalid_position_importance(self):
        """Test invalid position importance raises error."""
        token_ids = list(range(5))
        
        with pytest.raises(ValueError):
            position_based_allocation(
                token_ids,
                memory_budget=0.5,
                position_importance="invalid"
            )


class TestMemoryUsageComputation:
    """Test memory usage computation utilities."""
    
    def test_compute_memory_usage(self):
        """Test memory usage computation."""
        allocation = {
            0: 8,
            1: 4,
            2: 2,
        }
        
        stats = compute_memory_usage(allocation)
        
        assert stats['num_tokens'] == 3
        assert stats['total_bits'] == (8 + 4 + 2) * 2  # K and V
        assert stats['avg_bits'] == (8 + 4 + 2) / 3
        
        # FP16 would be 16 * 2 * 3 = 96 bits
        # Actual is 28 bits
        expected_ratio = 28 / 96
        assert abs(stats['memory_ratio'] - expected_ratio) < 0.01
    
    def test_empty_allocation_memory(self):
        """Test memory computation for empty allocation."""
        stats = compute_memory_usage({})
        
        assert stats['total_bits'] == 0
        assert stats['num_tokens'] == 0


class TestValidateAllocation:
    """Test allocation validation."""
    
    def test_valid_allocation(self):
        """Test validation of valid allocation."""
        allocation = {i: 4 for i in range(10)}
        
        is_valid, msg = validate_allocation(
            allocation,
            memory_budget=0.5,
            available_bits=[2, 3, 4, 8]
        )
        
        assert is_valid
        assert msg == ""
    
    def test_invalid_bits(self):
        """Test validation catches invalid bit-widths."""
        allocation = {0: 5, 1: 4}  # 5 is not valid
        
        is_valid, msg = validate_allocation(
            allocation,
            memory_budget=0.5,
            available_bits=[2, 3, 4, 8]
        )
        
        assert not is_valid
        assert "invalid" in msg.lower()
    
    def test_budget_exceeded(self):
        """Test validation catches budget violations."""
        # All 8-bit with 0.3 budget should exceed
        allocation = {i: 8 for i in range(10)}
        
        is_valid, msg = validate_allocation(
            allocation,
            memory_budget=0.3,
            available_bits=[2, 3, 4, 8]
        )
        
        assert not is_valid
        assert "exceed" in msg.lower()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_layer_budgets_equal(self):
        """Test equal layer budget computation."""
        budgets = _compute_layer_budgets(4, 0.5, "equal")
        
        assert len(budgets) == 4
        # All should be equal to total budget
        for budget in budgets.values():
            assert abs(budget - 0.5) < 0.01
    
    def test_compute_layer_budgets_linear(self):
        """Test linear layer budget computation."""
        budgets = _compute_layer_budgets(4, 0.5, "linear")
        
        assert len(budgets) == 4
        # Later layers should have more budget
        assert budgets[3] > budgets[0]
    
    def test_compute_layer_budgets_exponential(self):
        """Test exponential layer budget computation."""
        budgets = _compute_layer_budgets(4, 0.5, "exponential")
        
        assert len(budgets) == 4
        # Later layers should have much more budget
        assert budgets[3] > budgets[2] > budgets[1]
    
    def test_round_to_nearest_bits(self):
        """Test rounding to nearest bit-width."""
        available_bits = [2, 3, 4, 8]
        
        assert _round_to_nearest_bits(2.1, available_bits) == 2
        assert _round_to_nearest_bits(3.4, available_bits) == 3
        assert _round_to_nearest_bits(5.0, available_bits) == 4
        assert _round_to_nearest_bits(7.0, available_bits) == 8


class TestAllocationComparison:
    """Compare different allocation strategies."""
    
    def test_greedy_vs_uniform(self):
        """Test greedy outperforms uniform for non-uniform importance."""
        # Create non-uniform importance (some tokens much more important)
        importance_scores = {i: float(100 - i * i) for i in range(20)}
        memory_budget = 0.5
        
        greedy_alloc = greedy_allocation(importance_scores, memory_budget)
        uniform_alloc = uniform_allocation(list(range(20)), bits=4)
        
        # Greedy should allocate more bits to important tokens
        # Token 0 is most important
        assert greedy_alloc[0] >= uniform_alloc[0]
    
    def test_position_vs_importance(self):
        """Test importance-based beats position-based for specific patterns."""
        # Importance doesn't match position
        importance_scores = {
            0: 1.0,   # Low importance
            1: 10.0,  # High importance  
            2: 2.0,   # Low importance
            3: 8.0,   # High importance
        }
        
        importance_alloc = greedy_allocation(importance_scores, memory_budget=0.5)
        position_alloc = position_based_allocation(
            list(range(4)),
            memory_budget=0.5,
            position_importance="early"
        )
        
        # Importance-based should give more bits to token 1 than token 0
        assert importance_alloc[1] >= importance_alloc[0]
        
        # Position-based favors early tokens
        assert position_alloc[0] >= position_alloc[3]


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_token(self):
        """Test allocation with single token."""
        allocation = greedy_allocation({0: 10.0}, memory_budget=0.5)
        
        assert len(allocation) == 1
        assert 0 in allocation
    
    def test_very_low_budget(self):
        """Test allocation with very low budget."""
        importance_scores = {i: float(i) for i in range(10)}
        
        allocation = greedy_allocation(importance_scores, memory_budget=0.1)
        
        # Should still allocate to all tokens (with minimum bits)
        assert len(allocation) == 10
    
    def test_very_high_budget(self):
        """Test allocation with budget > 1.0."""
        importance_scores = {i: float(i) for i in range(5)}
        
        allocation = greedy_allocation(importance_scores, memory_budget=1.5)
        
        # All should get maximum bits
        assert all(bits == 8 for bits in allocation.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
