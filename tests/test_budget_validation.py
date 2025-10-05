"""
Tests for budget validation and minimum budget computation.
"""

import pytest
from smartkv.core.allocation import compute_minimum_budget, compute_memory_usage
from smartkv.core.cache import SmartKVCache


class TestMinimumBudget:
    """Test minimum budget computation."""

    def test_int8_fp32_minimum(self):
        """INT8 + FP32 scales minimum budget."""
        min_budget = compute_minimum_budget(
            num_heads=8,
            head_dim=128,
            scale_dtype="fp32",
            use_packing=False
        )
        # Expected: 0.5 (INT8) + 32/(128*8) = 0.5 + 0.03125 = 0.53125
        assert abs(min_budget - 0.53125) < 1e-6

    def test_int8_fp16_minimum(self):
        """INT8 + FP16 scales minimum budget."""
        min_budget = compute_minimum_budget(
            num_heads=8,
            head_dim=128,
            scale_dtype="fp16",
            use_packing=False
        )
        # Expected: 0.5 (INT8) + 16/(128*8) = 0.5 + 0.015625 = 0.515625
        assert abs(min_budget - 0.515625) < 1e-6

    def test_packed_4bit_fp32_minimum(self):
        """4-bit packed + FP32 scales minimum budget."""
        min_budget = compute_minimum_budget(
            num_heads=8,
            head_dim=128,
            scale_dtype="fp32",
            use_packing=True  # Uses 2-bit minimum
        )
        # Expected: 0.125 (2-bit packed) + 0.03125 = 0.15625
        assert abs(min_budget - 0.15625) < 1e-6

    def test_packed_4bit_fp16_minimum(self):
        """4-bit packed + FP16 scales minimum budget."""
        min_budget = compute_minimum_budget(
            num_heads=8,
            head_dim=128,
            scale_dtype="fp16",
            use_packing=True
        )
        # Expected: 0.125 (2-bit packed) + 0.015625 = 0.140625
        assert abs(min_budget - 0.140625) < 1e-6


class TestBudgetValidation:
    """Test budget validation in SmartKVCache."""

    def test_budget_below_minimum_warning(self):
        """Test warning when budget is below minimum."""
        with pytest.warns(UserWarning, match="below minimum achievable budget"):
            cache = SmartKVCache(
                num_layers=2,
                num_heads=8,
                head_dim=128,
                memory_budget=0.4,  # Below 0.53125 minimum
                decay=0.9,
                realloc_freq=16
            )
            # Should be clamped to minimum
            assert cache.memory_budget >= 0.53

    def test_budget_at_minimum_no_warning(self):
        """Test no warning when budget is at minimum."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=8,
            head_dim=128,
            memory_budget=0.54,  # Above minimum
            decay=0.9,
            realloc_freq=16
        )
        assert cache.memory_budget == 0.54

    def test_budget_clamping(self):
        """Test that budget is clamped to minimum."""
        cache = SmartKVCache(
            num_layers=2,
            num_heads=8,
            head_dim=128,
            memory_budget=0.3,  # Way below minimum
            decay=0.9,
            realloc_freq=16
        )
        # Should be clamped to minimum
        assert cache.memory_budget == cache.min_budget
        assert cache.memory_budget >= 0.53


class TestMemoryUsageComputation:
    """Test memory usage computation with different configurations."""

    def test_int8_storage_actual_memory(self):
        """Test that INT8 storage computes correct actual memory."""
        allocation = {i: 2 for i in range(100)}  # 100 tokens at 2-bit

        stats = compute_memory_usage(
            allocation,
            num_heads=8,
            head_dim=128,
            scale_dtype="fp32",
            use_packing=False  # INT8 storage
        )

        # Theoretical: 100 * 2 * 2 * 8 * 128 = 409,600 bits
        # Actual: 100 * (2 * 8 * 128 * 8 + 2 * 8 * 32) = 100 * (16384 + 512) = 1,689,600 bits
        # FP16 baseline: 100 * 2 * 8 * 128 * 16 = 3,276,800 bits

        assert stats['num_tokens'] == 100
        assert stats['avg_bits'] == 2.0  # Average assigned bits
        assert abs(stats['memory_ratio_true'] - 0.5156) < 0.01  # INT8 + FP32 scales

    def test_packed_storage_actual_memory(self):
        """Test that packed storage computes correct actual memory."""
        allocation = {i: 4 for i in range(100)}  # 100 tokens at 4-bit

        stats = compute_memory_usage(
            allocation,
            num_heads=8,
            head_dim=128,
            scale_dtype="fp16",
            use_packing=True
        )

        # With packing: actual = theoretical + scales
        # Payload: 100 * 4 * 2 * 8 * 128 = 819,200 bits
        # Scales: 100 * 2 * 8 * 16 = 25,600 bits
        # Total: 844,800 bits
        # FP16 baseline: 3,276,800 bits
        # Ratio: 844,800 / 3,276,800 ≈ 0.2578

        assert abs(stats['memory_ratio_true'] - 0.2578) < 0.01

    def test_dual_reporting(self):
        """Test that both theoretical and actual ratios are reported."""
        allocation = {i: (i % 4) + 2 for i in range(50)}  # Mixed precision

        stats = compute_memory_usage(
            allocation,
            num_heads=8,
            head_dim=128,
            scale_dtype="fp32",
            use_packing=False
        )

        assert 'memory_ratio' in stats  # Theoretical
        assert 'memory_ratio_true' in stats  # Actual
        assert 'avg_bits' in stats
        assert 'storage_mode' in stats
        assert 'scale_dtype' in stats

        # Without packing, true ratio should be much higher than theoretical
        assert stats['memory_ratio_true'] > stats['memory_ratio']

    def test_scale_overhead_percentage(self):
        """Test scale overhead percentage computation."""
        allocation = {i: 2 for i in range(100)}

        stats = compute_memory_usage(
            allocation,
            num_heads=8,
            head_dim=128,
            scale_dtype="fp32",
            use_packing=False
        )

        # Scale overhead: (100 * 2 * 8 * 32) / (100 * (2*8*128*8 + 2*8*32))
        # = 51,200 / 1,689,600 ≈ 3.03%
        assert 'scale_overhead_pct' in stats
        assert 0 < stats['scale_overhead_pct'] < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
