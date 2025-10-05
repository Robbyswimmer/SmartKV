"""
Tests for CPU quantization helper functions.
"""

import torch
import pytest
from smartkv.core._quant_cpu import quantize_per_head


class TestQuantizePerHead:
    """Test the quantize_per_head function."""

    def test_8bit_quantization_basic(self):
        """Test basic 8-bit quantization."""
        # Create simple test data: [seq_len=2, num_heads=4, head_dim=8]
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Check output shapes
        assert k_q.shape == (2, 4, 8)
        assert v_q.shape == (2, 4, 8)
        assert k_scale.shape == (2, 4)
        assert v_scale.shape == (2, 4)

        # Check dtypes
        assert k_q.dtype == torch.int8
        assert v_q.dtype == torch.int8
        assert k_scale.dtype == torch.float32
        assert v_scale.dtype == torch.float32

        # Check quantized values are in valid range for 8-bit
        assert k_q.min() >= -128
        assert k_q.max() <= 127
        assert v_q.min() >= -128
        assert v_q.max() <= 127

    def test_4bit_quantization_basic(self):
        """Test basic 4-bit quantization."""
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        bits = 4

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Check output shapes and dtypes
        assert k_q.shape == (2, 4, 8)
        assert k_q.dtype == torch.int8

        # Check quantized values are in valid range for 4-bit
        assert k_q.min() >= -8
        assert k_q.max() <= 7
        assert v_q.min() >= -8
        assert v_q.max() <= 7

    def test_2bit_quantization_basic(self):
        """Test basic 2-bit quantization."""
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        bits = 2

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Check quantized values are in valid range for 2-bit
        assert k_q.min() >= -2
        assert k_q.max() <= 1
        assert v_q.min() >= -2
        assert v_q.max() <= 1

    def test_1bit_quantization(self):
        """Test edge case: 1-bit quantization."""
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        bits = 1

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # For 1-bit, max_val and min_val are both 0
        assert k_q.min() >= 0
        assert k_q.max() <= 0
        assert v_q.min() >= 0
        assert v_q.max() <= 0

    def test_zero_input(self):
        """Test quantization of zero tensors."""
        k = torch.zeros(2, 4, 8)
        v = torch.zeros(2, 4, 8)
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Zero input should produce zero quantized values
        assert torch.all(k_q == 0)
        assert torch.all(v_q == 0)

        # Scales should be 1.0 (clamped from 0)
        assert torch.allclose(k_scale, torch.ones_like(k_scale))
        assert torch.allclose(v_scale, torch.ones_like(v_scale))

    def test_per_head_independence(self):
        """Test that each head is quantized independently."""
        # Create input where different heads have different magnitudes
        k = torch.zeros(2, 4, 8)
        k[:, 0, :] = 1.0  # Head 0: small values
        k[:, 1, :] = 10.0  # Head 1: large values
        k[:, 2, :] = 0.1  # Head 2: very small values
        k[:, 3, :] = 100.0  # Head 3: very large values

        v = k.clone()
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Each head should have different scale factors
        # Head 3 should have much larger scale than head 2
        assert k_scale[0, 3] > k_scale[0, 2] * 100
        assert k_scale[0, 1] > k_scale[0, 0] * 5

    def test_reconstruction_error(self):
        """Test that dequantization produces reasonable reconstruction."""
        k = torch.randn(2, 4, 8) * 10  # Scale up for numerical stability
        v = torch.randn(2, 4, 8) * 10
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Dequantize
        k_dequant = k_q.float() * k_scale.unsqueeze(-1)
        v_dequant = v_q.float() * v_scale.unsqueeze(-1)

        # Reconstruction error should be small for 8-bit
        k_error = torch.abs(k - k_dequant).mean()
        v_error = torch.abs(v - v_dequant).mean()

        # Error should be less than ~1% of input magnitude
        assert k_error < 0.2
        assert v_error < 0.2

    def test_lower_bits_higher_error(self):
        """Test that lower bit widths produce higher quantization error."""
        k = torch.randn(4, 8, 16) * 10
        v = torch.randn(4, 8, 16) * 10

        errors = {}
        for bits in [8, 4, 2]:
            k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)
            k_dequant = k_q.float() * k_scale.unsqueeze(-1)
            errors[bits] = torch.abs(k - k_dequant).mean().item()

        # More bits should produce less error
        assert errors[8] < errors[4]
        assert errors[4] < errors[2]

    def test_batch_consistency(self):
        """Test that quantization is consistent across batch dimension."""
        # Create identical tokens across batch
        k = torch.randn(1, 4, 8).repeat(5, 1, 1)  # 5 identical tokens
        v = torch.randn(1, 4, 8).repeat(5, 1, 1)
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # All tokens should have identical quantized values and scales
        for i in range(1, 5):
            assert torch.allclose(k_q[0], k_q[i])
            assert torch.allclose(v_q[0], v_q[i])
            assert torch.allclose(k_scale[0], k_scale[i])
            assert torch.allclose(v_scale[0], v_scale[i])

    def test_extreme_values(self):
        """Test quantization with extreme input values."""
        k = torch.tensor([[[1000.0] * 8] * 4] * 2)  # Very large values
        v = torch.tensor([[[0.001] * 8] * 4] * 2)  # Very small values
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Should not produce NaN or Inf
        assert not torch.isnan(k_q.float()).any()
        assert not torch.isnan(v_q.float()).any()
        assert not torch.isnan(k_scale).any()
        assert not torch.isnan(v_scale).any()

        assert not torch.isinf(k_scale).any()
        assert not torch.isinf(v_scale).any()

    def test_negative_values(self):
        """Test quantization with negative input values."""
        k = torch.randn(2, 4, 8)
        k[0] = -torch.abs(k[0])  # Make first token all negative
        k[1] = torch.abs(k[1])   # Make second token all positive

        v = k.clone()
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        # Should handle mixed positive/negative values
        assert k_q.min() < 0  # Should have negative quantized values
        assert k_q.max() > 0  # Should have positive quantized values

    def test_single_token(self):
        """Test quantization with single token (seq_len=1)."""
        k = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        bits = 8

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        assert k_q.shape == (1, 4, 8)
        assert k_scale.shape == (1, 4)

    def test_large_batch(self):
        """Test quantization with large batch size."""
        k = torch.randn(100, 4, 8)
        v = torch.randn(100, 4, 8)
        bits = 4

        k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

        assert k_q.shape == (100, 4, 8)
        assert k_scale.shape == (100, 4)

    def test_deterministic(self):
        """Test that quantization is deterministic."""
        torch.manual_seed(42)
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        bits = 8

        # Run twice with same input
        k_q1, v_q1, k_scale1, v_scale1 = quantize_per_head(k, v, bits)
        k_q2, v_q2, k_scale2, v_scale2 = quantize_per_head(k, v, bits)

        # Should produce identical results
        assert torch.equal(k_q1, k_q2)
        assert torch.equal(v_q1, v_q2)
        assert torch.equal(k_scale1, k_scale2)
        assert torch.equal(v_scale1, v_scale2)

    def test_all_bit_widths(self):
        """Test all common bit widths used in SmartKV."""
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)

        for bits in [2, 3, 4, 8]:
            k_q, v_q, k_scale, v_scale = quantize_per_head(k, v, bits)

            max_val = 2 ** (bits - 1) - 1
            min_val = -2 ** (bits - 1)

            # Check ranges
            assert k_q.min() >= min_val
            assert k_q.max() <= max_val
            assert v_q.min() >= min_val
            assert v_q.max() <= max_val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
