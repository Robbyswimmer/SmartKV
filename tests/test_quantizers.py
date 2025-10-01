"""
Unit tests for quantizer implementations.

Tests quantization/dequantization round-trip, error bounds, edge cases,
and proper handling of different tensor shapes and dtypes.
"""

import torch
import pytest
from smartkv.core.quantizers import (
    TwobitQuantizer,
    ThreebitQuantizer,
    FourbitQuantizer,
    EightbitQuantizer,
    get_quantizer,
    QuantizerBase,
)


class TestQuantizerBase:
    """Test base quantizer functionality."""
    
    def test_all_quantizers_implement_interface(self):
        """Verify all quantizers implement the required interface."""
        quantizers = [
            TwobitQuantizer(),
            ThreebitQuantizer(),
            FourbitQuantizer(),
            EightbitQuantizer(),
        ]
        
        for q in quantizers:
            assert isinstance(q, QuantizerBase)
            assert hasattr(q, 'quantize')
            assert hasattr(q, 'dequantize')
            assert hasattr(q, 'bits')
            assert callable(q.quantize)
            assert callable(q.dequantize)


class TestEightbitQuantizer:
    """Test INT8 quantizer."""
    
    def test_quantize_dequantize_roundtrip(self):
        """Test that quantize->dequantize preserves approximate values."""
        quantizer = EightbitQuantizer()
        x = torch.randn(128, 64)
        
        qdata = quantizer.quantize(x)
        x_reconstructed = quantizer.dequantize(qdata)
        
        # Check shapes match
        assert x_reconstructed.shape == x.shape
        
        # Check reasonable reconstruction error (< 1% for 8-bit)
        rel_error = (x - x_reconstructed).abs().mean() / x.abs().mean()
        assert rel_error < 0.01
    
    def test_quantized_range(self):
        """Test that quantized values are in INT8 range."""
        quantizer = EightbitQuantizer()
        x = torch.randn(100, 50) * 10  # Large values
        
        qdata = quantizer.quantize(x)
        qx = qdata['qx']
        
        assert qx.dtype == torch.int8
        assert qx.min() >= -128
        assert qx.max() <= 127
    
    def test_bits_property(self):
        """Test bits property returns correct value."""
        quantizer = EightbitQuantizer()
        assert quantizer.bits == 8
    
    def test_zero_tensor(self):
        """Test quantization of all-zero tensor."""
        quantizer = EightbitQuantizer()
        x = torch.zeros(10, 10)
        
        qdata = quantizer.quantize(x)
        x_reconstructed = quantizer.dequantize(qdata)
        
        assert torch.allclose(x_reconstructed, x)
    
    def test_extreme_values(self):
        """Test quantization with extreme values."""
        quantizer = EightbitQuantizer()
        x = torch.tensor([-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])
        
        qdata = quantizer.quantize(x)
        x_reconstructed = quantizer.dequantize(qdata)
        
        # Should preserve relative ordering
        assert x_reconstructed[0] < x_reconstructed[1]
        assert x_reconstructed[-1] > x_reconstructed[-2]
    
    def test_different_shapes(self):
        """Test quantization works with different tensor shapes."""
        quantizer = EightbitQuantizer()
        
        shapes = [(100,), (10, 10), (4, 8, 16), (2, 3, 4, 5)]
        for shape in shapes:
            x = torch.randn(shape)
            qdata = quantizer.quantize(x)
            x_reconstructed = quantizer.dequantize(qdata)
            assert x_reconstructed.shape == x.shape


class TestFourbitQuantizer:
    """Test INT4 quantizer."""
    
    def test_quantize_dequantize_roundtrip(self):
        """Test 4-bit quantization round-trip."""
        quantizer = FourbitQuantizer()
        x = torch.randn(128, 64)
        
        qdata = quantizer.quantize(x)
        x_reconstructed = quantizer.dequantize(qdata)
        
        assert x_reconstructed.shape == x.shape
        
        # 4-bit has higher error than 8-bit (expected ~17%)
        rel_error = (x - x_reconstructed).abs().mean() / x.abs().mean()
        assert rel_error < 0.20  # Allow up to 20% error
    
    def test_quantized_range(self):
        """Test that quantized values are in INT4 range."""
        quantizer = FourbitQuantizer()
        x = torch.randn(100, 50) * 10
        
        qdata = quantizer.quantize(x)
        qx = qdata['qx']
        
        assert qx.dtype == torch.int8  # Stored as int8
        assert qx.min() >= -8
        assert qx.max() <= 7
    
    def test_bits_property(self):
        """Test bits property."""
        quantizer = FourbitQuantizer()
        assert quantizer.bits == 4


class TestThreebitQuantizer:
    """Test 3-bit quantizer."""
    
    def test_quantize_dequantize_roundtrip(self):
        """Test 3-bit quantization round-trip."""
        quantizer = ThreebitQuantizer()
        x = torch.randn(128, 64)
        
        qdata = quantizer.quantize(x)
        x_reconstructed = quantizer.dequantize(qdata)
        
        assert x_reconstructed.shape == x.shape
        
        # 3-bit has even higher error (expected ~45%)
        rel_error = (x - x_reconstructed).abs().mean() / x.abs().mean()
        assert rel_error < 0.50  # Allow up to 50% error
    
    def test_quantized_range(self):
        """Test that quantized values are in 3-bit range."""
        quantizer = ThreebitQuantizer()
        x = torch.randn(100, 50) * 10
        
        qdata = quantizer.quantize(x)
        qx = qdata['qx']
        
        assert qx.dtype == torch.int8
        assert qx.min() >= -4
        assert qx.max() <= 3
    
    def test_bits_property(self):
        """Test bits property."""
        quantizer = ThreebitQuantizer()
        assert quantizer.bits == 3


class TestTwobitQuantizer:
    """Test 2-bit quantizer."""
    
    def test_quantize_dequantize_roundtrip(self):
        """Test 2-bit quantization round-trip."""
        quantizer = TwobitQuantizer()
        x = torch.randn(128, 64)
        
        qdata = quantizer.quantize(x)
        x_reconstructed = quantizer.dequantize(qdata)
        
        assert x_reconstructed.shape == x.shape
        
        # 2-bit has highest error - very lossy (expected ~95%)
        rel_error = (x - x_reconstructed).abs().mean() / x.abs().mean()
        assert rel_error < 1.0  # Very lossy, allow up to 100% error
    
    def test_quantized_range(self):
        """Test that quantized values are in 2-bit range."""
        quantizer = TwobitQuantizer()
        x = torch.randn(100, 50) * 10
        
        qdata = quantizer.quantize(x)
        qx = qdata['qx']
        
        assert qx.dtype == torch.int8
        assert qx.min() >= -2
        assert qx.max() <= 1
    
    def test_bits_property(self):
        """Test bits property."""
        quantizer = TwobitQuantizer()
        assert quantizer.bits == 2


class TestQuantizerComparison:
    """Compare quantizers against each other."""
    
    def test_error_increases_with_lower_bits(self):
        """Verify that lower bit-widths have higher quantization error."""
        x = torch.randn(1000, 100)
        
        quantizers = [
            (8, EightbitQuantizer()),
            (4, FourbitQuantizer()),
            (3, ThreebitQuantizer()),
            (2, TwobitQuantizer()),
        ]
        
        errors = []
        for bits, quantizer in quantizers:
            qdata = quantizer.quantize(x)
            x_reconstructed = quantizer.dequantize(qdata)
            error = (x - x_reconstructed).abs().mean()
            errors.append((bits, error))
        
        # Verify error increases as bits decrease
        assert errors[0][1] < errors[1][1]  # 8-bit < 4-bit
        assert errors[1][1] < errors[2][1]  # 4-bit < 3-bit
        assert errors[2][1] < errors[3][1]  # 3-bit < 2-bit
    
    def test_memory_efficiency(self):
        """Test that all quantizers store data efficiently."""
        x = torch.randn(1000, 100)
        
        # Original size
        original_size = x.element_size() * x.numel()
        
        quantizers = [
            EightbitQuantizer(),
            FourbitQuantizer(),
            ThreebitQuantizer(),
            TwobitQuantizer(),
        ]
        
        for quantizer in quantizers:
            qdata = quantizer.quantize(x)
            # All store as int8, so storage is same (1 byte per element)
            # Real memory savings come from packing bits in production
            quantized_size = qdata['qx'].element_size() * qdata['qx'].numel()
            assert quantized_size < original_size


class TestGetQuantizer:
    """Test quantizer factory function."""
    
    def test_get_quantizer_returns_correct_types(self):
        """Test factory function returns correct quantizer types."""
        assert isinstance(get_quantizer(2), TwobitQuantizer)
        assert isinstance(get_quantizer(3), ThreebitQuantizer)
        assert isinstance(get_quantizer(4), FourbitQuantizer)
        assert isinstance(get_quantizer(8), EightbitQuantizer)
    
    def test_get_quantizer_invalid_bits(self):
        """Test factory function raises error for invalid bits."""
        with pytest.raises(ValueError):
            get_quantizer(1)
        
        with pytest.raises(ValueError):
            get_quantizer(5)
        
        with pytest.raises(ValueError):
            get_quantizer(16)
    
    def test_get_quantizer_bits_property(self):
        """Test that returned quantizers have correct bits property."""
        for bits in [2, 3, 4, 8]:
            quantizer = get_quantizer(bits)
            assert quantizer.bits == bits


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_element_tensor(self):
        """Test quantization of single element."""
        quantizers = [
            EightbitQuantizer(),
            FourbitQuantizer(),
            ThreebitQuantizer(),
            TwobitQuantizer(),
        ]
        
        x = torch.tensor([3.14159])
        
        for quantizer in quantizers:
            qdata = quantizer.quantize(x)
            x_reconstructed = quantizer.dequantize(qdata)
            assert x_reconstructed.shape == x.shape
    
    def test_negative_only_tensor(self):
        """Test quantization of all-negative tensor."""
        quantizers = [
            EightbitQuantizer(),
            FourbitQuantizer(),
            ThreebitQuantizer(),
            TwobitQuantizer(),
        ]
        
        x = -torch.abs(torch.randn(50, 50))
        
        for quantizer in quantizers:
            qdata = quantizer.quantize(x)
            x_reconstructed = quantizer.dequantize(qdata)
            assert torch.all(x_reconstructed <= 0)
    
    def test_positive_only_tensor(self):
        """Test quantization of all-positive tensor."""
        quantizers = [
            EightbitQuantizer(),
            FourbitQuantizer(),
            ThreebitQuantizer(),
            TwobitQuantizer(),
        ]
        
        x = torch.abs(torch.randn(50, 50))
        
        for quantizer in quantizers:
            qdata = quantizer.quantize(x)
            x_reconstructed = quantizer.dequantize(qdata)
            assert torch.all(x_reconstructed >= 0)
    
    def test_constant_tensor(self):
        """Test quantization of constant tensor."""
        quantizers = [
            EightbitQuantizer(),
            FourbitQuantizer(),
            ThreebitQuantizer(),
            TwobitQuantizer(),
        ]
        
        x = torch.full((100, 100), 5.0)
        
        for quantizer in quantizers:
            qdata = quantizer.quantize(x)
            x_reconstructed = quantizer.dequantize(qdata)
            # All values should be approximately equal
            assert x_reconstructed.std() < 0.01
    
    def test_very_small_values(self):
        """Test quantization of very small values."""
        quantizers = [
            EightbitQuantizer(),
            FourbitQuantizer(),
            ThreebitQuantizer(),
            TwobitQuantizer(),
        ]
        
        x = torch.randn(100, 100) * 1e-6
        
        for quantizer in quantizers:
            qdata = quantizer.quantize(x)
            x_reconstructed = quantizer.dequantize(qdata)
            assert x_reconstructed.shape == x.shape


class TestQuantizationSymmetry:
    """Test that quantization is symmetric around zero."""
    
    def test_symmetric_values(self):
        """Test that positive and negative values are treated symmetrically."""
        quantizers = [
            EightbitQuantizer(),
            FourbitQuantizer(),
            ThreebitQuantizer(),
            TwobitQuantizer(),
        ]
        
        x_pos = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        x_neg = -x_pos
        
        for quantizer in quantizers:
            qdata_pos = quantizer.quantize(x_pos)
            qdata_neg = quantizer.quantize(x_neg)
            
            x_pos_reconstructed = quantizer.dequantize(qdata_pos)
            x_neg_reconstructed = quantizer.dequantize(qdata_neg)
            
            # Reconstructed values should be approximately symmetric
            assert torch.allclose(x_pos_reconstructed, -x_neg_reconstructed, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
