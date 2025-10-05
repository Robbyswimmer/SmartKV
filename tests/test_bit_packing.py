"""
Tests for bit-packing kernels.

Validates pack/unpack roundtrip and memory compression for 2/3/4-bit quantization.
"""

import torch
import pytest

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestBitPacking:
    """Test bit-packing kernels for memory compression."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_pack_unpack_roundtrip(self, bits):
        """Test that pack → unpack recovers original values."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, unpack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        torch.manual_seed(42 + bits)

        # Generate valid quantized values for this bit-width
        max_val = 2 ** (bits - 1) - 1
        min_val = -2 ** (bits - 1)

        # Create random int8 values in valid range
        num_elements = 128
        values = torch.randint(min_val, max_val + 1, (num_elements,), dtype=torch.int8).cuda()

        # Pack
        packed = pack_tensor(values, bits)

        # Check packed size is smaller
        assert packed.nbytes < values.nbytes, \
            f"{bits}-bit packing should reduce memory"

        # Unpack
        unpacked = unpack_tensor(packed, bits, values.shape)

        # Check roundtrip
        assert torch.equal(values, unpacked), \
            f"{bits}-bit pack/unpack roundtrip failed"

    def test_2bit_compression_ratio(self):
        """Test 2-bit packing achieves 4× compression."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        torch.manual_seed(42)

        # Create int8 tensor
        values = torch.randint(-2, 2, (1024,), dtype=torch.int8).cuda()
        original_bytes = values.nbytes  # 1024 bytes

        # Pack to 2-bit
        packed = pack_tensor(values, bits=2)
        packed_bytes = packed.nbytes  # Should be ~256 bytes

        compression_ratio = original_bytes / packed_bytes

        # Should be close to 4× (allowing for padding)
        assert compression_ratio >= 3.5, \
            f"2-bit compression ratio {compression_ratio:.2f}× < 3.5×"

        print(f"\n2-bit: {original_bytes} → {packed_bytes} bytes ({compression_ratio:.2f}× compression)")

    def test_3bit_compression_ratio(self):
        """Test 3-bit packing achieves ~2.67× compression."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        torch.manual_seed(42)

        values = torch.randint(-4, 4, (1024,), dtype=torch.int8).cuda()
        original_bytes = values.nbytes

        packed = pack_tensor(values, bits=3)
        packed_bytes = packed.nbytes

        compression_ratio = original_bytes / packed_bytes

        # Should be close to 8/3 ≈ 2.67× (allowing for padding)
        assert compression_ratio >= 2.3, \
            f"3-bit compression ratio {compression_ratio:.2f}× < 2.3×"

        print(f"\n3-bit: {original_bytes} → {packed_bytes} bytes ({compression_ratio:.2f}× compression)")

    def test_4bit_compression_ratio(self):
        """Test 4-bit packing achieves 2× compression."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        torch.manual_seed(42)

        values = torch.randint(-8, 8, (1024,), dtype=torch.int8).cuda()
        original_bytes = values.nbytes

        packed = pack_tensor(values, bits=4)
        packed_bytes = packed.nbytes

        compression_ratio = original_bytes / packed_bytes

        # Should be exactly 2× (no padding issues)
        assert compression_ratio >= 1.9, \
            f"4-bit compression ratio {compression_ratio:.2f}× < 1.9×"

        print(f"\n4-bit: {original_bytes} → {packed_bytes} bytes ({compression_ratio:.2f}× compression)")

    def test_8bit_no_compression(self):
        """Test 8-bit 'packing' returns original (no compression)."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, unpack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        torch.manual_seed(42)

        values = torch.randint(-128, 128, (256,), dtype=torch.int8).cuda()

        # 8-bit should be pass-through
        packed = pack_tensor(values, bits=8)

        # Size should be same
        assert packed.nbytes == values.nbytes

        # Unpack
        unpacked = unpack_tensor(packed, bits=8, shape=values.shape)
        assert torch.equal(values, unpacked)

    def test_large_tensor_packing(self):
        """Test packing works with large tensors (KV cache sized)."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, unpack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        torch.manual_seed(42)

        # Simulate KV cache for 1 token: [num_heads=32, head_dim=128]
        num_heads = 32
        head_dim = 128
        values = torch.randint(-4, 4, (num_heads, head_dim), dtype=torch.int8).cuda()

        # Flatten for packing
        values_flat = values.flatten()

        # Pack to 3-bit
        packed = pack_tensor(values_flat, bits=3)

        # Unpack
        unpacked_flat = unpack_tensor(packed, bits=3, shape=values_flat.shape)

        # Reshape back
        unpacked = unpacked_flat.view(num_heads, head_dim)

        assert torch.equal(values, unpacked)

        print(f"\nKV cache packing: {values.nbytes} → {packed.nbytes} bytes "
              f"({values.nbytes / packed.nbytes:.2f}× compression)")

    def test_edge_case_small_tensor(self):
        """Test packing with very small tensors."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, unpack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        torch.manual_seed(42)

        # Single value
        values = torch.tensor([-2], dtype=torch.int8).cuda()
        packed = pack_tensor(values, bits=2)
        unpacked = unpack_tensor(packed, bits=2, shape=values.shape)
        assert torch.equal(values, unpacked)

        # 3 values (tests padding)
        values = torch.tensor([-1, 0, 1], dtype=torch.int8).cuda()
        packed = pack_tensor(values, bits=2)
        unpacked = unpack_tensor(packed, bits=2, shape=values.shape)
        assert torch.equal(values, unpacked)

    def test_all_values_in_range(self):
        """Test packing all possible values for each bit-width."""
        try:
            from smartkv.kernels.bit_packing import pack_tensor, unpack_tensor, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        for bits in [2, 3, 4]:
            max_val = 2 ** (bits - 1) - 1
            min_val = -2 ** (bits - 1)

            # Test all possible values
            all_values = torch.arange(min_val, max_val + 1, dtype=torch.int8).cuda()

            packed = pack_tensor(all_values, bits)
            unpacked = unpack_tensor(packed, bits, all_values.shape)

            assert torch.equal(all_values, unpacked), \
                f"{bits}-bit packing failed for full range [{min_val}, {max_val}]"

    def test_memory_savings_vs_fp16(self):
        """Test actual memory savings vs FP16 baseline."""
        try:
            from smartkv.kernels.bit_packing import get_memory_savings_vs_fp16, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("Bit-packing CUDA extension not available")
        except ImportError:
            pytest.skip("Bit-packing module not available")

        # Check theoretical memory savings
        assert get_memory_savings_vs_fp16(2) == 0.125  # 2/16 = 12.5%
        assert get_memory_savings_vs_fp16(3) == 0.1875  # 3/16 = 18.75%
        assert get_memory_savings_vs_fp16(4) == 0.25   # 4/16 = 25%
        assert get_memory_savings_vs_fp16(8) == 0.5    # 8/16 = 50%

        print("\nMemory vs FP16 baseline:")
        for bits in [2, 3, 4, 8]:
            ratio = get_memory_savings_vs_fp16(bits)
            print(f"  {bits}-bit: {ratio*100:.2f}% of FP16 ({1/ratio:.1f}× compression)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
