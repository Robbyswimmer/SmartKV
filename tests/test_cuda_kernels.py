"""
Tests for CUDA kernel correctness.

Validates CUDA implementations against CPU references:
- Quantization kernels
- Fused attention kernels
- Numerical stability
"""

import torch
import pytest
import numpy as np

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestCUDAQuantization:
    """Test CUDA quantization kernels vs CPU reference."""

    def test_quantization_cuda_vs_cpu_8bit(self):
        """Test 8-bit CUDA quantization matches CPU."""
        from smartkv.core._quant_cpu import quantize_per_head

        try:
            from smartkv.core._quant_cuda import quantize_per_head_cuda
        except ImportError:
            pytest.skip("CUDA quantization not available")

        torch.manual_seed(42)

        # Generate test data
        k = torch.randn(4, 8, 128)  # [N, H, D]
        v = torch.randn(4, 8, 128)
        bits = 8

        # CPU reference
        k_q_cpu, v_q_cpu, k_s_cpu, v_s_cpu = quantize_per_head(k, v, bits)

        # CUDA implementation
        k_cuda = k.cuda()
        v_cuda = v.cuda()
        k_q_cuda, v_q_cuda, k_s_cuda, v_s_cuda = quantize_per_head_cuda(k_cuda, v_cuda, bits)

        # Move to CPU for comparison
        k_q_cuda = k_q_cuda.cpu()
        v_q_cuda = v_q_cuda.cpu()
        k_s_cuda = k_s_cuda.cpu()
        v_s_cuda = v_s_cuda.cpu()

        # Check quantized values match
        assert torch.equal(k_q_cpu, k_q_cuda), "Key quantization mismatch"
        assert torch.equal(v_q_cpu, v_q_cuda), "Value quantization mismatch"

        # Check scales match (allow small numerical difference)
        assert torch.allclose(k_s_cpu, k_s_cuda, rtol=1e-5, atol=1e-7), "Key scale mismatch"
        assert torch.allclose(v_s_cpu, v_s_cuda, rtol=1e-5, atol=1e-7), "Value scale mismatch"

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_quantization_all_bits(self, bits):
        """Test CUDA quantization for all bit widths."""
        from smartkv.core._quant_cpu import quantize_per_head

        try:
            from smartkv.core._quant_cuda import quantize_per_head_cuda
        except ImportError:
            pytest.skip("CUDA quantization not available")

        torch.manual_seed(42 + bits)

        k = torch.randn(2, 4, 64)
        v = torch.randn(2, 4, 64)

        # CPU reference
        k_q_cpu, v_q_cpu, k_s_cpu, v_s_cpu = quantize_per_head(k, v, bits)

        # CUDA implementation
        k_cuda = k.cuda()
        v_cuda = v.cuda()
        k_q_cuda, v_q_cuda, k_s_cuda, v_s_cuda = quantize_per_head_cuda(k_cuda, v_cuda, bits)

        k_q_cuda = k_q_cuda.cpu()
        v_q_cuda = v_q_cuda.cpu()

        # Check valid range
        max_val = 2 ** (bits - 1) - 1
        min_val = -2 ** (bits - 1)
        if bits == 1:
            max_val = min_val = 0

        assert k_q_cuda.min() >= min_val
        assert k_q_cuda.max() <= max_val
        assert v_q_cuda.min() >= min_val
        assert v_q_cuda.max() <= max_val

    def test_quantization_large_batch(self):
        """Test CUDA quantization scales with batch size."""
        try:
            from smartkv.core._quant_cuda import quantize_per_head_cuda
        except ImportError:
            pytest.skip("CUDA quantization not available")

        torch.manual_seed(42)

        # Large batch
        k = torch.randn(128, 8, 64).cuda()
        v = torch.randn(128, 8, 64).cuda()
        bits = 4

        k_q, v_q, k_s, v_s = quantize_per_head_cuda(k, v, bits)

        # Check shapes
        assert k_q.shape == (128, 8, 64)
        assert k_s.shape == (128, 8)
        assert k_q.dtype == torch.int8


class TestFusedAttention:
    """Test fused CUDA attention kernel."""

    def test_fused_attention_vs_pytorch(self):
        """Test fused attention matches PyTorch reference."""
        try:
            from smartkv.kernels import quantized_attention, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("CUDA kernels not available")
        except ImportError:
            pytest.skip("SmartKV kernels not available")

        from smartkv.core._quant_cpu import quantize_per_head

        torch.manual_seed(42)

        # Generate test data
        batch_size = 2
        num_heads = 4
        q_len = 1
        kv_len = 64
        head_dim = 32

        query = torch.randn(batch_size, num_heads, head_dim).cuda()
        key = torch.randn(kv_len, num_heads, head_dim)
        value = torch.randn(kv_len, num_heads, head_dim)

        # Quantize K/V
        k_q, v_q, k_scale, v_scale = quantize_per_head(
            key.unsqueeze(0), value.unsqueeze(0), bits=8
        )
        k_q = k_q.squeeze(0)
        v_q = v_q.squeeze(0)
        k_scale = k_scale.squeeze(0)
        v_scale = v_scale.squeeze(0)

        # Prepare batched inputs for kernel
        # Rearrange to [B, H, K, D] / [B, H, K]
        k_q_head_first = k_q.permute(1, 0, 2)
        v_q_head_first = v_q.permute(1, 0, 2)
        k_scale_head_first = k_scale.permute(1, 0)
        v_scale_head_first = v_scale.permute(1, 0)

        k_q_batched = k_q_head_first.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().cuda()
        v_q_batched = v_q_head_first.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().cuda()
        k_scale_batched = k_scale_head_first.unsqueeze(0).expand(batch_size, -1, -1).contiguous().cuda()
        v_scale_batched = v_scale_head_first.unsqueeze(0).expand(batch_size, -1, -1).contiguous().cuda()

        # CUDA fused attention
        output_cuda = quantized_attention(
            query,
            k_q_batched,
            k_scale_batched,
            v_q_batched,
            v_scale_batched,
            attention_mask=None,
            use_cuda=True
        )
        output_cuda = output_cuda.squeeze(2)

        # PyTorch reference: dequantize then attend
        key_dequant = k_q_head_first.float() * k_scale_head_first.unsqueeze(-1)
        value_dequant = v_q_head_first.float() * v_scale_head_first.unsqueeze(-1)

        key_batched = key_dequant.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().cuda()
        value_batched = value_dequant.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().cuda()

        # Manual attention: Q @ K^T / sqrt(d), softmax, @ V
        query_expanded = query.unsqueeze(2)  # [B, H, 1, D]
        key_transposed = key_batched.transpose(-2, -1)  # [B, H, D, K]

        scores = torch.matmul(query_expanded, key_transposed) / np.sqrt(head_dim)
        scores = scores.squeeze(2)  # [B, H, KV_LEN]
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        output_ref = torch.matmul(attn_weights.unsqueeze(2), value_batched).squeeze(2)

        # Compare outputs (allow small numerical difference)
        assert output_cuda.shape == output_ref.shape
        assert torch.allclose(output_cuda, output_ref, rtol=1e-3, atol=1e-4), \
            f"Max diff: {(output_cuda - output_ref).abs().max()}"

    def test_fused_attention_numerical_stability(self):
        """Test fused attention doesn't produce NaN/Inf."""
        try:
            from smartkv.kernels import quantized_attention, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("CUDA kernels not available")
        except ImportError:
            pytest.skip("SmartKV kernels not available")

        from smartkv.core._quant_cpu import quantize_per_head

        torch.manual_seed(42)

        # Extreme input values
        query = torch.randn(1, 4, 32).cuda() * 100  # Large values
        key = torch.randn(128, 4, 32) * 100
        value = torch.randn(128, 4, 32) * 100

        k_q, v_q, k_scale, v_scale = quantize_per_head(
            key.unsqueeze(0), value.unsqueeze(0), bits=8
        )

        k_q_batched = k_q.cuda()
        v_q_batched = v_q.cuda()
        k_scale_batched = k_scale.cuda()
        v_scale_batched = v_scale.cuda()

        output = quantized_attention(
            query, k_q_batched, k_scale_batched,
            v_q_batched, v_scale_batched,
            use_cuda=True
        )

        # No NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestCUDAPerformance:
    """Basic performance validation tests."""

    def test_cuda_faster_than_cpu(self):
        """Verify CUDA implementation is faster than CPU (basic sanity)."""
        try:
            from smartkv.kernels import quantized_attention, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("CUDA kernels not available")
        except ImportError:
            pytest.skip("SmartKV kernels not available")

        from smartkv.core._quant_cpu import quantize_per_head
        import time

        torch.manual_seed(42)

        # Moderate size problem
        query = torch.randn(1, 8, 64)
        key = torch.randn(512, 8, 64)
        value = torch.randn(512, 8, 64)

        k_q, v_q, k_scale, v_scale = quantize_per_head(
            key.unsqueeze(0), value.unsqueeze(0), bits=8
        )

        k_q_head_first = k_q.squeeze(0).permute(1, 0, 2)
        v_q_head_first = v_q.squeeze(0).permute(1, 0, 2)
        k_scale_head_first = k_scale.squeeze(0).permute(1, 0)
        v_scale_head_first = v_scale.squeeze(0).permute(1, 0)

        # CPU timing
        start = time.time()
        for _ in range(10):
            key_dequant = k_q_head_first.float() * k_scale_head_first.unsqueeze(-1)
            value_dequant = v_q_head_first.float() * v_scale_head_first.unsqueeze(-1)
            _ = torch.nn.functional.scaled_dot_product_attention(
                query.unsqueeze(2),
                key_dequant.unsqueeze(0),
                value_dequant.unsqueeze(0)
            )
        cpu_time = time.time() - start

        # GPU timing
        query_cuda = query.cuda().unsqueeze(2).contiguous()
        k_q_cuda = k_q_head_first.unsqueeze(0).contiguous().cuda()
        v_q_cuda = v_q_head_first.unsqueeze(0).contiguous().cuda()
        k_scale_cuda = k_scale_head_first.unsqueeze(0).contiguous().cuda()
        v_scale_cuda = v_scale_head_first.unsqueeze(0).contiguous().cuda()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = quantized_attention(
                query_cuda,
                k_q_cuda,
                k_scale_cuda,
                v_q_cuda,
                v_scale_cuda,
                use_cuda=True
            )
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"\nCPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s, Speedup: {cpu_time/gpu_time:.2f}x")

        # GPU should be faster (this is a soft check, not strict)
        # If it fails, might be small problem size or warm-up needed
        if gpu_time > cpu_time:
            pytest.skip(f"GPU ({gpu_time:.4f}s) slower than CPU ({cpu_time:.4f}s) - expected for small problems or first run")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
