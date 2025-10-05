"""
End-to-end GPU generation tests.

Tests full model inference with SmartKV on GPU:
- LLaMA model generation
- CPU vs GPU output consistency
- Fused kernel integration
- Long context handling
"""

import torch
import pytest

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestGPUGeneration:
    """Test end-to-end generation with SmartKV on GPU."""

    @pytest.fixture
    def tiny_model_config(self):
        """Minimal model config for fast testing."""
        return {
            'hidden_size': 256,
            'num_attention_heads': 4,
            'num_key_value_heads': 4,
            'num_hidden_layers': 2,
            'intermediate_size': 512,
            'max_position_embeddings': 512,
            'vocab_size': 1000,
        }

    def test_gpu_cache_device_placement(self):
        """Test that SmartKV cache keeps tensors on GPU."""
        from smartkv.core.cache import SmartKVCache

        cache = SmartKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=64,
            memory_budget=0.5,
            device="cuda"
        )

        # Store some KV on GPU
        k_batch = torch.randn(4, 4, 64).cuda()
        v_batch = torch.randn(4, 4, 64).cuda()
        token_ids = [0, 1, 2, 3]

        cache.quantize_and_store_batch(
            layer_idx=0,
            token_ids=token_ids,
            k_batch=k_batch,
            v_batch=v_batch
        )

        # Retrieve should return GPU tensors
        keys, values = cache.retrieve_all(layer_idx=0)

        assert keys.device.type == 'cuda', "Retrieved keys not on CUDA"
        assert values.device.type == 'cuda', "Retrieved values not on CUDA"

    def test_gpu_cpu_cache_consistency(self):
        """Test GPU and CPU caches produce same quantized values."""
        from smartkv.core.cache import SmartKVCache

        torch.manual_seed(42)

        k_batch = torch.randn(4, 4, 64)
        v_batch = torch.randn(4, 4, 64)
        token_ids = [0, 1, 2, 3]

        # CPU cache
        cache_cpu = SmartKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=64,
            memory_budget=0.5,
            device="cpu"
        )
        cache_cpu.quantize_and_store_batch(0, token_ids, k_batch, v_batch)
        keys_cpu, values_cpu = cache_cpu.retrieve_all(0)

        # GPU cache
        cache_gpu = SmartKVCache(
            num_layers=1,
            num_heads=4,
            head_dim=64,
            memory_budget=0.5,
            device="cuda"
        )
        cache_gpu.quantize_and_store_batch(0, token_ids, k_batch.cuda(), v_batch.cuda())
        keys_gpu, values_gpu = cache_gpu.retrieve_all(0)

        # Compare (move GPU to CPU)
        keys_gpu_cpu = keys_gpu.cpu()
        values_gpu_cpu = values_gpu.cpu()

        assert torch.allclose(keys_cpu, keys_gpu_cpu, rtol=1e-5, atol=1e-6), \
            "CPU and GPU keys don't match"
        assert torch.allclose(values_cpu, values_gpu_cpu, rtol=1e-5, atol=1e-6), \
            "CPU and GPU values don't match"

    def test_bit_packing_integration(self):
        """Test bit-packing enabled in GPU cache."""
        from smartkv.core.cache import SmartKVCache

        torch.manual_seed(42)

        k_batch = torch.randn(8, 8, 128).cuda()
        v_batch = torch.randn(8, 8, 128).cuda()
        token_ids = list(range(8))

        # Cache with bit-packing enabled
        cache_packed = SmartKVCache(
            num_layers=1,
            num_heads=8,
            head_dim=128,
            memory_budget=0.3,  # Low budget = more 2/3-bit
            device="cuda",
            use_bit_packing=True
        )

        # Cache without bit-packing
        cache_unpacked = SmartKVCache(
            num_layers=1,
            num_heads=8,
            head_dim=128,
            memory_budget=0.3,
            device="cuda",
            use_bit_packing=False
        )

        # Store and retrieve with both
        cache_packed.quantize_and_store_batch(0, token_ids, k_batch, v_batch)
        cache_unpacked.quantize_and_store_batch(0, token_ids, k_batch, v_batch)

        keys_packed, values_packed = cache_packed.retrieve_all(0)
        keys_unpacked, values_unpacked = cache_unpacked.retrieve_all(0)

        # Outputs should be similar (small quantization differences expected)
        assert torch.allclose(keys_packed, keys_unpacked, rtol=0.1, atol=0.05), \
            "Bit-packing produces very different results"

        print(f"\nPacking enabled: use_packing={cache_packed.use_packing}")
        print(f"Packing disabled: use_packing={cache_unpacked.use_packing}")

    @pytest.mark.slow
    def test_fused_gpu_attention_integration(self):
        """Test fused GPU attention in full forward pass."""
        try:
            from smartkv.kernels import quantized_attention, CUDA_AVAILABLE
            if not CUDA_AVAILABLE:
                pytest.skip("CUDA kernels not available")
        except ImportError:
            pytest.skip("SmartKV kernels not available")

        from smartkv.core.cache import SmartKVCache

        torch.manual_seed(42)

        # Simulate attention layer forward pass
        batch_size = 1
        num_heads = 8
        head_dim = 64
        seq_len = 128

        cache = SmartKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_budget=0.5,
            device="cuda"
        )

        # Store past KV
        past_k = torch.randn(seq_len, num_heads, head_dim).cuda()
        past_v = torch.randn(seq_len, num_heads, head_dim).cuda()
        token_ids = list(range(seq_len))

        cache.quantize_and_store_batch(0, token_ids, past_k, past_v)

        # New query (single token)
        query = torch.randn(batch_size, num_heads, head_dim).cuda()

        # Retrieve quantized KV
        quant_data = cache.retrieve_all_quantized(0)

        # Prepare for fused kernel
        k_qx = quant_data['k_qx']
        k_scale = quant_data['k_scale']
        v_qx = quant_data['v_qx']
        v_scale = quant_data['v_scale']

        # Move to GPU and batch
        k_qx_batched = k_qx.unsqueeze(0).cuda()
        k_scale_batched = k_scale.unsqueeze(0).cuda()
        v_qx_batched = v_qx.unsqueeze(0).cuda()
        v_scale_batched = v_scale.unsqueeze(0).cuda()

        # Fused GPU attention
        output = quantized_attention(
            query,
            k_qx_batched,
            k_scale_batched,
            v_qx_batched,
            v_scale_batched,
            use_cuda=True
        )

        # Check output
        assert output.shape == (batch_size, num_heads, head_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print(f"\nFused GPU attention output shape: {output.shape}")

    @pytest.mark.slow
    def test_long_context_gpu(self):
        """Test GPU cache with long context (4K tokens)."""
        from smartkv.core.cache import SmartKVCache

        torch.manual_seed(42)

        context_length = 4096
        num_heads = 8
        head_dim = 64

        cache = SmartKVCache(
            num_layers=2,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_budget=0.4,
            device="cuda"
        )

        # Store in chunks (simulate autoregressive generation)
        chunk_size = 128
        for i in range(0, context_length, chunk_size):
            k_chunk = torch.randn(chunk_size, num_heads, head_dim).cuda()
            v_chunk = torch.randn(chunk_size, num_heads, head_dim).cuda()
            token_ids = list(range(i, i + chunk_size))

            cache.quantize_and_store_batch(0, token_ids, k_chunk, v_chunk)

        # Retrieve all
        keys, values = cache.retrieve_all(0)

        assert keys.shape[0] == context_length
        assert values.shape[0] == context_length

        # Check memory stats
        stats = cache.get_memory_stats()
        print(f"\nLong context ({context_length} tokens):")
        print(f"  Memory budget: {stats['memory_budget']:.1%}")
        print(f"  Actual memory: {stats['memory_ratio']:.1%}")
        print(f"  Avg bits: {stats['avg_bits']:.2f}")
        print(f"  Num tokens: {stats['num_tokens']}")


class TestModelIntegration:
    """Test SmartKV integration with actual models (if transformers available)."""

    def test_llama_smartkv_model_creation(self):
        """Test creating SmartKV-enabled model."""
        try:
            from smartkv.models.llama_smartkv import SmartKVConfig
        except ImportError:
            pytest.skip("Model integration not available")

        config = SmartKVConfig(
            enabled=True,
            memory_budget=0.5,
            decay=0.9,
            available_bits=[2, 3, 4, 8],
            device="cuda"
        )

        assert config.enabled
        assert config.device == "cuda"

    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Requires HuggingFace model download - manual test only")
    def test_llama_generation_gpu(self):
        """
        Test full LLaMA generation with SmartKV on GPU.

        NOTE: This test is skipped by default as it requires:
        - HuggingFace transformers
        - Downloaded LLaMA model
        - Significant GPU memory

        Run manually with: pytest test_gpu_generation.py::TestModelIntegration::test_llama_generation_gpu -v -s
        """
        from smartkv.models.llama_smartkv import load_llama_with_smartkv, SmartKVConfig
        from transformers import AutoTokenizer

        model_name = "meta-llama/Llama-2-7b-hf"  # Adjust as needed

        # Load model with SmartKV
        config = SmartKVConfig(
            enabled=True,
            memory_budget=0.5,
            device="cuda"
        )

        model = load_llama_with_smartkv(
            model_name,
            smartkv_config=config,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Generate
        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert len(generated_text) > len(prompt)
        print(f"\nGenerated: {generated_text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
