"""
Unit tests for baseline implementations.

Tests uniform quantization, KIVI, and evaluation framework.
"""

import torch
import pytest
from smartkv.baselines.uniform_quant import (
    UniformQuantCache,
    UniformQuantAttention,
    UniformQuantConfig,
    create_uniform_baseline
)
from smartkv.baselines.kivi import (
    KIVICache,
    KIVIAttention,
    KIVIConfig,
    create_kivi_baseline
)
from smartkv.baselines.evaluator import (
    BaselineEvaluator,
    EvaluationResult,
    get_default_baselines
)


class TestUniformQuantConfig:
    """Test uniform quantization configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = UniformQuantConfig()
        
        assert config.bits == 8
        assert config.enabled == True
        assert config.device == "cpu"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = UniformQuantConfig(bits=4, enabled=False, device="cuda")
        
        assert config.bits == 4
        assert config.enabled == False
        assert config.device == "cuda"


class TestUniformQuantCache:
    """Test uniform quantization cache."""
    
    def test_initialization_int8(self):
        """Test INT8 cache initialization."""
        cache = UniformQuantCache(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            bits=8
        )
        
        assert cache.bits == 8
        assert cache.quantizer is not None
    
    def test_initialization_int4(self):
        """Test INT4 cache initialization."""
        cache = UniformQuantCache(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            bits=4
        )
        
        assert cache.bits == 4
        assert cache.quantizer is not None
    
    def test_initialization_fp16(self):
        """Test FP16 (no quantization) initialization."""
        cache = UniformQuantCache(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            bits=16
        )
        
        assert cache.bits == 16
        assert cache.quantizer is None  # No quantizer for FP16
    
    def test_store_and_retrieve_int8(self):
        """Test storing and retrieving with INT8."""
        cache = UniformQuantCache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            bits=8
        )
        
        k_orig = torch.randn(8, 64)
        v_orig = torch.randn(8, 64)
        
        cache.quantize_and_store(0, 0, k_orig, v_orig)
        k_recon, v_recon = cache.retrieve(0, 0)
        
        assert k_recon.shape == k_orig.shape
        assert v_recon.shape == v_orig.shape
        # Check approximate reconstruction (allow quantization error)
        assert torch.allclose(k_recon, k_orig, rtol=0.1, atol=0.1)
    
    def test_store_and_retrieve_fp16(self):
        """Test storing and retrieving with FP16."""
        cache = UniformQuantCache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            bits=16
        )
        
        k_orig = torch.randn(8, 64)
        v_orig = torch.randn(8, 64)
        
        cache.quantize_and_store(0, 0, k_orig, v_orig)
        k_recon, v_recon = cache.retrieve(0, 0)
        
        # FP16 should have very small error
        assert torch.allclose(k_recon, k_orig, rtol=1e-3, atol=1e-3)
    
    def test_retrieve_all(self):
        """Test retrieving all tokens."""
        cache = UniformQuantCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            bits=8
        )
        
        # Store 10 tokens
        for token_id in range(10):
            k = torch.randn(4, 32)
            v = torch.randn(4, 32)
            cache.quantize_and_store(0, token_id, k, v)
        
        keys, values = cache.retrieve_all(0)
        
        assert keys.shape == (10, 4, 32)
        assert values.shape == (10, 4, 32)
    
    def test_memory_stats(self):
        """Test getting memory statistics."""
        cache = UniformQuantCache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            bits=8
        )
        
        for i in range(20):
            k = torch.randn(8, 64)
            v = torch.randn(8, 64)
            cache.quantize_and_store(0, i, k, v)
        
        stats = cache.get_memory_stats()
        
        assert 'bits' in stats
        assert 'num_tokens' in stats
        assert 'memory_ratio' in stats
        assert stats['bits'] == 8
        assert stats['num_tokens'] == 20


class TestKIVIConfig:
    """Test KIVI configuration."""
    
    def test_default_config(self):
        """Test default KIVI configuration."""
        config = KIVIConfig()
        
        assert config.key_bits == 2
        assert config.value_bits == 4
        assert config.enabled == True
        assert config.per_channel_keys == True


class TestKIVICache:
    """Test KIVI cache."""
    
    def test_initialization(self):
        """Test KIVI cache initialization."""
        cache = KIVICache(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            key_bits=2,
            value_bits=4
        )
        
        assert cache.key_bits == 2
        assert cache.value_bits == 4
        assert cache.per_channel_keys == True
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving with KIVI."""
        cache = KIVICache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            key_bits=2,
            value_bits=4
        )
        
        k_orig = torch.randn(8, 64)
        v_orig = torch.randn(8, 64)
        
        cache.quantize_and_store(0, 0, k_orig, v_orig)
        k_recon, v_recon = cache.retrieve(0, 0)
        
        assert k_recon.shape == k_orig.shape
        assert v_recon.shape == v_orig.shape
    
    def test_per_channel_quantization(self):
        """Test per-channel quantization for keys."""
        cache = KIVICache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            key_bits=2,
            value_bits=4,
            per_channel_keys=True
        )
        
        k = torch.randn(8, 64)
        v = torch.randn(8, 64)
        
        cache.quantize_and_store(0, 0, k, v)
        
        # Check that key was quantized per-channel
        assert (0, 0) in cache.k_cache
        k_quant = cache.k_cache[(0, 0)]
        assert k_quant.get('per_channel', False) == True
    
    def test_memory_stats(self):
        """Test KIVI memory statistics."""
        cache = KIVICache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            key_bits=2,
            value_bits=4
        )
        
        for i in range(10):
            k = torch.randn(8, 64)
            v = torch.randn(8, 64)
            cache.quantize_and_store(0, i, k, v)
        
        stats = cache.get_memory_stats()
        
        assert 'key_bits' in stats
        assert 'value_bits' in stats
        assert 'memory_ratio' in stats
        assert stats['key_bits'] == 2
        assert stats['value_bits'] == 4
        # KIVI should use less memory than FP16
        assert stats['memory_ratio'] < 1.0


class TestUniformQuantAttention:
    """Test uniform quantization attention layer."""
    
    def test_initialization(self):
        """Test attention layer initialization."""
        attn = UniformQuantAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0,
            bits=8
        )
        
        assert attn.embed_dim == 256
        assert attn.num_heads == 4
        assert attn.bits == 8
    
    def test_forward_without_cache(self):
        """Test forward pass without cache."""
        attn = UniformQuantAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0,
            bits=8
        )
        
        hidden_states = torch.randn(2, 10, 256)
        output, _ = attn(hidden_states)
        
        assert output.shape == (2, 10, 256)
    
    def test_forward_with_cache(self):
        """Test forward pass with cache enabled."""
        attn = UniformQuantAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0,
            bits=8
        )
        
        cache = UniformQuantCache(
            num_layers=4,
            num_heads=4,
            head_dim=32,
            bits=8
        )
        
        attn.enable_cache(cache)
        
        hidden_states = torch.randn(1, 10, 128)
        token_ids = list(range(10))
        
        output, _ = attn(hidden_states, token_ids=token_ids)
        
        assert output.shape == (1, 10, 128)
        assert len(cache) > 0  # Cache should have data


class TestKIVIAttention:
    """Test KIVI attention layer."""
    
    def test_initialization(self):
        """Test KIVI attention initialization."""
        attn = KIVIAttention(
            embed_dim=256,
            num_heads=4,
            layer_idx=0,
            key_bits=2,
            value_bits=4
        )
        
        assert attn.key_bits == 2
        assert attn.value_bits == 4
    
    def test_forward(self):
        """Test forward pass."""
        attn = KIVIAttention(
            embed_dim=128,
            num_heads=4,
            layer_idx=0
        )
        
        hidden_states = torch.randn(2, 10, 128)
        output, _ = attn(hidden_states)
        
        assert output.shape == (2, 10, 128)


class TestBaselineFactories:
    """Test baseline factory functions."""
    
    def test_create_uniform_baseline_int8(self):
        """Test creating INT8 uniform baseline."""
        baseline = create_uniform_baseline(bits=8)
        
        assert baseline['name'] == 'Uniform-INT8'
        assert baseline['bits'] == 8
        assert baseline['type'] == 'uniform'
    
    def test_create_uniform_baseline_fp16(self):
        """Test creating FP16 baseline."""
        baseline = create_uniform_baseline(bits=16)
        
        assert baseline['name'] == 'FP16-Baseline'
        assert baseline['bits'] == 16
    
    def test_create_kivi_baseline(self):
        """Test creating KIVI baseline."""
        baseline = create_kivi_baseline(key_bits=2, value_bits=4)
        
        assert baseline['name'] == 'KIVI-K2V4'
        assert baseline['key_bits'] == 2
        assert baseline['value_bits'] == 4
        assert baseline['type'] == 'kivi'


class TestBaselineEvaluator:
    """Test baseline evaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = BaselineEvaluator(
            num_layers=12,
            num_heads=8,
            head_dim=64
        )
        
        assert evaluator.num_layers == 12
        assert evaluator.num_heads == 8
        assert len(evaluator.baselines) > 0
    
    def test_default_baselines_registered(self):
        """Test that default baselines are registered."""
        evaluator = BaselineEvaluator(
            num_layers=4,
            num_heads=4,
            head_dim=32
        )
        
        baseline_names = evaluator.get_baseline_names()
        
        assert 'FP16' in baseline_names
        assert 'Uniform-INT8' in baseline_names
        assert 'Uniform-INT4' in baseline_names
        assert 'KIVI' in baseline_names
    
    def test_create_cache(self):
        """Test creating cache for baselines."""
        evaluator = BaselineEvaluator(
            num_layers=4,
            num_heads=8,
            head_dim=64
        )
        
        # Test creating different cache types
        fp16_cache = evaluator.create_cache('FP16')
        assert isinstance(fp16_cache, UniformQuantCache)
        assert fp16_cache.bits == 16
        
        int8_cache = evaluator.create_cache('Uniform-INT8')
        assert isinstance(int8_cache, UniformQuantCache)
        assert int8_cache.bits == 8
        
        kivi_cache = evaluator.create_cache('KIVI')
        assert isinstance(kivi_cache, KIVICache)
    
    def test_evaluate_memory(self):
        """Test memory evaluation."""
        evaluator = BaselineEvaluator(
            num_layers=4,
            num_heads=4,
            head_dim=32
        )
        
        result = evaluator.evaluate_memory('Uniform-INT8', num_tokens=100)
        
        assert isinstance(result, EvaluationResult)
        assert result.baseline_name == 'Uniform-INT8'
        assert result.memory_ratio > 0
        assert result.memory_ratio < 1.0  # INT8 uses less than FP16
    
    def test_compare_all_baselines(self):
        """Test comparing all baselines."""
        evaluator = BaselineEvaluator(
            num_layers=4,
            num_heads=4,
            head_dim=32
        )
        
        results = evaluator.compare_all_baselines(num_tokens=50)
        
        assert len(results) >= 4  # At least 4 default baselines
        
        # FP16 should have memory_ratio = 1.0
        fp16_result = [r for r in results if r.baseline_name == 'FP16'][0]
        assert fp16_result.memory_ratio == 1.0
        
        # INT8 should use less memory
        int8_result = [r for r in results if r.baseline_name == 'Uniform-INT8'][0]
        assert int8_result.memory_ratio < 1.0
    
    def test_register_custom_baseline(self):
        """Test registering custom baseline."""
        evaluator = BaselineEvaluator(
            num_layers=4,
            num_heads=4,
            head_dim=32
        )
        
        custom_baseline = create_uniform_baseline(bits=4, name="Custom-INT4")
        evaluator.register_baseline(custom_baseline)
        
        assert 'Custom-INT4' in evaluator.get_baseline_names()


class TestEvaluationResult:
    """Test evaluation result dataclass."""
    
    def test_creation(self):
        """Test creating evaluation result."""
        result = EvaluationResult(
            baseline_name="Test",
            memory_ratio=0.5,
            accuracy=0.95
        )
        
        assert result.baseline_name == "Test"
        assert result.memory_ratio == 0.5
        assert result.accuracy == 0.95
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result = EvaluationResult(
            baseline_name="Test",
            memory_ratio=0.5,
            avg_bits=4.5
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['baseline'] == "Test"
        assert result_dict['memory_ratio'] == 0.5
        assert result_dict['avg_bits'] == 4.5


class TestMemoryComparison:
    """Test memory usage comparisons across baselines."""
    
    def test_memory_ordering(self):
        """Test that memory usage follows expected ordering."""
        evaluator = BaselineEvaluator(
            num_layers=4,
            num_heads=8,
            head_dim=64
        )
        
        results = evaluator.compare_all_baselines(num_tokens=100)
        
        # Extract memory ratios
        memory_map = {r.baseline_name: r.memory_ratio for r in results}
        
        # FP16 should be baseline (1.0)
        assert memory_map['FP16'] == 1.0
        
        # INT8 should use ~0.5 of FP16
        assert 0.4 <= memory_map['Uniform-INT8'] <= 0.6
        
        # INT4 should use ~0.25 of FP16
        assert 0.2 <= memory_map['Uniform-INT4'] <= 0.3
        
        # KIVI (2-bit keys, 4-bit values) should be between INT4 and INT8
        assert memory_map['KIVI'] < memory_map['Uniform-INT8']


class TestReconstructionQuality:
    """Test reconstruction quality of different baselines."""
    
    def test_reconstruction_error_ordering(self):
        """Test that reconstruction error follows expected ordering."""
        evaluator = BaselineEvaluator(
            num_layers=2,
            num_heads=4,
            head_dim=32
        )
        
        # Compute reconstruction errors
        fp16_error = evaluator.evaluate_reconstruction_error('FP16', num_samples=50)
        int8_error = evaluator.evaluate_reconstruction_error('Uniform-INT8', num_samples=50)
        int4_error = evaluator.evaluate_reconstruction_error('Uniform-INT4', num_samples=50)
        
        # FP16 should have lowest error
        assert fp16_error < int8_error
        
        # INT8 should have lower error than INT4
        assert int8_error < int4_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
