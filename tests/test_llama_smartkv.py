"""
Unit tests for Llama SmartKV integration.

Tests the LlamaSmartKV wrapper and configuration management.
Note: These tests use mock Llama models to avoid requiring HuggingFace downloads.
"""

import torch
import pytest
from unittest.mock import Mock, MagicMock, patch

# Try to import transformers-dependent modules
try:
    from smartkv.models.llama_smartkv import (
        SmartKVConfig,
        LlamaSmartKVAttention,
        LlamaSmartKV,
        TRANSFORMERS_AVAILABLE,
    )
    IMPORTS_AVAILABLE = True
except (ImportError, ValueError) as e:
    # transformers not available or has dependency issues
    IMPORTS_AVAILABLE = False
    SmartKVConfig = None
    LlamaSmartKVAttention = None
    LlamaSmartKV = None
    TRANSFORMERS_AVAILABLE = False

# Skip all tests if imports failed
pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason="transformers library not available or has dependency issues"
)


class TestSmartKVConfig:
    """Test SmartKV configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SmartKVConfig()
        
        assert config.enabled == False
        assert config.memory_budget == 0.5
        assert config.decay == 0.9
        assert config.realloc_freq == 16
        assert config.available_bits == [2, 3, 4, 8]
        assert config.device == "cpu"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SmartKVConfig(
            enabled=True,
            memory_budget=0.3,
            decay=0.95,
            realloc_freq=32,
            available_bits=[2, 4, 8],
            device="cuda"
        )
        
        assert config.enabled == True
        assert config.memory_budget == 0.3
        assert config.decay == 0.95
        assert config.realloc_freq == 32
        assert config.available_bits == [2, 4, 8]
        assert config.device == "cuda"


class TestLlamaSmartKVAttention:
    """Test LlamaSmartKVAttention wrapper."""
    
    def create_mock_llama_attention(self):
        """Create a mock Llama attention layer."""
        mock_attn = Mock()
        mock_attn.config = Mock()
        mock_attn.hidden_size = 512
        mock_attn.num_heads = 8
        mock_attn.head_dim = 64
        mock_attn.num_key_value_heads = 8
        mock_attn.max_position_embeddings = 2048
        
        # Mock projections
        mock_attn.q_proj = torch.nn.Linear(512, 512)
        mock_attn.k_proj = torch.nn.Linear(512, 512)
        mock_attn.v_proj = torch.nn.Linear(512, 512)
        mock_attn.o_proj = torch.nn.Linear(512, 512)
        
        return mock_attn
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_initialization(self):
        """Test SmartKV attention initialization."""
        mock_attn = self.create_mock_llama_attention()
        
        smartkv_attn = LlamaSmartKVAttention(
            original_attention=mock_attn,
            layer_idx=0
        )
        
        assert smartkv_attn.layer_idx == 0
        assert smartkv_attn.hidden_size == 512
        assert smartkv_attn.num_heads == 8
        assert smartkv_attn.use_smartkv == False
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_enable_disable_smartkv(self):
        """Test enabling and disabling SmartKV."""
        mock_attn = self.create_mock_llama_attention()
        smartkv_attn = LlamaSmartKVAttention(mock_attn, layer_idx=0)
        
        # Enable
        smartkv_attn.enable_smartkv()
        assert smartkv_attn.use_smartkv == True
        
        # Disable
        smartkv_attn.disable_smartkv()
        assert smartkv_attn.use_smartkv == False
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_forward_without_smartkv(self):
        """Test forward pass without SmartKV (falls back to original)."""
        mock_attn = self.create_mock_llama_attention()
        
        # Mock the forward method
        expected_output = (torch.randn(1, 10, 512),)
        mock_attn.return_value = expected_output
        
        smartkv_attn = LlamaSmartKVAttention(mock_attn, layer_idx=0)
        
        hidden_states = torch.randn(1, 10, 512)
        output = smartkv_attn(hidden_states)
        
        # Should call original attention
        mock_attn.assert_called_once()


class TestLlamaSmartKV:
    """Test LlamaSmartKV wrapper."""
    
    def create_mock_llama_model(self):
        """Create a mock Llama model structure."""
        mock_model = Mock()
        mock_model.model = Mock()
        
        # Create mock layers
        mock_layers = []
        for i in range(4):
            layer = Mock()
            layer.self_attn = Mock()
            layer.self_attn.num_heads = 8
            layer.self_attn.head_dim = 64
            layer.self_attn.hidden_size = 512
            layer.self_attn.num_key_value_heads = 8
            layer.self_attn.max_position_embeddings = 2048
            
            # Mock projections
            layer.self_attn.q_proj = torch.nn.Linear(512, 512)
            layer.self_attn.k_proj = torch.nn.Linear(512, 512)
            layer.self_attn.v_proj = torch.nn.Linear(512, 512)
            layer.self_attn.o_proj = torch.nn.Linear(512, 512)
            layer.self_attn.config = Mock()
            
            mock_layers.append(layer)
        
        mock_model.model.layers = mock_layers
        
        return mock_model
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_initialization_without_smartkv(self):
        """Test initialization without enabling SmartKV."""
        mock_model = self.create_mock_llama_model()
        
        config = SmartKVConfig(enabled=False)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        assert llama_smartkv.smartkv_config.enabled == False
        assert llama_smartkv.smartkv_cache is None
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_initialization_with_smartkv(self):
        """Test initialization with SmartKV enabled."""
        mock_model = self.create_mock_llama_model()
        
        config = SmartKVConfig(enabled=True, memory_budget=0.5)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        assert llama_smartkv.smartkv_config.enabled == True
        assert llama_smartkv.smartkv_cache is not None
        
        # Check that attention layers were replaced
        for layer in mock_model.model.layers:
            assert isinstance(layer.self_attn, LlamaSmartKVAttention)
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_enable_smartkv(self):
        """Test enabling SmartKV after initialization."""
        mock_model = self.create_mock_llama_model()
        
        config = SmartKVConfig(enabled=False)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        # Enable SmartKV
        llama_smartkv.enable_smartkv(memory_budget=0.4, decay=0.95)
        
        assert llama_smartkv.smartkv_config.enabled == True
        assert llama_smartkv.smartkv_config.memory_budget == 0.4
        assert llama_smartkv.smartkv_config.decay == 0.95
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_disable_smartkv(self):
        """Test disabling SmartKV."""
        mock_model = self.create_mock_llama_model()
        
        # Store original attention references
        original_attentions = [layer.self_attn for layer in mock_model.model.layers]
        
        config = SmartKVConfig(enabled=True)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        # Disable
        llama_smartkv.disable_smartkv()
        
        assert llama_smartkv.smartkv_config.enabled == False
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_get_smartkv_stats(self):
        """Test getting SmartKV statistics."""
        mock_model = self.create_mock_llama_model()
        
        config = SmartKVConfig(enabled=True)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        stats = llama_smartkv.get_smartkv_stats()
        
        assert isinstance(stats, dict)
        assert 'memory_budget' in stats
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_reset_cache(self):
        """Test resetting cache."""
        mock_model = self.create_mock_llama_model()
        
        config = SmartKVConfig(enabled=True)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        # Should not raise error
        llama_smartkv.reset_cache()
        
        stats = llama_smartkv.get_smartkv_stats()
        assert stats['num_tokens'] == 0
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_forward_delegation(self):
        """Test that forward method exists and is callable."""
        mock_model = self.create_mock_llama_model()
        
        config = SmartKVConfig(enabled=False)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        # Verify forward method exists and is callable
        assert hasattr(llama_smartkv, 'forward')
        assert callable(llama_smartkv.forward)
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_attribute_delegation(self):
        """Test that attribute access is delegated to wrapped model."""
        mock_model = self.create_mock_llama_model()
        mock_model.some_attribute = "test_value"
        
        config = SmartKVConfig(enabled=False)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        assert llama_smartkv.some_attribute == "test_value"


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_update(self):
        """Test updating configuration parameters."""
        config = SmartKVConfig()
        
        # Update values
        config.memory_budget = 0.3
        config.decay = 0.95
        config.realloc_freq = 32
        
        assert config.memory_budget == 0.3
        assert config.decay == 0.95
        assert config.realloc_freq == 32
    
    def test_config_validation_ranges(self):
        """Test configuration with various parameter ranges."""
        # Test various budgets
        for budget in [0.1, 0.3, 0.5, 0.7, 0.9]:
            config = SmartKVConfig(memory_budget=budget)
            assert config.memory_budget == budget
        
        # Test various decay values
        for decay in [0.8, 0.85, 0.9, 0.95, 0.99]:
            config = SmartKVConfig(decay=decay)
            assert config.decay == decay
        
        # Test various realloc frequencies
        for freq in [1, 8, 16, 32, 64]:
            config = SmartKVConfig(realloc_freq=freq)
            assert config.realloc_freq == freq


class TestImportHandling:
    """Test handling of missing transformers library."""
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', False)
    def test_import_error_on_init(self):
        """Test that ImportError is raised when transformers not available."""
        with pytest.raises(ImportError):
            LlamaSmartKVAttention(
                original_attention=Mock(),
                layer_idx=0
            )
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', False)
    def test_import_error_on_model_init(self):
        """Test ImportError for model initialization."""
        with pytest.raises(ImportError):
            LlamaSmartKV(Mock())


class TestEdgeCases:
    """Test edge cases."""
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_empty_layers(self):
        """Test handling of model without layers."""
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = []
        
        config = SmartKVConfig(enabled=True)
        
        # Should handle gracefully
        llama_smartkv = LlamaSmartKV(mock_model, config)
        assert llama_smartkv.smartkv_cache is None
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_single_layer(self):
        """Test with single layer model."""
        mock_model = Mock()
        mock_model.model = Mock()
        
        layer = Mock()
        layer.self_attn = Mock()
        layer.self_attn.num_heads = 8
        layer.self_attn.head_dim = 64
        layer.self_attn.hidden_size = 512
        layer.self_attn.num_key_value_heads = 8
        layer.self_attn.max_position_embeddings = 2048
        layer.self_attn.q_proj = torch.nn.Linear(512, 512)
        layer.self_attn.k_proj = torch.nn.Linear(512, 512)
        layer.self_attn.v_proj = torch.nn.Linear(512, 512)
        layer.self_attn.o_proj = torch.nn.Linear(512, 512)
        layer.self_attn.config = Mock()
        
        mock_model.model.layers = [layer]
        
        config = SmartKVConfig(enabled=True)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        
        assert llama_smartkv.smartkv_cache is not None
        assert llama_smartkv.smartkv_cache.num_layers == 1


class TestIntegration:
    """Integration tests."""
    
    @patch('smartkv.models.llama_smartkv.TRANSFORMERS_AVAILABLE', True)
    def test_full_workflow(self):
        """Test complete workflow: init -> enable -> use -> disable."""
        # Create mock model
        mock_model = Mock()
        mock_model.model = Mock()
        
        layers = []
        for i in range(4):
            layer = Mock()
            layer.self_attn = Mock()
            layer.self_attn.num_heads = 8
            layer.self_attn.head_dim = 64
            layer.self_attn.hidden_size = 512
            layer.self_attn.num_key_value_heads = 8
            layer.self_attn.max_position_embeddings = 2048
            layer.self_attn.q_proj = torch.nn.Linear(512, 512)
            layer.self_attn.k_proj = torch.nn.Linear(512, 512)
            layer.self_attn.v_proj = torch.nn.Linear(512, 512)
            layer.self_attn.o_proj = torch.nn.Linear(512, 512)
            layer.self_attn.config = Mock()
            layers.append(layer)
        
        mock_model.model.layers = layers
        
        # Initialize without SmartKV
        config = SmartKVConfig(enabled=False)
        llama_smartkv = LlamaSmartKV(mock_model, config)
        assert not llama_smartkv.smartkv_config.enabled
        
        # Enable SmartKV
        llama_smartkv.enable_smartkv(memory_budget=0.4)
        assert llama_smartkv.smartkv_config.enabled
        assert llama_smartkv.smartkv_cache is not None
        
        # Get stats
        stats = llama_smartkv.get_smartkv_stats()
        assert 'memory_budget' in stats
        
        # Reset cache
        llama_smartkv.reset_cache()
        
        # Disable
        llama_smartkv.disable_smartkv()
        assert not llama_smartkv.smartkv_config.enabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
