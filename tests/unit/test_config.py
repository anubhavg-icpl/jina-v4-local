"""
Unit tests for configuration module.
"""

import pytest
import json
import tempfile
from pathlib import Path

from jina_embeddings.config.settings import (
    Config, ModelConfig, DeviceConfig, 
    EmbeddingConfig, PerformanceConfig
)


@pytest.mark.unit
class TestConfig:
    """Test configuration management."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = Config()
        
        assert config.model.name == "jinaai/jina-embeddings-v4"
        assert config.device.preference == "auto"
        assert config.embedding.default_dim == 2048
        assert config.performance.batch_size == 8
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "device" in config_dict
        assert "embedding" in config_dict
        assert config_dict["model"]["name"] == "jinaai/jina-embeddings-v4"
    
    def test_config_to_json(self):
        """Test converting config to JSON."""
        config = Config()
        json_str = config.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model"]["name"] == "jinaai/jina-embeddings-v4"
    
    def test_config_save_load(self, temp_dir):
        """Test saving and loading configuration."""
        config = Config()
        config.model.name = "custom-model"
        config.performance.batch_size = 16
        
        # Save config
        config_path = temp_dir / "test_config.json"
        config.save(str(config_path))
        
        # Load config
        loaded_config = Config.load(str(config_path))
        
        assert loaded_config.model.name == "custom-model"
        assert loaded_config.performance.batch_size == 16
    
    def test_model_config(self):
        """Test ModelConfig dataclass."""
        model_config = ModelConfig(
            name="test-model",
            torch_dtype="float16",
            cache_dir="/custom/cache"
        )
        
        assert model_config.name == "test-model"
        assert model_config.torch_dtype == "float16"
        assert model_config.cache_dir == "/custom/cache"
        assert model_config.trust_remote_code is True
    
    def test_device_config(self):
        """Test DeviceConfig dataclass."""
        device_config = DeviceConfig(
            preference="cuda",
            enable_mps=False,
            use_mps_for_images=True
        )
        
        assert device_config.preference == "cuda"
        assert device_config.enable_mps is False
        assert device_config.use_mps_for_images is True
    
    def test_embedding_config(self):
        """Test EmbeddingConfig dataclass."""
        embed_config = EmbeddingConfig(
            default_dim=1024,
            min_dim=64,
            max_dim=4096,
            normalize=False
        )
        
        assert embed_config.default_dim == 1024
        assert embed_config.min_dim == 64
        assert embed_config.max_dim == 4096
        assert embed_config.normalize is False
    
    def test_performance_config(self):
        """Test PerformanceConfig dataclass."""
        perf_config = PerformanceConfig(
            batch_size=64,
            num_workers=8,
            enable_amp=True
        )
        
        assert perf_config.batch_size == 64
        assert perf_config.num_workers == 8
        assert perf_config.enable_amp is True