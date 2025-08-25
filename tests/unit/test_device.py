"""
Unit tests for device management utilities.
"""

import pytest
import torch

from jina_embeddings.utils.device import DeviceManager


@pytest.mark.unit
class TestDeviceManager:
    """Test device management functionality."""
    
    def test_device_manager_init(self):
        """Test DeviceManager initialization."""
        manager = DeviceManager()
        
        assert hasattr(manager, 'has_cuda')
        assert hasattr(manager, 'has_mps')
        assert hasattr(manager, 'system')
        assert hasattr(manager, 'machine')
    
    def test_get_device_cpu(self):
        """Test getting CPU device."""
        manager = DeviceManager()
        device = manager.get_device("cpu")
        
        assert device == "cpu"
    
    def test_get_device_auto(self):
        """Test automatic device selection."""
        manager = DeviceManager()
        device = manager.get_device("auto")
        
        assert device in ["cpu", "cuda", "mps"]
    
    def test_get_best_device(self):
        """Test getting best available device."""
        manager = DeviceManager()
        device = manager.get_best_device()
        
        # Should return a valid device
        assert device in ["cpu", "cuda", "mps"]
        
        # Verify it matches actual availability
        if torch.cuda.is_available():
            assert device == "cuda"
        elif torch.backends.mps.is_available():
            assert device == "mps"
        else:
            assert device == "cpu"
    
    def test_get_device_properties_cpu(self):
        """Test getting CPU device properties."""
        manager = DeviceManager()
        props = manager.get_device_properties("cpu")
        
        assert props["device"] == "cpu"
        assert props["available"] is True
        assert "threads" in props
    
    def test_optimize_for_device_cpu(self):
        """Test optimization settings for CPU."""
        manager = DeviceManager()
        settings = manager.optimize_for_device("cpu")
        
        assert settings["device"] == "cpu"
        assert settings["batch_size"] == 8
        assert settings["dtype"] == torch.float32
        assert settings["pin_memory"] is False
    
    def test_optimize_for_device_cuda(self):
        """Test optimization settings for CUDA."""
        manager = DeviceManager()
        settings = manager.optimize_for_device("cuda")
        
        assert settings["device"] == "cuda"
        assert settings["batch_size"] == 32
        assert settings["dtype"] == torch.float16
        assert settings["pin_memory"] is True
        assert settings["use_amp"] is True
    
    def test_optimize_for_device_mps(self):
        """Test optimization settings for MPS."""
        manager = DeviceManager()
        settings = manager.optimize_for_device("mps")
        
        assert settings["device"] == "mps"
        assert settings["batch_size"] == 16
        assert settings["dtype"] == torch.float32
        assert settings["use_amp"] is False