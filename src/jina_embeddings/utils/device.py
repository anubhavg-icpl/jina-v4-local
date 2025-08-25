"""
Device management utilities for Jina Embeddings v4.
"""

import torch
import platform
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and optimization for different hardware."""
    
    def __init__(self):
        """Initialize device manager."""
        self.system = platform.system()
        self.machine = platform.machine()
        self._detect_available_devices()
    
    def _detect_available_devices(self):
        """Detect available compute devices."""
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = torch.backends.mps.is_available()
        self.cuda_device_count = torch.cuda.device_count() if self.has_cuda else 0
        
        # Detect Apple Silicon
        self.is_apple_silicon = (
            self.system == "Darwin" and 
            self.machine == "arm64"
        )
        
        logger.info(f"System: {self.system} {self.machine}")
        logger.info(f"CUDA available: {self.has_cuda} (devices: {self.cuda_device_count})")
        logger.info(f"MPS available: {self.has_mps}")
        logger.info(f"Apple Silicon: {self.is_apple_silicon}")
    
    def get_device(self, preference: Optional[str] = None) -> str:
        """
        Get the optimal device based on preference and availability.
        
        Args:
            preference: Preferred device ('auto', 'cuda', 'mps', 'cpu', or None)
            
        Returns:
            Device string for PyTorch
        """
        if preference == "auto" or preference is None:
            return self.get_best_device()
        
        if preference == "cuda" and self.has_cuda:
            return "cuda"
        elif preference == "mps" and self.has_mps:
            return "mps"
        else:
            return "cpu"
    
    def get_best_device(self) -> str:
        """
        Automatically select the best available device.
        
        Returns:
            Best available device string
        """
        if self.has_cuda:
            return "cuda"
        elif self.has_mps:
            return "mps"
        else:
            return "cpu"
    
    def get_device_properties(self, device: str) -> dict:
        """
        Get properties of a specific device.
        
        Args:
            device: Device string
            
        Returns:
            Dictionary of device properties
        """
        properties = {
            "device": device,
            "available": True
        }
        
        if device == "cuda" and self.has_cuda:
            properties.update({
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(0),
                "compute_capability": torch.cuda.get_device_capability(0)
            })
        elif device == "mps" and self.has_mps:
            properties.update({
                "name": "Apple Silicon GPU",
                "apple_silicon": self.is_apple_silicon
            })
        elif device == "cpu":
            properties.update({
                "name": "CPU",
                "threads": torch.get_num_threads()
            })
        else:
            properties["available"] = False
        
        return properties
    
    def optimize_for_device(self, device: str) -> dict:
        """
        Get optimization settings for a specific device.
        
        Args:
            device: Device string
            
        Returns:
            Dictionary of optimization settings
        """
        settings = {
            "device": device,
            "batch_size": 8,
            "num_workers": 2,
            "pin_memory": False,
            "dtype": torch.float32
        }
        
        if device == "cuda":
            settings.update({
                "batch_size": 32,
                "num_workers": 4,
                "pin_memory": True,
                "dtype": torch.float16,
                "use_amp": True
            })
        elif device == "mps":
            settings.update({
                "batch_size": 16,
                "num_workers": 2,
                "dtype": torch.float32,  # MPS works better with float32
                "use_amp": False  # AMP can cause issues on MPS
            })
        
        return settings
    
    @staticmethod
    def synchronize(device: str):
        """
        Synchronize computations on the device.
        
        Args:
            device: Device string
        """
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    
    @staticmethod
    def empty_cache(device: str):
        """
        Clear device cache.
        
        Args:
            device: Device string
        """
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()