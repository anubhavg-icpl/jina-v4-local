#!/usr/bin/env python3
"""
Configuration file for Jina Embeddings v4 Project

This file contains configurable parameters and settings for the project.

Author: Claude
Date: 2025
"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for Jina Embeddings v4"""
    
    # Model Configuration
    MODEL_NAME = "jinaai/jina-embeddings-v4"
    TORCH_DTYPE = "float16"  # Options: "float16", "float32"
    TRUST_REMOTE_CODE = True
    
    # Device Configuration
    DEVICE_PREFERENCE = "auto"  # Options: "auto", "mps", "cuda", "cpu"
    ENABLE_MPS = True  # Use Apple Silicon GPU if available
    
    # Embedding Configuration
    DEFAULT_EMBEDDING_DIM = 2048
    MIN_EMBEDDING_DIM = 128
    MAX_EMBEDDING_DIM = 2048
    
    # Task Configuration
    DEFAULT_TEXT_TASK = "retrieval"  # Options: "retrieval", "classification", "clustering"
    DEFAULT_IMAGE_TASK = "retrieval"
    
    # Performance Configuration
    DEFAULT_BATCH_SIZE = 8
    MAX_BATCH_SIZE = 32
    TIMEOUT_SECONDS = 300
    
    # File Paths
    ASSETS_DIR = "assets"
    SAMPLES_DIR = "assets/samples"
    OUTPUTS_DIR = "outputs"
    
    # Image Configuration
    SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    MAX_IMAGE_SIZE_MB = 50
    DEFAULT_IMAGE_SIZE = (300, 200)
    
    # Logging Configuration
    LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    # API Configuration (for future use)
    API_HOST = "localhost"
    API_PORT = 8000
    API_WORKERS = 1
    
    # Cache Configuration
    ENABLE_MODEL_CACHE = True
    CACHE_DIR = ".cache"
    MAX_CACHE_SIZE_GB = 10
    
    @classmethod
    def get_device(cls) -> str:
        """Get the appropriate device based on configuration and availability"""
        if cls.DEVICE_PREFERENCE == "auto":
            import torch
            if cls.ENABLE_MPS and torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            return cls.DEVICE_PREFERENCE
    
    @classmethod
    def get_torch_dtype(cls):
        """Get PyTorch dtype based on configuration"""
        import torch
        if cls.TORCH_DTYPE == "float16":
            return torch.float16
        elif cls.TORCH_DTYPE == "float32":
            return torch.float32
        else:
            return torch.float16  # Default fallback
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.ASSETS_DIR,
            cls.SAMPLES_DIR, 
            cls.OUTPUTS_DIR,
            cls.CACHE_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_image_format(cls, file_path: str) -> bool:
        """Validate if image format is supported"""
        _, ext = os.path.splitext(file_path.lower())
        return ext in cls.SUPPORTED_IMAGE_FORMATS
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "model_name": cls.MODEL_NAME,
            "trust_remote_code": cls.TRUST_REMOTE_CODE,
            "torch_dtype": cls.get_torch_dtype(),
            "device": cls.get_device(),
            "cache_dir": cls.CACHE_DIR if cls.ENABLE_MODEL_CACHE else None
        }
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding configuration dictionary"""
        return {
            "default_dim": cls.DEFAULT_EMBEDDING_DIM,
            "min_dim": cls.MIN_EMBEDDING_DIM,
            "max_dim": cls.MAX_EMBEDDING_DIM,
            "default_text_task": cls.DEFAULT_TEXT_TASK,
            "default_image_task": cls.DEFAULT_IMAGE_TASK
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("📋 Current Configuration:")
        print(f"   Model: {cls.MODEL_NAME}")
        print(f"   Device: {cls.get_device()}")
        print(f"   Precision: {cls.TORCH_DTYPE}")
        print(f"   Default Embedding Dim: {cls.DEFAULT_EMBEDDING_DIM}")
        print(f"   Batch Size: {cls.DEFAULT_BATCH_SIZE}")
        print(f"   Assets Dir: {cls.ASSETS_DIR}")


# Environment-based configuration overrides
def load_environment_config():
    """Load configuration from environment variables"""
    
    # Model configuration
    if "JINA_MODEL_NAME" in os.environ:
        Config.MODEL_NAME = os.environ["JINA_MODEL_NAME"]
    
    if "JINA_DEVICE" in os.environ:
        Config.DEVICE_PREFERENCE = os.environ["JINA_DEVICE"]
    
    if "JINA_PRECISION" in os.environ:
        Config.TORCH_DTYPE = os.environ["JINA_PRECISION"]
    
    # Performance configuration
    if "JINA_BATCH_SIZE" in os.environ:
        try:
            Config.DEFAULT_BATCH_SIZE = int(os.environ["JINA_BATCH_SIZE"])
        except ValueError:
            pass
    
    if "JINA_EMBEDDING_DIM" in os.environ:
        try:
            dim = int(os.environ["JINA_EMBEDDING_DIM"])
            if Config.MIN_EMBEDDING_DIM <= dim <= Config.MAX_EMBEDDING_DIM:
                Config.DEFAULT_EMBEDDING_DIM = dim
        except ValueError:
            pass
    
    # Directory configuration
    if "JINA_ASSETS_DIR" in os.environ:
        Config.ASSETS_DIR = os.environ["JINA_ASSETS_DIR"]
        Config.SAMPLES_DIR = os.path.join(Config.ASSETS_DIR, "samples")
    
    if "JINA_CACHE_DIR" in os.environ:
        Config.CACHE_DIR = os.environ["JINA_CACHE_DIR"]


# Load environment configuration on import
load_environment_config()


# Example usage
if __name__ == "__main__":
    Config.create_directories()
    Config.print_config()
    
    print(f"\n🔧 Model Config: {Config.get_model_config()}")
    print(f"🎯 Embedding Config: {Config.get_embedding_config()}")