"""
Core model class for Jina Embeddings v4.
"""

import torch
from transformers import AutoModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Base model wrapper for Jina Embeddings v4."""
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            torch_dtype: PyTorch data type for model weights
            trust_remote_code: Whether to trust remote code for custom models
            cache_dir: Directory to cache downloaded models
            local_files_only: Use only locally cached models (offline mode)
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        
        # Set device
        self.device = self._determine_device(device)
        
        # Set dtype
        self.torch_dtype = torch_dtype or torch.float32
        
        # Load model
        self._load_model()
        
    def _determine_device(self, device: Optional[str] = None) -> str:
        """Determine the best available device."""
        if device:
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the transformer model."""
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only
        )
        
        self.model.to(self.device)
        logger.info("Model loaded successfully")
    
    def get_model(self) -> AutoModel:
        """Get the underlying transformer model."""
        return self.model
    
    def to(self, device: str):
        """Move model to specified device."""
        self.model.to(device)
        self.device = device
        
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        
    def train(self):
        """Set model to training mode."""
        self.model.train()