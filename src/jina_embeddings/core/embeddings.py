"""
Main embeddings interface for Jina Embeddings v4.
"""

import torch
import numpy as np
from typing import List, Union, Optional
from PIL import Image
import logging

from jina_embeddings.core.model import EmbeddingModel
from jina_embeddings.utils.device import DeviceManager
from jina_embeddings.utils.image import ImageProcessor

logger = logging.getLogger(__name__)


class JinaEmbeddings:
    """High-level interface for Jina Embeddings v4."""
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: Optional[str] = None,
        use_mps_for_images: bool = False,
        offline_mode: bool = False,
        **kwargs
    ):
        """
        Initialize Jina Embeddings.
        
        Args:
            model_name: HuggingFace model identifier or local path
            device: Device to use (None for auto-detection)
            use_mps_for_images: Whether to use MPS for image encoding
            offline_mode: Use only cached models (no downloads)
            **kwargs: Additional arguments for model initialization
        """
        self.device_manager = DeviceManager()
        self.image_processor = ImageProcessor()
        
        # Determine devices
        self.text_device = self.device_manager.get_device(device)
        self.image_device = (
            self.text_device if use_mps_for_images or self.text_device != "mps"
            else "cpu"
        )
        
        # Initialize model
        self.model_wrapper = EmbeddingModel(
            model_name=model_name,
            device=self.text_device,
            local_files_only=offline_mode,
            **kwargs
        )
        self.model = self.model_wrapper.get_model()
        
        logger.info(f"Text device: {self.text_device}, Image device: {self.image_device}")
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        task: str = "retrieval",
        prompt_name: str = "query",
        return_numpy: bool = True,
        show_progress: bool = False,
        batch_size: int = 32,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            task: Task type ('retrieval', 'classification', 'clustering')
            prompt_name: Prompt type for retrieval ('query' or 'document')
            return_numpy: Return numpy array instead of torch tensor
            show_progress: Show progress bar for batch processing
            batch_size: Batch size for processing
            **kwargs: Additional encoding parameters
            
        Returns:
            Text embeddings as numpy array or torch tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.debug(f"Encoding {len(texts)} text(s) with task: {task}")
        
        # Ensure model is on correct device
        self.model_wrapper.to(self.text_device)
        
        embeddings = self.model.encode_text(
            texts=texts,
            task=task,
            prompt_name=prompt_name,
            return_numpy=return_numpy,
            show_progress=show_progress,
            batch_size=batch_size,
            **kwargs
        )
        
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def encode_image(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]],
        task: str = "retrieval",
        return_numpy: bool = True,
        show_progress: bool = False,
        batch_size: int = 8,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode images into embeddings.
        
        Args:
            images: Image paths, PIL Images, or lists thereof
            task: Task type for encoding
            return_numpy: Return numpy array instead of torch tensor
            show_progress: Show progress bar for batch processing
            batch_size: Batch size for processing
            **kwargs: Additional encoding parameters
            
        Returns:
            Image embeddings as numpy array or torch tensor
        """
        # Process images
        processed_images = self.image_processor.prepare_images(images)
        
        if not processed_images:
            logger.warning("No valid images to encode")
            return np.array([]) if return_numpy else torch.tensor([])
        
        logger.debug(f"Encoding {len(processed_images)} image(s) with task: {task}")
        
        # Handle device switching for MPS if needed
        original_device = self.text_device
        if self.text_device == "mps" and self.image_device == "cpu":
            logger.debug("Switching to CPU for image encoding (MPS compatibility)")
            self.model_wrapper.to("cpu")
        
        try:
            # Disable autocast for MPS compatibility
            with torch.autocast(
                device_type="cpu" if self.image_device == "cpu" else self.text_device,
                enabled=False
            ):
                embeddings = self.model.encode_image(
                    images=processed_images,
                    task=task,
                    return_numpy=return_numpy,
                    show_progress=show_progress,
                    batch_size=batch_size,
                    **kwargs
                )
                
        finally:
            # Restore original device if changed
            if original_device != self.image_device:
                self.model_wrapper.to(original_device)
        
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def encode(
        self,
        inputs: Union[str, List[str], Image.Image, List[Image.Image], List[Union[str, Image.Image]]],
        task: str = "retrieval",
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode mixed inputs (text and/or images).
        
        Args:
            inputs: Mixed list of texts and images
            task: Task type for encoding
            **kwargs: Additional encoding parameters
            
        Returns:
            Embeddings for all inputs
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        text_inputs = []
        image_inputs = []
        input_order = []
        
        # Separate text and image inputs
        for i, inp in enumerate(inputs):
            if isinstance(inp, str) and not self.image_processor.is_image_path(inp):
                text_inputs.append(inp)
                input_order.append(('text', len(text_inputs) - 1))
            else:
                image_inputs.append(inp)
                input_order.append(('image', len(image_inputs) - 1))
        
        # Encode separately
        text_embeddings = None
        image_embeddings = None
        
        if text_inputs:
            text_embeddings = self.encode_text(text_inputs, task=task, **kwargs)
        if image_inputs:
            image_embeddings = self.encode_image(image_inputs, task=task, **kwargs)
        
        # Combine in original order
        combined = []
        for input_type, idx in input_order:
            if input_type == 'text':
                combined.append(text_embeddings[idx])
            else:
                combined.append(image_embeddings[idx])
        
        return np.array(combined) if kwargs.get('return_numpy', True) else torch.stack(combined)
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            a: First embedding
            b: Second embedding
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            a: First embedding
            b: Second embedding
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(a - b))