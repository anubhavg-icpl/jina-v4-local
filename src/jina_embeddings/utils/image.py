"""
Image processing utilities for Jina Embeddings v4.
"""

import os
from typing import List, Union, Optional
from PIL import Image
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image loading, validation, and preprocessing."""
    
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', 
        '.tiff', '.tif', '.webp', '.gif'
    }
    
    MAX_IMAGE_SIZE_MB = 50
    DEFAULT_IMAGE_SIZE = (224, 224)
    
    def __init__(
        self,
        max_size_mb: float = MAX_IMAGE_SIZE_MB,
        supported_formats: Optional[set] = None
    ):
        """
        Initialize image processor.
        
        Args:
            max_size_mb: Maximum image file size in MB
            supported_formats: Set of supported image extensions
        """
        self.max_size_mb = max_size_mb
        self.supported_formats = supported_formats or self.SUPPORTED_FORMATS
    
    def is_image_path(self, path: str) -> bool:
        """
        Check if a string is likely an image file path.
        
        Args:
            path: Path string to check
            
        Returns:
            True if path has an image extension
        """
        if not isinstance(path, str):
            return False
        
        ext = Path(path).suffix.lower()
        return ext in self.supported_formats
    
    def validate_image_path(self, path: str) -> bool:
        """
        Validate that an image path exists and is valid.
        
        Args:
            path: Image file path
            
        Returns:
            True if image is valid
        """
        if not os.path.exists(path):
            logger.warning(f"Image not found: {path}")
            return False
        
        if not self.is_image_path(path):
            logger.warning(f"Unsupported image format: {path}")
            return False
        
        # Check file size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > self.max_size_mb:
            logger.warning(f"Image too large ({size_mb:.1f}MB): {path}")
            return False
        
        return True
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object or None if loading fails
        """
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def prepare_images(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]]
    ) -> List[Union[str, Image.Image]]:
        """
        Prepare images for encoding.
        
        Args:
            images: Image paths, PIL Images, or lists thereof
            
        Returns:
            List of validated image paths or PIL Images
        """
        if not isinstance(images, list):
            images = [images]
        
        prepared = []
        for img in images:
            if isinstance(img, str):
                if self.validate_image_path(img):
                    prepared.append(img)
            elif isinstance(img, Image.Image):
                prepared.append(img)
            else:
                logger.warning(f"Invalid image type: {type(img)}")
        
        return prepared
    
    def resize_image(
        self,
        image: Image.Image,
        size: tuple = DEFAULT_IMAGE_SIZE,
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Resize an image.
        
        Args:
            image: PIL Image to resize
            size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            image.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(size, Image.Resampling.LANCZOS)
        
        return image
    
    def create_sample_image(
        self,
        text: str = "Sample Image",
        size: tuple = (400, 200),
        bg_color: str = "lightblue",
        text_color: str = "darkblue",
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a simple sample image for testing.
        
        Args:
            text: Text to display on image
            size: Image size (width, height)
            bg_color: Background color
            text_color: Text color
            output_path: Where to save the image
            
        Returns:
            Path to saved image
        """
        from PIL import ImageDraw
        
        # Create image
        img = Image.new('RGB', size, color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add text (centered)
        text_bbox = draw.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        position = (
            (size[0] - text_width) // 2,
            (size[1] - text_height) // 2
        )
        
        draw.text(position, text, fill=text_color)
        
        # Save image
        if output_path is None:
            output_path = "sample_image.png"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        img.save(output_path)
        logger.info(f"Created sample image: {output_path}")
        
        return output_path
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """
        Get information about an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        info = {
            "path": image_path,
            "exists": os.path.exists(image_path)
        }
        
        if info["exists"]:
            try:
                with Image.open(image_path) as img:
                    info.update({
                        "format": img.format,
                        "mode": img.mode,
                        "size": img.size,
                        "width": img.width,
                        "height": img.height
                    })
                
                info["file_size_mb"] = os.path.getsize(image_path) / (1024 * 1024)
                
            except Exception as e:
                info["error"] = str(e)
        
        return info