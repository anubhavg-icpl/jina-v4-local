"""
Unit tests for image processing utilities.
"""

import pytest
from pathlib import Path
from PIL import Image

from jina_embeddings.utils.image import ImageProcessor


@pytest.mark.unit
class TestImageProcessor:
    """Test image processing functionality."""
    
    def test_image_processor_init(self):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor()
        
        assert processor.max_size_mb == 50
        assert '.jpg' in processor.supported_formats
        assert '.png' in processor.supported_formats
    
    def test_is_image_path(self):
        """Test image path detection."""
        processor = ImageProcessor()
        
        assert processor.is_image_path("image.jpg") is True
        assert processor.is_image_path("photo.PNG") is True
        assert processor.is_image_path("/path/to/image.jpeg") is True
        assert processor.is_image_path("document.pdf") is False
        assert processor.is_image_path("text.txt") is False
        assert processor.is_image_path(123) is False
    
    def test_validate_image_path_not_exists(self):
        """Test validation of non-existent image."""
        processor = ImageProcessor()
        
        result = processor.validate_image_path("/fake/path/image.jpg")
        assert result is False
    
    def test_validate_image_path_valid(self, sample_image):
        """Test validation of valid image."""
        processor = ImageProcessor()
        
        result = processor.validate_image_path(str(sample_image))
        assert result is True
    
    def test_load_image(self, sample_image):
        """Test loading an image."""
        processor = ImageProcessor()
        
        img = processor.load_image(str(sample_image))
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
    
    def test_load_image_invalid_path(self):
        """Test loading from invalid path."""
        processor = ImageProcessor()
        
        img = processor.load_image("/fake/path.jpg")
        assert img is None
    
    def test_prepare_images_single_path(self, sample_image):
        """Test preparing a single image path."""
        processor = ImageProcessor()
        
        prepared = processor.prepare_images(str(sample_image))
        assert len(prepared) == 1
        assert prepared[0] == str(sample_image)
    
    def test_prepare_images_multiple(self, sample_images):
        """Test preparing multiple images."""
        processor = ImageProcessor()
        
        paths = [str(p) for p in sample_images]
        prepared = processor.prepare_images(paths)
        assert len(prepared) == len(sample_images)
    
    def test_prepare_images_mixed(self, sample_image):
        """Test preparing mixed valid and invalid images."""
        processor = ImageProcessor()
        
        images = [
            str(sample_image),
            "/fake/image.jpg",
            "not_an_image.txt"
        ]
        
        prepared = processor.prepare_images(images)
        assert len(prepared) == 1
        assert prepared[0] == str(sample_image)
    
    def test_resize_image(self):
        """Test image resizing."""
        processor = ImageProcessor()
        
        # Create a test image
        img = Image.new('RGB', (800, 600), color='blue')
        
        # Resize with aspect ratio
        resized = processor.resize_image(img, (400, 400), maintain_aspect=True)
        assert resized.width <= 400
        assert resized.height <= 400
        
        # Resize without aspect ratio
        resized = processor.resize_image(img, (200, 200), maintain_aspect=False)
        assert resized.width == 200
        assert resized.height == 200
    
    def test_create_sample_image(self, temp_dir):
        """Test creating a sample image."""
        processor = ImageProcessor()
        
        output_path = temp_dir / "test_sample.png"
        created_path = processor.create_sample_image(
            text="Test Image",
            size=(200, 100),
            output_path=str(output_path)
        )
        
        assert Path(created_path).exists()
        assert created_path == str(output_path)
        
        # Verify the image
        img = Image.open(created_path)
        assert img.size == (200, 100)
    
    def test_get_image_info(self, sample_image):
        """Test getting image information."""
        processor = ImageProcessor()
        
        info = processor.get_image_info(str(sample_image))
        
        assert info["exists"] is True
        assert info["format"] == "PNG"
        assert info["mode"] == "RGB"
        assert info["width"] == 100
        assert info["height"] == 100
        assert "file_size_mb" in info