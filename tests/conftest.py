"""
Pytest configuration and shared fixtures for Jina Embeddings v4 tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Generator
from PIL import Image

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jina_embeddings import JinaEmbeddings, Config


@pytest.fixture(scope="session")
def test_config() -> Config:
    """Create a test configuration."""
    config = Config()
    config.model.cache_dir = ".test_cache"
    config.performance.batch_size = 2
    config.device.preference = "cpu"  # Use CPU for tests
    return config


@pytest.fixture(scope="session")
def model_name() -> str:
    """Return the model name for testing."""
    # Use a smaller model for testing if available
    return "jinaai/jina-embeddings-v4"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_texts() -> list:
    """Sample texts for testing."""
    return [
        "Hello World!",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
        "Artificial intelligence is the future.",
    ]


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 100), color='red')
    img_path = temp_dir / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_images(temp_dir: Path) -> list:
    """Create multiple sample images for testing."""
    colors = ['red', 'green', 'blue', 'yellow']
    images = []
    
    for i, color in enumerate(colors):
        img = Image.new('RGB', (100, 100), color=color)
        img_path = temp_dir / f"test_image_{i}.png"
        img.save(img_path)
        images.append(img_path)
    
    return images


@pytest.fixture(scope="module")
def embeddings_model(test_config: Config) -> JinaEmbeddings:
    """Create a JinaEmbeddings instance for testing."""
    # Note: This will download the model on first run
    # Consider using a mock or smaller model for CI/CD
    try:
        model = JinaEmbeddings(
            device="cpu",
            offline_mode=True  # Try to use cached model
        )
        return model
    except Exception as e:
        pytest.skip(f"Model not available for testing: {e}")


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """Create mock embeddings for testing."""
    return np.random.randn(4, 768).astype(np.float32)


@pytest.fixture
def device_cpu() -> str:
    """CPU device string."""
    return "cpu"


@pytest.fixture
def device_available() -> str:
    """Best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Markers for test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require model loading"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require model"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take significant time"
    )
    config.addinivalue_line(
        "markers", "requires_model: Tests that require the actual model"
    )