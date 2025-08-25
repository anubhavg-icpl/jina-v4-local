# Jina Embeddings v4 - Professional Python Package

[![CI](https://github.com/jina-ai/jina-embeddings-v4/workflows/CI/badge.svg)](https://github.com/jina-ai/jina-embeddings-v4/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A professional, production-ready implementation of **Jina Embeddings v4** - state-of-the-art multimodal embeddings for text and images with task-specific adapters.

## ✨ Features

- 🚀 **Multimodal Support**: Unified embeddings for text (30+ languages) and images
- 🎯 **Task-Specific Adapters**: Optimized LoRA adapters for retrieval, classification, clustering
- 📏 **Matryoshka Learning**: Truncatable embeddings from 128 to 2048 dimensions
- 🏗️ **Production Ready**: Professional package structure with comprehensive testing
- ⚡ **Device Optimization**: Automatic optimization for CPU, CUDA, and Apple Silicon (MPS)
- 🔧 **Configuration Management**: Flexible settings with environment overrides

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/jina-ai/jina-embeddings-v4.git
cd jina-embeddings-v4

# Install package
pip install -e .

# Or install with all extras
pip install -e ".[all]"
```

### Basic Usage

```python
from jina_embeddings import JinaEmbeddings

# Initialize model
model = JinaEmbeddings()

# Encode text
texts = ["Hello World!", "Machine Learning is amazing"]
text_embeddings = model.encode_text(texts)

# Encode images
images = ["path/to/image1.jpg", "path/to/image2.png"]
image_embeddings = model.encode_image(images)

# Calculate similarity
similarity = model.cosine_similarity(text_embeddings[0], image_embeddings[0])
print(f"Cross-modal similarity: {similarity:.4f}")
```

## 🏗️ Project Structure

```
jina-embeddings-v4/
├── src/jina_embeddings/     # Source code
│   ├── core/                # Core functionality
│   │   ├── embeddings.py    # Main embedding interface
│   │   └── model.py         # Model management
│   ├── utils/               # Utilities
│   │   ├── device.py        # Device optimization
│   │   └── image.py         # Image processing
│   └── config/              # Configuration
│       └── settings.py      # Settings management
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Usage examples
├── Makefile               # Development commands
└── pyproject.toml         # Package configuration
```

## 📦 Development

### Setup Development Environment

```bash
# Quick setup with make
make setup

# Or manually
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Available Commands

```bash
make help          # Show all commands
make test          # Run tests
make lint          # Run linters
make format        # Format code
make build         # Build package
make clean         # Clean artifacts
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
make test-cov

# Run specific test
pytest tests/unit/test_config.py -v
```

## 🔧 Configuration

### Using Configuration

```python
from jina_embeddings import Config

# Create configuration
config = Config()
config.device.preference = "cuda"
config.performance.batch_size = 32

# Save configuration
config.save("my_config.json")

# Load configuration
config = Config.load("my_config.json")
```

### Environment Variables

```bash
export JINA_MODEL_NAME="jinaai/jina-embeddings-v4"
export JINA_DEVICE="cuda"
export JINA_BATCH_SIZE=32
export JINA_LOG_LEVEL="INFO"
```

## 📚 Examples

### Text Similarity Search

```python
from jina_embeddings import JinaEmbeddings

model = JinaEmbeddings()

# Documents to search
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming technology",
    "Python is a versatile programming language"
]

# Encode documents
doc_embeddings = model.encode_text(documents, task="retrieval")

# Search query
query = "programming with Python"
query_embedding = model.encode_text(query, task="retrieval")

# Find most similar document
similarities = [model.cosine_similarity(query_embedding[0], doc) 
                for doc in doc_embeddings]
best_match = documents[similarities.index(max(similarities))]
print(f"Best match: {best_match}")
```

### Cross-Modal Search

```python
# Initialize model
model = JinaEmbeddings()

# Text descriptions
descriptions = [
    "A cute cat sitting on a sofa",
    "Mountain landscape at sunset",
    "Modern city skyline"
]

# Encode text and images
text_embeddings = model.encode_text(descriptions)
image_embeddings = model.encode_image(["cat.jpg", "mountain.jpg", "city.jpg"])

# Cross-modal similarity matrix
for i, desc in enumerate(descriptions):
    for j, img in enumerate(["cat.jpg", "mountain.jpg", "city.jpg"]):
        sim = model.cosine_similarity(text_embeddings[i], image_embeddings[j])
        print(f"{desc} <-> {img}: {sim:.3f}")
```

## ⚡ Performance Optimization

### Device Optimization

The package automatically optimizes for your hardware:

| Device | Batch Size | Precision | Features |
|--------|------------|-----------|----------|
| CPU | 8 | float32 | Multi-threading |
| CUDA | 32 | float16 | AMP, Pin memory |
| MPS | 16 | float32 | Apple Silicon GPU |

### Batch Processing

```python
# Process in batches for better performance
model = JinaEmbeddings()

# Large dataset
texts = ["text"] * 1000

# Efficient batch processing
embeddings = model.encode_text(
    texts, 
    batch_size=32,
    show_progress=True
)
```

## 🔄 CI/CD

The project includes GitHub Actions workflows for:

- **Continuous Integration**: Linting, testing, building
- **Multi-platform Testing**: Ubuntu, macOS, Windows
- **Python Version Testing**: 3.8, 3.9, 3.10, 3.11
- **Automated Releases**: PyPI publishing on tags

## 📊 API Reference

### JinaEmbeddings Class

```python
class JinaEmbeddings:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: Optional[str] = None,
        offline_mode: bool = False
    )
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        task: str = "retrieval",
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray
    
    def encode_image(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]],
        task: str = "retrieval",
        batch_size: int = 8,
        show_progress: bool = False
    ) -> np.ndarray
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float
```

### Configuration Classes

```python
@dataclass
class Config:
    model: ModelConfig
    device: DeviceConfig
    embedding: EmbeddingConfig
    performance: PerformanceConfig
    paths: PathConfig
    logging: LoggingConfig
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Jina AI Homepage](https://jina.ai)
- [Model Documentation](https://jina.ai/embeddings)
- [HuggingFace Model](https://huggingface.co/jinaai/jina-embeddings-v4)
- [Issue Tracker](https://github.com/jina-ai/jina-embeddings-v4/issues)

## 🙏 Acknowledgments

- Jina AI team for the amazing embedding model
- Open source community for contributions
- HuggingFace for model hosting

---

**Built with ❤️ by Jina AI**