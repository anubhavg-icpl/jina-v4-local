# Jina Embeddings v4 - vLLM Examples

This directory contains examples for using Jina Embeddings v4 with vLLM for high-performance embedding generation.

## 🚀 Quick Start

### Installation

```bash
# Install vLLM
pip install vllm

# Optional: For GPU acceleration
pip install vllm[cuda]  # For NVIDIA GPUs
```

### Basic Usage

```python
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs.data import TextPrompt

# Initialize model
model = LLM(
    model="jinaai/jina-embeddings-v4-vllm-retrieval",
    task="embed",
    override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False),
    dtype="float16"
)

# Create prompts
prompts = [TextPrompt(prompt="Query: Hello world")]

# Generate embeddings
outputs = model.encode(prompts)
```

## 📁 Available Models & Examples

### 1. **Retrieval Adapter**
- **Model**: `jinaai/jina-embeddings-v4-vllm-retrieval`
- **File**: `retrieval_example.py`
- **Use Case**: Query-document matching, search engines, RAG systems
- **Features**: Asymmetric retrieval, document ranking

### 2. **Text Matching Adapter**  
- **Model**: `jinaai/jina-embeddings-v4-vllm-text-matching`
- **File**: `text_matching_example.py`
- **Use Case**: Text similarity, multilingual matching, deduplication
- **Features**: Symmetric similarity, cross-language support

### 3. **Code Search Adapter**
- **Model**: `jinaai/jina-embeddings-v4-vllm-code`
- **File**: `code_search_example.py`
- **Use Case**: Natural language ↔ code search, code similarity
- **Features**: Multi-language code search, documentation matching

## 🏃‍♂️ Running Examples

### Retrieval Example
```bash
cd vllm_examples
python retrieval_example.py
```

**What it demonstrates:**
- Text-to-text retrieval
- Cross-modal text-image search  
- Document ranking by relevance
- Query-passage matching

### Text Matching Example
```bash
python text_matching_example.py
```

**What it demonstrates:**
- Multilingual text similarity
- Content deduplication detection
- Paraphrase identification
- Cross-language matching

### Code Search Example  
```bash
python code_search_example.py
```

**What it demonstrates:**
- Natural language to code search
- Cross-language code similarity
- Documentation search
- Code image analysis

## ⚙️ Configuration Options

### Model Initialization
```python
# Performance optimized
model = LLM(
    model="jinaai/jina-embeddings-v4-vllm-retrieval",
    task="embed",
    override_pooler_config=PoolerConfig(
        pooling_type="ALL",
        normalize=False
    ),
    dtype="float16",          # or "float32", "bfloat16"
    max_model_len=8192,      # Context length
    gpu_memory_utilization=0.8,
    tensor_parallel_size=1,   # Multi-GPU setup
    trust_remote_code=True
)
```

### Prompt Formats

#### Text Prompts
```python
# Query prompt
query_prompt = TextPrompt(prompt="Query: your search query")

# Document prompt  
doc_prompt = TextPrompt(prompt="Passage: your document text")
```

#### Image Prompts
```python
from PIL import Image

image = Image.open("image.jpg")
image_prompt = TextPrompt(
    prompt="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n",
    multi_modal_data={"image": image}
)
```

### Embedding Extraction
```python
def get_embeddings(outputs):
    VISION_START_TOKEN_ID, VISION_END_TOKEN_ID = 151652, 151653
    embeddings = []
    
    for output in outputs:
        if VISION_START_TOKEN_ID in output.prompt_token_ids:
            # Extract vision tokens for images
            img_start = torch.where(torch.tensor(output.prompt_token_ids) == VISION_START_TOKEN_ID)[0][0]
            img_end = torch.where(torch.tensor(output.prompt_token_ids) == VISION_END_TOKEN_ID)[0][0]
            embeddings_tensor = output.outputs.data[img_start:img_end + 1]
        else:
            # Use all tokens for text
            embeddings_tensor = output.outputs.data
        
        # Mean pool and normalize
        pooled = embeddings_tensor.sum(dim=0) / embeddings_tensor.shape[0]
        normalized = torch.nn.functional.normalize(pooled, dim=-1)
        embeddings.append(normalized)
    
    return embeddings
```

## 🔧 Troubleshooting

### Common Issues

#### vLLM Installation
```bash
# If pip install fails
conda install -c conda-forge vllm

# For Apple Silicon Macs
pip install vllm --no-deps
pip install torch torchvision torchaudio
```

#### Memory Issues
```bash
# Reduce GPU memory usage
export CUDA_VISIBLE_DEVICES=0
python -c "
model = LLM(..., gpu_memory_utilization=0.6)
"

# Use CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

#### Model Loading
```bash
# Check model availability
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v4-vllm-retrieval')
print('Model accessible')
"
```

## 📊 Performance Comparison

| Method | Speed | Memory | Scalability | Best For |
|--------|-------|--------|-------------|-----------|
| **vLLM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Production, high-throughput |
| **Transformers** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Development, research |
| **Sentence-Transformers** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Simple use cases |

## 🌟 Key Advantages of vLLM

1. **High Throughput**: Optimized for batch processing
2. **Memory Efficient**: Advanced memory management
3. **Production Ready**: Built for serving at scale
4. **GPU Optimized**: Excellent CUDA performance
5. **Easy Integration**: Simple API interface

## 🎯 Use Case Matrix

| Adapter | Text Search | Image Search | Multilingual | Code Search | Real-time |
|---------|-------------|--------------|--------------|-------------|-----------|
| **Retrieval** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Text Matching** | ✅ | ✅ | ⭐ | ❌ | ✅ |
| **Code** | ⭐ | ✅ | ✅ | ⭐ | ✅ |

Legend: ✅ Excellent, ⭐ Good, ❌ Not optimized

## 💡 Tips for Best Performance

1. **Batch Processing**: Process multiple items together
2. **Appropriate Adapter**: Choose the right model for your use case
3. **Memory Management**: Monitor GPU memory usage
4. **Prompt Engineering**: Use proper prompt formats
5. **Normalization**: Normalize embeddings for similarity search

---

For more information, see the individual example files and the main project documentation.