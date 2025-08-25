# Getting Started with Jina Embeddings v4

## Quick Start Guide

This guide will help you get up and running with Jina Embeddings v4 in minutes.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Examples](#examples)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- macOS 10.15+ (Catalina or later)
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd jina-understanding
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `flash-attention`, it's optional and can be skipped.

### Step 4: Verify Installation

```bash
python test_setup.py
```

You should see:
```
🎉 ALL TESTS PASSED! 🎉
✅ Your setup is ready to go!
```

---

## Basic Usage

### Your First Embedding

```python
from hello_world import JinaEmbeddingsV4

# Initialize the model
jina = JinaEmbeddingsV4()

# Generate text embedding
text = "Hello, Jina Embeddings v4!"
embedding = jina.encode_text(text)

print(f"Embedding shape: {embedding.shape}")
# Output: Embedding shape: (1, 2048)
```

### Working with Images

```python
# Generate image embedding
image_path = "assets/sample_image.png"
image_embedding = jina.encode_image(image_path)

print(f"Image embedding shape: {image_embedding.shape}")
# Output: Image embedding shape: (1, 2048)
```

### Calculating Similarity

```python
# Compare two texts
text1 = "Machine learning is fascinating"
text2 = "AI and deep learning are amazing"

emb1 = jina.encode_text(text1)
emb2 = jina.encode_text(text2)

similarity = jina.cosine_similarity(emb1[0], emb2[0])
print(f"Text similarity: {similarity:.4f}")
```

---

## Examples

### Example 1: Text Search

Find the most similar document to a query:

```python
# Documents database
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Coffee is a popular beverage",
    "Neural networks power AI"
]

# Search query
query = "artificial intelligence"

# Generate embeddings
doc_embeddings = jina.encode_text(documents)
query_embedding = jina.encode_text(query)

# Find most similar document
similarities = []
for i, doc_emb in enumerate(doc_embeddings):
    sim = jina.cosine_similarity(query_embedding[0], doc_emb)
    similarities.append((i, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Display top result
best_match_idx = similarities[0][0]
best_match_score = similarities[0][1]
print(f"Best match: '{documents[best_match_idx]}'")
print(f"Similarity: {best_match_score:.4f}")
```

### Example 2: Image-to-Text Search

Find text descriptions that match an image:

```python
# Text descriptions
descriptions = [
    "A beautiful sunset over the ocean",
    "A cute cat sleeping on a sofa",
    "Modern office with computers",
    "Delicious pizza with cheese"
]

# Image to search
image_path = "assets/nature_scene.png"

# Generate embeddings
text_embeddings = jina.encode_text(descriptions)
image_embedding = jina.encode_image(image_path)

# Find matching descriptions
matches = []
for i, text_emb in enumerate(text_embeddings):
    sim = jina.cosine_similarity(image_embedding[0], text_emb)
    matches.append((descriptions[i], sim))

# Sort and display
matches.sort(key=lambda x: x[1], reverse=True)
for desc, score in matches[:2]:
    print(f"{desc}: {score:.4f}")
```

### Example 3: Batch Processing

Process multiple items efficiently:

```python
# Large batch of texts
texts = [
    f"Document number {i}" 
    for i in range(100)
]

# Process in batches
batch_size = 10
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = jina.encode_text(batch)
    all_embeddings.append(embeddings)

# Combine results
import numpy as np
final_embeddings = np.vstack(all_embeddings)
print(f"Total embeddings: {final_embeddings.shape}")
```

---

## Common Use Cases

### 1. Semantic Search Engine

```python
class SemanticSearch:
    def __init__(self):
        self.jina = JinaEmbeddingsV4()
        self.documents = []
        self.embeddings = None
    
    def index_documents(self, documents):
        self.documents = documents
        self.embeddings = self.jina.encode_text(documents)
    
    def search(self, query, top_k=5):
        query_emb = self.jina.encode_text([query])[0]
        
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = self.jina.cosine_similarity(query_emb, doc_emb)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            results.append({
                'document': self.documents[idx],
                'score': score
            })
        
        return results

# Usage
search_engine = SemanticSearch()
search_engine.index_documents([
    "Apple announces new iPhone",
    "Google releases AI model",
    "Recipe for chocolate cake",
    "Stock market analysis"
])

results = search_engine.search("technology news")
for r in results[:2]:
    print(f"{r['score']:.3f}: {r['document']}")
```

### 2. Image Similarity Finder

```python
def find_similar_images(target_image, image_collection):
    jina = JinaEmbeddingsV4()
    
    # Encode target image
    target_emb = jina.encode_image([target_image])[0]
    
    # Encode collection
    collection_embs = jina.encode_image(image_collection)
    
    # Calculate similarities
    similarities = []
    for i, img_emb in enumerate(collection_embs):
        sim = jina.cosine_similarity(target_emb, img_emb)
        similarities.append((image_collection[i], sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities

# Example usage
similar = find_similar_images(
    "assets/nature_scene.png",
    ["assets/photo_sunset.png", "assets/photo_ocean.png", "assets/abstract_art.png"]
)

for img, score in similar:
    print(f"{img}: {score:.4f}")
```

### 3. Content Recommendation

```python
def recommend_content(user_interests, available_content):
    jina = JinaEmbeddingsV4()
    
    # Encode user interests
    interest_emb = jina.encode_text([user_interests])[0]
    
    # Encode available content
    content_embs = jina.encode_text(available_content)
    
    # Find best matches
    recommendations = []
    for i, content_emb in enumerate(content_embs):
        sim = jina.cosine_similarity(interest_emb, content_emb)
        if sim > 0.7:  # Threshold for relevance
            recommendations.append((available_content[i], sim))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# Example
interests = "machine learning and artificial intelligence"
content = [
    "Introduction to Neural Networks",
    "Cooking Italian Pasta",
    "Deep Learning Fundamentals",
    "Travel Guide to Paris",
    "Computer Vision Applications"
]

recs = recommend_content(interests, content)
print("Recommended for you:")
for content, score in recs:
    print(f"- {content} ({score:.2f})")
```

---

## Troubleshooting

### Issue: Model Download Fails

**Error**: `HTTPError: 403 Client Error`

**Solution**:
```bash
# Check internet connection
ping huggingface.co

# Try manual download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('jinaai/jina-embeddings-v4')"
```

### Issue: Out of Memory

**Error**: `RuntimeError: MPS out of memory`

**Solution**:
```python
# Reduce batch size
Config.DEFAULT_BATCH_SIZE = 4

# Use CPU instead
Config.DEVICE_PREFERENCE = "cpu"

# Clear cache
import torch
torch.mps.empty_cache()  # For MPS
```

### Issue: Slow Performance

**Solution**:
```python
# Ensure using MPS on Apple Silicon
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")

# Use float16 precision
Config.TORCH_DTYPE = "float16"

# Process in batches
batch_size = 8  # Adjust based on your hardware
```

### Issue: Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

---

## Next Steps

1. **Explore Advanced Examples**:
   ```bash
   python examples/text_similarity.py
   python examples/multimodal_search.py
   ```

2. **Read the API Documentation**:
   - [API Reference](api_reference.md)
   - [Architecture Guide](architecture.md)

3. **Build Your Application**:
   - Start with the provided examples
   - Customize for your use case
   - Optimize for performance

4. **Join the Community**:
   - Report issues on GitHub
   - Share your use cases
   - Contribute improvements

---

## Quick Reference Card

```python
# Initialize
from hello_world import JinaEmbeddingsV4
jina = JinaEmbeddingsV4()

# Text embedding
text_emb = jina.encode_text("your text")

# Image embedding  
img_emb = jina.encode_image("path/to/image.jpg")

# Batch processing
texts = ["text1", "text2", "text3"]
batch_emb = jina.encode_text(texts)

# Similarity
sim = jina.cosine_similarity(emb1[0], emb2[0])

# Task types
emb = jina.encode_text(text, task="retrieval")  # or "classification", "clustering"
```

---

*Happy Embedding! 🚀*