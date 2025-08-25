# API Reference

## JinaEmbeddingsV4 Class

The main class for interacting with Jina Embeddings v4 model.

### Constructor

```python
JinaEmbeddingsV4(model_name: str = "jinaai/jina-embeddings-v4")
```

#### Parameters
- `model_name` (str): The HuggingFace model identifier. Default: "jinaai/jina-embeddings-v4"

#### Example
```python
from hello_world import JinaEmbeddingsV4

# Initialize with default model
jina = JinaEmbeddingsV4()

# Initialize with custom model path
jina = JinaEmbeddingsV4(model_name="path/to/local/model")
```

---

### Methods

## encode_text

Encode text into embeddings.

```python
encode_text(
    texts: Union[str, List[str]], 
    task: str = "retrieval"
) -> np.ndarray
```

#### Parameters
- `texts` (str | List[str]): Single text or list of texts to encode
- `task` (str): Task type. Options: "retrieval", "classification", "clustering". Default: "retrieval"

#### Returns
- `np.ndarray`: Array of embeddings with shape (n_texts, 2048)

#### Example
```python
# Single text
embedding = jina.encode_text("Hello world")

# Multiple texts
texts = ["First document", "Second document", "Third document"]
embeddings = jina.encode_text(texts, task="retrieval")

print(embeddings.shape)  # (3, 2048)
```

---

## encode_image

Encode images into embeddings.

```python
encode_image(
    image_paths: Union[str, List[str]], 
    task: str = "retrieval"
) -> np.ndarray
```

#### Parameters
- `image_paths` (str | List[str]): Single image path or list of image paths
- `task` (str): Task type. Options: "retrieval", "classification", "clustering". Default: "retrieval"

#### Returns
- `np.ndarray`: Array of embeddings with shape (n_images, 2048)

#### Example
```python
# Single image
embedding = jina.encode_image("path/to/image.jpg")

# Multiple images
images = ["img1.jpg", "img2.png", "img3.bmp"]
embeddings = jina.encode_image(images, task="retrieval")

print(embeddings.shape)  # (3, 2048)
```

---

## cosine_similarity

Calculate cosine similarity between two embeddings.

```python
@staticmethod
cosine_similarity(a: np.ndarray, b: np.ndarray) -> float
```

#### Parameters
- `a` (np.ndarray): First embedding vector
- `b` (np.ndarray): Second embedding vector

#### Returns
- `float`: Cosine similarity score between -1 and 1

#### Example
```python
text_emb = jina.encode_text("A photo of a cat")
image_emb = jina.encode_image("cat.jpg")

similarity = JinaEmbeddingsV4.cosine_similarity(
    text_emb[0], 
    image_emb[0]
)
print(f"Similarity: {similarity:.4f}")
```

---

## Configuration Module

### Config Class

Central configuration management for the project.

```python
from config import Config
```

#### Key Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `MODEL_NAME` | str | "jinaai/jina-embeddings-v4" | Model identifier |
| `TORCH_DTYPE` | str | "float16" | Precision setting |
| `DEVICE_PREFERENCE` | str | "auto" | Device selection |
| `DEFAULT_EMBEDDING_DIM` | int | 2048 | Default embedding size |
| `DEFAULT_BATCH_SIZE` | int | 8 | Default batch size |

#### Class Methods

##### get_device()
```python
@classmethod
def get_device(cls) -> str
```
Returns the appropriate device based on availability.

##### get_torch_dtype()
```python
@classmethod
def get_torch_dtype(cls)
```
Returns PyTorch dtype based on configuration.

##### create_directories()
```python
@classmethod
def create_directories(cls)
```
Creates necessary project directories.

---

## Task Types

Different task types optimize embeddings for specific use cases:

### retrieval
Optimized for search and retrieval tasks.
```python
embeddings = jina.encode_text(texts, task="retrieval")
```

### classification
Optimized for classification tasks.
```python
embeddings = jina.encode_text(texts, task="classification")
```

### clustering
Optimized for clustering and grouping tasks.
```python
embeddings = jina.encode_text(texts, task="clustering")
```

---

## Prompt Names

For text encoding, different prompt names can be used:

### query
For search queries:
```python
model.encode_text(texts, prompt_name="query")
```

### document
For document encoding:
```python
model.encode_text(texts, prompt_name="document")
```

---

## Advanced Usage

### Batch Processing

```python
# Process in batches for better performance
def batch_encode(texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = jina.encode_text(batch)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)
```

### Embedding Truncation

```python
# Truncate embeddings to smaller dimensions
def truncate_embeddings(embeddings, target_dim=512):
    return embeddings[:, :target_dim]

# Use truncated embeddings
full_emb = jina.encode_text("Sample text")
truncated_emb = truncate_embeddings(full_emb, 512)
print(f"Truncated shape: {truncated_emb.shape}")  # (1, 512)
```

### Similarity Matrix

```python
def compute_similarity_matrix(embeddings):
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = JinaEmbeddingsV4.cosine_similarity(
                embeddings[i], 
                embeddings[j]
            )
    
    return similarity_matrix
```

### Cross-Modal Search

```python
def cross_modal_search(query_text, image_paths, top_k=5):
    # Encode query
    query_emb = jina.encode_text([query_text])[0]
    
    # Encode images
    image_embs = jina.encode_image(image_paths)
    
    # Calculate similarities
    similarities = []
    for i, img_emb in enumerate(image_embs):
        sim = JinaEmbeddingsV4.cosine_similarity(query_emb, img_emb)
        similarities.append((i, sim))
    
    # Sort and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
```

---

## Error Handling

### Common Exceptions

```python
try:
    embeddings = jina.encode_text(texts)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Reduce batch size")
    else:
        raise e

try:
    embeddings = jina.encode_image(["nonexistent.jpg"])
except FileNotFoundError:
    print("Image file not found")
```

### Validation

```python
def validate_inputs(texts):
    if not texts:
        raise ValueError("Empty input")
    if isinstance(texts, str):
        texts = [texts]
    if any(not text.strip() for text in texts):
        raise ValueError("Empty text in input")
    return texts
```

---

## Performance Tips

1. **Use appropriate batch sizes**
   - MPS (Apple Silicon): 8
   - CUDA (NVIDIA): 16
   - CPU: 4

2. **Enable float16 precision**
   ```python
   Config.TORCH_DTYPE = "float16"
   ```

3. **Truncate embeddings when possible**
   ```python
   # For faster similarity calculations
   embeddings = embeddings[:, :512]
   ```

4. **Cache frequently used embeddings**
   ```python
   import pickle
   
   # Save embeddings
   with open("embeddings.pkl", "wb") as f:
       pickle.dump(embeddings, f)
   
   # Load embeddings
   with open("embeddings.pkl", "rb") as f:
       embeddings = pickle.load(f)
   ```

---

*API Version: 1.0.0 | Last Updated: 2025*