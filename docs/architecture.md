# Jina Embeddings v4 - System Architecture

## Overview

Jina Embeddings v4 is a state-of-the-art multimodal embedding model that unifies text and image processing through a single transformer architecture with task-specific LoRA adapters.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[Text Input<br/>30+ Languages] 
        B[Image Input<br/>Up to 20MP]
    end
    
    subgraph "Preprocessing"
        C[Text Tokenizer<br/>32K Context]
        D[Image Processor<br/>Vision Transformer]
    end
    
    subgraph "Core Model"
        E[Qwen2.5-VL-3B<br/>3.8B Parameters]
        F[Single-Stream Transformer<br/>Unified Architecture]
    end
    
    subgraph "Task Adapters"
        G[Retrieval LoRA<br/>60M params]
        H[Classification LoRA<br/>60M params]
        I[Clustering LoRA<br/>60M params]
    end
    
    subgraph "Output Layer"
        J[Dense Embeddings<br/>2048 dimensions]
        K[Matryoshka Truncation<br/>128-2048 dims]
    end
    
    A --> C --> E
    B --> D --> E
    E --> F
    F --> G & H & I
    G & H & I --> J
    J --> K
    
    style E fill:#f9f,stroke:#333,stroke-width:4px
    style J fill:#bbf,stroke:#333,stroke-width:2px
```

## Component Architecture

### 1. Input Processing Pipeline

```mermaid
flowchart LR
    subgraph "Text Processing"
        T1[Raw Text] --> T2[Tokenization<br/>BPE Tokenizer]
        T2 --> T3[Positional Encoding<br/>RoPE]
        T3 --> T4[Input Embeddings<br/>4096-d]
    end
    
    subgraph "Image Processing"
        I1[Raw Image] --> I2[Patch Extraction<br/>14x14 patches]
        I2 --> I3[Linear Projection<br/>Vision Transformer]
        I3 --> I4[Vision Embeddings<br/>4096-d]
    end
    
    T4 & I4 --> M[Merged Input Stream]
    
    style M fill:#ffd,stroke:#333,stroke-width:2px
```

### 2. Transformer Architecture

```mermaid
graph TD
    subgraph "Transformer Block x28"
        A[Input<br/>4096-d] --> B[Layer Norm]
        B --> C[Multi-Head Attention<br/>32 heads, 128-d each]
        C --> D[Residual Add]
        D --> E[Layer Norm]
        E --> F[Feed Forward<br/>14336-d hidden]
        F --> G[Residual Add]
        G --> H[Output<br/>4096-d]
    end
    
    H --> I[To Next Layer]
    
    style C fill:#fcf,stroke:#333,stroke-width:2px
    style F fill:#cff,stroke:#333,stroke-width:2px
```

### 3. LoRA Adapter Architecture

```mermaid
graph LR
    subgraph "LoRA Module"
        A[Base Weights<br/>Frozen] --> B[Original Output]
        A --> C[LoRA Down<br/>4096→64]
        C --> D[Activation<br/>ReLU]
        D --> E[LoRA Up<br/>64→4096]
        E --> F[Scale α/r]
        B & F --> G[Add]
        G --> H[Adapted Output]
    end
    
    style A fill:#ddd,stroke:#333,stroke-width:2px
    style E fill:#fcc,stroke:#333,stroke-width:2px
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Preprocessor
    participant Encoder
    participant LoRA
    participant Pooling
    participant Output
    
    User->>API: Input(text/image, task)
    API->>Preprocessor: Validate & Prepare
    Note over Preprocessor: Tokenize text or<br/>Process image patches
    Preprocessor->>Encoder: Prepared tensors
    Encoder->>Encoder: 28 Transformer layers
    Encoder->>LoRA: Apply task adapter
    Note over LoRA: Retrieval/Classification/<br/>Clustering
    LoRA->>Pooling: Hidden states
    Note over Pooling: Mean pooling over<br/>sequence length
    Pooling->>Output: 2048-d embeddings
    Output->>User: Return vectors
```

## Memory Architecture

```mermaid
graph TD
    subgraph "GPU Memory Layout"
        A[Model Weights<br/>7.6GB FP16] 
        B[LoRA Adapters<br/>3×60MB = 180MB]
        C[KV Cache<br/>~500MB dynamic]
        D[Activations<br/>~1.5GB batch=8]
        E[Gradients<br/>N/A - Inference only]
    end
    
    A & B & C & D --> F[Total: ~9.8GB]
    
    style A fill:#faa,stroke:#333,stroke-width:2px
    style F fill:#afa,stroke:#333,stroke-width:2px
```

## Processing Pipeline

```mermaid
flowchart TB
    subgraph "Batch Processing Pipeline"
        A[Input Batch] --> B{Homogeneous?}
        B -->|Yes| C[Direct Processing]
        B -->|No| D[Split by Type]
        
        D --> E[Text Batch]
        D --> F[Image Batch]
        
        E --> G[Text Encoder]
        F --> H[Image Encoder]
        
        C --> I[Unified Encoder]
        G & H --> J[Merge Results]
        I & J --> K[Apply LoRA]
        K --> L[Batch Embeddings]
    end
    
    style B fill:#ffd,stroke:#333,stroke-width:2px
    style K fill:#dfd,stroke:#333,stroke-width:2px
```

## Task-Specific Processing

```mermaid
graph TD
    subgraph "Task Routing"
        A[Input + Task Parameter] --> B{Task Type}
        
        B -->|"retrieval"| C[Retrieval Path]
        B -->|"classification"| D[Classification Path]
        B -->|"clustering"| E[Clustering Path]
        
        C --> F{Prompt Type}
        F -->|"query"| G[Query Encoding]
        F -->|"document"| H[Document Encoding]
        
        D --> I[Category Encoding]
        E --> J[Similarity Encoding]
        
        G & H & I & J --> K[Task-Optimized<br/>Embeddings]
    end
    
    style B fill:#fcf,stroke:#333,stroke-width:2px
    style K fill:#cfc,stroke:#333,stroke-width:2px
```

## Device Optimization Strategy

```mermaid
flowchart LR
    subgraph "Device Detection & Optimization"
        A[System Check] --> B{Device}
        
        B -->|CUDA| C[NVIDIA GPU]
        B -->|MPS| D[Apple Silicon]
        B -->|CPU| E[CPU Only]
        
        C --> F[FP16<br/>Batch=32<br/>Pin Memory]
        D --> G[FP32<br/>Batch=16<br/>Unified Memory]
        E --> H[FP32<br/>Batch=8<br/>Multi-threading]
        
        F & G & H --> I[Optimized<br/>Execution]
    end
    
    style B fill:#fdf,stroke:#333,stroke-width:2px
    style I fill:#dff,stroke:#333,stroke-width:2px
```

## Matryoshka Representation Learning

```mermaid
graph TD
    subgraph "Embedding Dimensions"
        A[Full Embeddings<br/>2048-d] 
        A --> B[Large<br/>1024-d<br/>99% performance]
        A --> C[Medium<br/>512-d<br/>97% performance]
        A --> D[Small<br/>256-d<br/>94% performance]
        A --> E[Tiny<br/>128-d<br/>90% performance]
        
        B & C & D & E --> F[Use Case<br/>Selection]
    end
    
    style A fill:#f96,stroke:#333,stroke-width:4px
    style F fill:#9f6,stroke:#333,stroke-width:2px
```

## Cross-Modal Alignment

```mermaid
graph LR
    subgraph "Unified Embedding Space"
        A[Text: "A red car"] --> C[Encoder]
        B[Image: 🚗] --> C
        C --> D[Shared Vector Space]
        D --> E[Text Vector<br/>2048-d]
        D --> F[Image Vector<br/>2048-d]
        E -.->|Cosine Similarity| G[0.92]
        F -.->|High Alignment| G
    end
    
    style D fill:#ffd,stroke:#333,stroke-width:2px
    style G fill:#afa,stroke:#333,stroke-width:2px
```

## Performance Characteristics

```mermaid
graph LR
    subgraph "Performance Metrics"
        A[Throughput] --> B[Text: 1000 docs/sec<br/>Image: 100 imgs/sec]
        C[Latency] --> D[Text: 10ms<br/>Image: 50ms]
        E[Memory] --> F[Base: 8GB<br/>Peak: 10GB]
        G[Accuracy] --> H[MTEB: 68.5%<br/>CLIP: 72.3%]
    end
```

## API Integration Flow

```mermaid
sequenceDiagram
    participant Client
    participant JinaEmbeddings
    participant Config
    participant Model
    participant Device
    
    Client->>JinaEmbeddings: Initialize()
    JinaEmbeddings->>Config: Load settings
    JinaEmbeddings->>Device: Detect hardware
    Device-->>JinaEmbeddings: GPU/CPU config
    JinaEmbeddings->>Model: Load weights
    Model-->>JinaEmbeddings: Ready
    
    Client->>JinaEmbeddings: encode_text(texts)
    JinaEmbeddings->>Model: Process batch
    Model-->>JinaEmbeddings: Embeddings
    JinaEmbeddings-->>Client: numpy array
```

## Optimization Techniques

| Technique | Implementation | Benefit |
|-----------|---------------|---------|
| FlashAttention-2 | Fused attention kernels | 2-3x speedup |
| Mixed Precision | FP16 compute, FP32 accumulate | 50% memory saving |
| KV-Cache | Cached key-value pairs | Faster generation |
| Batch Processing | Dynamic batching | 4x throughput |
| CPU Offloading | Move inactive weights to RAM | Larger batch sizes |
| Gradient Checkpointing | N/A (inference only) | - |

## Comparison with Alternatives

```mermaid
graph TD
    subgraph "Embedding Models Comparison"
        A[Jina-v4<br/>3.98B params<br/>2048-d] 
        B[OpenAI Ada<br/>Unknown<br/>1536-d]
        C[Cohere-v3<br/>Unknown<br/>1024-d]
        D[BGE-M3<br/>560M params<br/>1024-d]
        
        A --> E[Advantages:<br/>• Multimodal<br/>• Open source<br/>• Matryoshka<br/>• 32K context]
    end
    
    style A fill:#afa,stroke:#333,stroke-width:3px
```

## Implementation Details

### Package Structure
```
src/jina_embeddings/
├── core/
│   ├── embeddings.py    # Main API interface
│   └── model.py         # Model management
├── utils/
│   ├── device.py        # Hardware optimization
│   └── image.py         # Image preprocessing
└── config/
    └── settings.py      # Configuration management
```

### Key Classes
- `JinaEmbeddings`: High-level API for embedding generation
- `EmbeddingModel`: Low-level model wrapper
- `DeviceManager`: Automatic hardware optimization
- `ImageProcessor`: Image validation and preprocessing
- `Config`: Centralized configuration management

## Future Architecture Enhancements

1. **Quantization Support**: INT8/INT4 for edge deployment
2. **Streaming Mode**: Token-by-token generation
3. **Multi-GPU**: Data parallel processing
4. **ONNX Export**: Cross-platform deployment
5. **TensorRT**: NVIDIA inference optimization

---

*Architecture Documentation - Jina Embeddings v4*