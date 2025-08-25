#!/usr/bin/env python3
"""
Complete Jina Embeddings v4 Options and Configurations Guide

This script demonstrates ALL available options, configurations, and use cases
for Jina Embeddings v4, including memory optimization for Apple Silicon.

Author: Claude
Date: 2025
"""

import torch
import numpy as np
from transformers import AutoModel
import os
import warnings
from typing import List, Union, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class JinaEmbeddingsV4Complete:
    """Complete implementation with all Jina v4 options"""
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v4",
                 device: str = "auto",
                 torch_dtype: str = "float32",
                 low_memory: bool = True,
                 cache_dir: str = None,
                 trust_remote_code: bool = True,
                 local_files_only: bool = False):
        """
        Initialize Jina Embeddings v4 with all available options
        
        Args:
            model_name: Model identifier or path
            device: Device to use ("auto", "cpu", "mps", "cuda")
            torch_dtype: Precision ("float16", "float32", "bfloat16")
            low_memory: Use memory optimization
            cache_dir: Custom cache directory
            trust_remote_code: Allow custom code execution
            local_files_only: Only use cached files
        """
        print(f"🚀 Loading Jina Embeddings v4 with options:")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        print(f"   Precision: {torch_dtype}")
        print(f"   Low Memory: {low_memory}")
        
        # Set memory optimization for MPS
        if device == "auto" and torch.backends.mps.is_available():
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            print("   🍎 MPS memory optimization enabled")
        
        # Configure torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float32)
        
        # Load model with optimizations
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "local_files_only": local_files_only,
        }
        
        if low_memory:
            load_kwargs.update({
                "low_cpu_mem_usage": True,
                "device_map": "auto" if device == "auto" else None
            })
            
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
            
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        
        # Set device
        self.device = self._get_device(device)
        if not low_memory or device != "auto":
            self.model.to(self.device)
            
        print(f"✅ Model loaded on {self.device}")
        
    def _get_device(self, device_pref: str) -> str:
        """Determine the best device"""
        if device_pref == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_pref
    
    def encode_text(self, 
                    texts: Union[str, List[str]], 
                    task: str = "retrieval",
                    prompt_name: str = "query",
                    batch_size: int = 8,
                    max_length: int = 8192,
                    truncate_dim: int = None,
                    normalize: bool = False,
                    return_numpy: bool = True,
                    show_progress: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text with all available options
        
        Args:
            texts: Input text(s)
            task: Task type ("retrieval", "classification", "clustering")
            prompt_name: Prompt type ("query", "document")  
            batch_size: Batch size for processing
            max_length: Maximum token length
            truncate_dim: Truncate to specific dimensions
            normalize: L2 normalize embeddings
            return_numpy: Return numpy array vs torch tensor
            show_progress: Show progress bar
            
        Returns:
            Embeddings array/tensor
        """
        if isinstance(texts, str):
            texts = [texts]
            
        print(f"📝 Encoding {len(texts)} text(s) with:")
        print(f"   Task: {task}")
        print(f"   Prompt: {prompt_name}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max length: {max_length}")
        if truncate_dim:
            print(f"   Truncate to: {truncate_dim} dimensions")
            
        embeddings = self.model.encode_text(
            texts=texts,
            task=task,
            prompt_name=prompt_name,
            batch_size=batch_size,
            max_length=max_length,
            return_numpy=return_numpy,
            show_progress_bar=show_progress
        )
        
        # Apply truncation if requested (Matryoshka)
        if truncate_dim and truncate_dim < embeddings.shape[-1]:
            embeddings = embeddings[..., :truncate_dim]
            print(f"   ✂️ Truncated to {truncate_dim} dimensions")
            
        # Apply normalization if requested
        if normalize:
            if return_numpy:
                norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                embeddings = embeddings / norms
            else:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            print(f"   🔄 L2 normalized")
            
        print(f"✅ Generated embeddings: {embeddings.shape}")
        return embeddings
    
    def encode_image(self, 
                     images: Union[str, List[str]], 
                     task: str = "retrieval",
                     batch_size: int = 4,
                     max_pixels: int = 20_000_000,
                     truncate_dim: int = None,
                     normalize: bool = False,
                     return_numpy: bool = True,
                     show_progress: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode images with all available options
        
        Args:
            images: Image path(s) or PIL images
            task: Task type ("retrieval", "classification", "clustering") 
            batch_size: Batch size for processing
            max_pixels: Maximum image resolution
            truncate_dim: Truncate to specific dimensions
            normalize: L2 normalize embeddings
            return_numpy: Return numpy array vs torch tensor
            show_progress: Show progress bar
            
        Returns:
            Embeddings array/tensor
        """
        if isinstance(images, str):
            images = [images]
            
        # Validate image paths
        valid_images = []
        for img in images:
            if isinstance(img, str) and os.path.exists(img):
                valid_images.append(img)
            elif hasattr(img, 'size'):  # PIL Image
                valid_images.append(img)
            else:
                print(f"⚠️ Invalid image: {img}")
                
        if not valid_images:
            print("❌ No valid images found!")
            return np.array([]) if return_numpy else torch.tensor([])
            
        print(f"🖼️ Encoding {len(valid_images)} image(s) with:")
        print(f"   Task: {task}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max pixels: {max_pixels:,}")
        if truncate_dim:
            print(f"   Truncate to: {truncate_dim} dimensions")
            
        # Use CPU for image encoding to avoid MPS issues
        original_device = next(self.model.parameters()).device
        if str(original_device) == "mps:0":
            self.model.to("cpu")
            print("   📱 Using CPU for image encoding (MPS compatibility)")
            
        try:
            embeddings = self.model.encode_image(
                images=valid_images,
                task=task,
                batch_size=batch_size,
                max_pixels=max_pixels,
                return_numpy=return_numpy,
                show_progress_bar=show_progress
            )
            
            # Apply truncation if requested
            if truncate_dim and truncate_dim < embeddings.shape[-1]:
                embeddings = embeddings[..., :truncate_dim]
                print(f"   ✂️ Truncated to {truncate_dim} dimensions")
                
            # Apply normalization if requested  
            if normalize:
                if return_numpy:
                    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                    embeddings = embeddings / norms
                else:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                print(f"   🔄 L2 normalized")
                
        finally:
            # Move model back if needed
            if str(original_device) == "mps:0":
                self.model.to(original_device)
                
        print(f"✅ Generated embeddings: {embeddings.shape}")
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            "model_name": self.model.config.name_or_path if hasattr(self.model, 'config') else "Unknown",
            "model_type": type(self.model).__name__,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "memory_footprint": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3),
            "embedding_dim": getattr(self.model.config, 'hidden_size', 2048) if hasattr(self.model, 'config') else 2048,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 32768) if hasattr(self.model, 'config') else 32768,
        }
        return info


def demonstrate_all_options():
    """Demonstrate all available Jina v4 options"""
    
    print("=" * 80)
    print("🌟 JINA EMBEDDINGS V4 - COMPLETE OPTIONS GUIDE")
    print("=" * 80)
    
    # 1. INITIALIZATION OPTIONS
    print("\n1️⃣ INITIALIZATION OPTIONS")
    print("-" * 40)
    
    init_options = {
        "Standard (CPU)": {
            "device": "cpu",
            "torch_dtype": "float32",
            "low_memory": False
        },
        "Apple Silicon (Optimized)": {
            "device": "auto", 
            "torch_dtype": "float32",
            "low_memory": True
        },
        "Memory Efficient": {
            "device": "auto",
            "torch_dtype": "float16", 
            "low_memory": True
        },
        "Offline Mode": {
            "device": "cpu",
            "local_files_only": True
        }
    }
    
    for name, options in init_options.items():
        print(f"   {name}:")
        for key, value in options.items():
            print(f"     {key}: {value}")
        print()
    
    # Initialize with memory-optimized settings
    try:
        jina = JinaEmbeddingsV4Complete(
            device="auto",
            torch_dtype="float32", 
            low_memory=True
        )
        
        # 2. MODEL INFORMATION
        print("\n2️⃣ MODEL INFORMATION") 
        print("-" * 40)
        info = jina.get_model_info()
        for key, value in info.items():
            if key == "parameters":
                print(f"   {key}: {value:,}")
            elif key == "memory_footprint":
                print(f"   {key}: {value:.2f} GB")
            else:
                print(f"   {key}: {value}")
        
        # 3. TEXT ENCODING OPTIONS
        print("\n3️⃣ TEXT ENCODING OPTIONS")
        print("-" * 40)
        
        sample_texts = [
            "Machine learning and artificial intelligence",
            "Natural language processing with transformers", 
            "Computer vision and image recognition"
        ]
        
        text_configs = [
            {
                "name": "Standard Retrieval",
                "task": "retrieval",
                "prompt_name": "query",
                "batch_size": 8
            },
            {
                "name": "Document Encoding", 
                "task": "retrieval",
                "prompt_name": "document",
                "batch_size": 4
            },
            {
                "name": "Classification Task",
                "task": "classification", 
                "batch_size": 16
            },
            {
                "name": "Matryoshka (512d)",
                "task": "retrieval",
                "truncate_dim": 512,
                "normalize": True
            }
        ]
        
        for config in text_configs[:2]:  # Limit to 2 for demo
            print(f"\n   🔤 {config['name']}:")
            config_copy = config.copy()
            name = config_copy.pop('name')
            
            embeddings = jina.encode_text(sample_texts[:1], **config_copy)
            print(f"      Result: {embeddings.shape}")
        
        # 4. IMAGE ENCODING OPTIONS (if images available)
        print("\n4️⃣ IMAGE ENCODING OPTIONS")
        print("-" * 40)
        
        # Check for sample images
        sample_images = []
        for img_name in ["tech_concepts.png", "nature_scene.png", "abstract_art.png"]:
            img_path = f"assets/{img_name}"
            if os.path.exists(img_path):
                sample_images.append(img_path)
        
        if sample_images:
            image_configs = [
                {
                    "name": "Standard Retrieval",
                    "task": "retrieval",
                    "batch_size": 2
                },
                {
                    "name": "Matryoshka (256d)",
                    "task": "retrieval",
                    "truncate_dim": 256,
                    "normalize": True
                }
            ]
            
            for config in image_configs[:1]:  # Limit to 1 for demo
                print(f"\n   🖼️ {config['name']}:")
                config_copy = config.copy()
                name = config_copy.pop('name')
                
                embeddings = jina.encode_image(sample_images[:1], **config_copy)
                print(f"      Result: {embeddings.shape}")
        else:
            print("   ⚠️ No sample images found in assets/")
        
        # 5. TASK TYPES
        print("\n5️⃣ AVAILABLE TASK TYPES")
        print("-" * 40)
        
        tasks = {
            "retrieval": "Asymmetric query-document matching (default)",
            "classification": "For classification and labeling tasks", 
            "clustering": "For clustering and grouping similar items"
        }
        
        for task, description in tasks.items():
            print(f"   {task}: {description}")
        
        # 6. PROMPT NAMES
        print("\n6️⃣ PROMPT NAME OPTIONS") 
        print("-" * 40)
        
        prompts = {
            "query": "For search queries (shorter texts)",
            "document": "For documents being searched (longer texts)"
        }
        
        for prompt, description in prompts.items():
            print(f"   {prompt}: {description}")
        
        # 7. DIMENSION OPTIONS
        print("\n7️⃣ DIMENSION OPTIONS (MATRYOSHKA)")
        print("-" * 40)
        
        dimensions = [2048, 1024, 512, 256, 128]
        print("   Available truncation sizes:")
        for dim in dimensions:
            performance = {
                2048: "100% (full)",
                1024: "99.2%", 
                512: "97.6%",
                256: "94.4%",
                128: "89.5%"
            }
            print(f"   {dim:4d}d: ~{performance.get(dim, '~85%')} performance")
        
        # 8. MEMORY OPTIMIZATION
        print("\n8️⃣ MEMORY OPTIMIZATION OPTIONS")
        print("-" * 40)
        
        memory_tips = [
            "Use low_memory=True for large models",
            "Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 for MPS",
            "Use float16 to halve memory usage", 
            "Process in smaller batches",
            "Use CPU for images on MPS to avoid errors",
            "Enable device_map='auto' for automatic placement"
        ]
        
        for tip in memory_tips:
            print(f"   • {tip}")
        
        print("\n" + "=" * 80)
        print("🎉 COMPLETE GUIDE FINISHED")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Memory Optimization Tips:")
        print("1. export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
        print("2. Use device='cpu' for stability")
        print("3. Reduce batch_size to 1-2")
        print("4. Close other applications to free memory")
        return False


def show_quick_examples():
    """Show quick usage examples for each option"""
    
    print("\n" + "=" * 60)
    print("📚 QUICK USAGE EXAMPLES")
    print("=" * 60)
    
    examples = {
        "Basic Text": '''
# Basic text encoding
jina = JinaEmbeddingsV4Complete()
embeddings = jina.encode_text("Hello world")
        ''',
        
        "Memory Optimized": '''
# Memory optimized for Apple Silicon  
jina = JinaEmbeddingsV4Complete(
    device="auto",
    torch_dtype="float32",
    low_memory=True
)
        ''',
        
        "Matryoshka Embeddings": '''
# Truncated embeddings (faster similarity)
embeddings = jina.encode_text(
    texts=["Sample text"],
    truncate_dim=512,
    normalize=True
)
        ''',
        
        "Batch Processing": '''
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = jina.encode_text(
    texts=texts,
    batch_size=8,
    task="retrieval"
)
        ''',
        
        "Image Encoding": '''
# Encode images
embeddings = jina.encode_image(
    images=["image1.jpg", "image2.png"], 
    task="retrieval",
    batch_size=2
)
        ''',
        
        "Cross-Modal Search": '''
# Text-to-image similarity
text_emb = jina.encode_text("A cat")
img_emb = jina.encode_image("cat.jpg")
similarity = np.dot(text_emb[0], img_emb[0])
        '''
    }
    
    for title, code in examples.items():
        print(f"\n{title}:")
        print(code.strip())


if __name__ == "__main__":
    # Set memory optimization
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    success = demonstrate_all_options()
    
    if success:
        show_quick_examples()
    
    print(f"\n📖 For detailed documentation, see:")
    print(f"   docs/api_reference.md")
    print(f"   docs/getting_started.md")
    print(f"   docs/architecture.md")