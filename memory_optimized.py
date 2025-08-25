#!/usr/bin/env python3
"""
Memory-Optimized Jina Embeddings v4 for Apple Silicon

This script provides memory-optimized configurations that work reliably
on Apple Silicon Macs with limited memory.

Author: Claude
Date: 2025
"""

import torch
import numpy as np
import os
import gc
from transformers import AutoModel

# Set memory optimizations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "page"

class JinaV4MemoryOptimized:
    """Memory-optimized Jina v4 for Apple Silicon"""
    
    def __init__(self):
        print("🚀 Loading Jina v4 (Memory Optimized)...")
        print("   🍎 Apple Silicon MPS optimizations enabled")
        print("   🧠 Low memory configuration active")
        
        # Load model with memory optimizations
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"  # Start on CPU
        )
        
        # Use hybrid approach: text on MPS, images on CPU
        self.text_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.image_device = "cpu"
        
        print(f"✅ Model loaded successfully!")
        print(f"   Text device: {self.text_device}")
        print(f"   Image device: {self.image_device}")
        
    def encode_text_optimized(self, texts, batch_size=2):
        """Memory-optimized text encoding"""
        if isinstance(texts, str):
            texts = [texts]
            
        print(f"📝 Encoding {len(texts)} texts (batch_size={batch_size})...")
        
        # Move to MPS for text
        self.model.to(self.text_device)
        
        # Clear cache before encoding
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        try:
            embeddings = self.model.encode_text(
                texts=texts,
                task="retrieval",
                prompt_name="query", 
                batch_size=batch_size,
                return_numpy=True
            )
            
            print(f"✅ Generated text embeddings: {embeddings.shape}")
            return embeddings
            
        finally:
            # Clear cache after encoding
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
    
    def encode_image_optimized(self, images, batch_size=1):
        """Memory-optimized image encoding (CPU-only for stability)"""
        if isinstance(images, str):
            images = [images]
            
        # Filter valid images
        valid_images = [img for img in images if os.path.exists(img)]
        if not valid_images:
            print("❌ No valid images found")
            return np.array([])
            
        print(f"🖼️  Encoding {len(valid_images)} images (batch_size={batch_size})...")
        print("   📱 Using CPU for stability (avoiding MPS issues)")
        
        # Move to CPU for images
        self.model.to("cpu")
        
        # Clear cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        
        try:
            with torch.no_grad():
                embeddings = self.model.encode_image(
                    images=valid_images,
                    task="retrieval",
                    batch_size=batch_size,
                    return_numpy=True
                )
            
            print(f"✅ Generated image embeddings: {embeddings.shape}")
            return embeddings
            
        finally:
            gc.collect()


def memory_optimized_demo():
    """Run memory-optimized demo"""
    
    print("=" * 70)
    print("🧠 MEMORY-OPTIMIZED JINA V4 DEMO")
    print("=" * 70)
    
    # Check system memory
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon detected")
        # Try to get memory info
        try:
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            print(f"   MPS allocated: {allocated:.2f} GB")
        except:
            print("   MPS available but memory info unavailable")
    else:
        print("💻 Using CPU mode")
    
    try:
        # Initialize optimized model
        jina = JinaV4MemoryOptimized()
        
        # Test with small batch
        print("\n1️⃣ Testing text encoding (small batch)...")
        texts = [
            "Artificial intelligence and machine learning",
            "Natural language processing"
        ]
        
        text_emb = jina.encode_text_optimized(texts, batch_size=1)
        
        # Test with single image
        print("\n2️⃣ Testing image encoding...")
        test_images = []
        for img_name in ["tech_concepts.png", "nature_scene.png"]:
            img_path = f"assets/{img_name}"
            if os.path.exists(img_path):
                test_images.append(img_path)
                break
        
        if test_images:
            img_emb = jina.encode_image_optimized(test_images, batch_size=1)
            
            # Test similarity
            if len(text_emb) > 0 and len(img_emb) > 0:
                print("\n3️⃣ Testing cross-modal similarity...")
                similarity = np.dot(text_emb[0], img_emb[0]) / (
                    np.linalg.norm(text_emb[0]) * np.linalg.norm(img_emb[0])
                )
                print(f"🔗 Text-Image similarity: {similarity:.4f}")
        else:
            print("   ⚠️ No test images found in assets/")
        
        print("\n🎉 SUCCESS! Memory-optimized version works!")
        
        # Memory usage info
        if torch.backends.mps.is_available():
            try:
                final_allocated = torch.mps.current_allocated_memory() / (1024**3)
                print(f"\n📊 Final MPS memory: {final_allocated:.2f} GB")
            except:
                pass
                
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Try these fixes:")
        print("1. export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
        print("2. Close other applications")
        print("3. Restart terminal and try again")
        print("4. Use CPU-only mode:")
        print("   python3 -c 'import torch; print(\"CPU:\", not torch.backends.mps.is_available())'")
        return False


def show_memory_config_options():
    """Show all memory configuration options"""
    
    print("\n" + "=" * 70)
    print("⚙️ MEMORY CONFIGURATION OPTIONS")
    print("=" * 70)
    
    configs = {
        "Ultra Low Memory (CPU only)": {
            "device": "cpu",
            "torch_dtype": "float16",
            "batch_size": 1,
            "description": "Most stable, slowest"
        },
        
        "Balanced (Hybrid)": {
            "device": "mps for text, cpu for images", 
            "torch_dtype": "float32",
            "batch_size": 2,
            "description": "Good balance of speed/stability"
        },
        
        "Performance (MPS)": {
            "device": "mps",
            "torch_dtype": "float32", 
            "batch_size": 4,
            "description": "Fastest, may need memory optimization"
        }
    }
    
    for name, config in configs.items():
        print(f"\n🔧 {name}:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    print(f"\n📝 Environment Variables:")
    env_vars = {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0 (disable memory limit)",
        "PYTORCH_MPS_ALLOCATOR_POLICY": "page (optimize allocation)",
        "HF_HOME": "/custom/path (change model cache location)"
    }
    
    for var, desc in env_vars.items():
        print(f"   export {var}={desc}")


if __name__ == "__main__":
    success = memory_optimized_demo()
    
    if success:
        show_memory_config_options()
        
    print(f"\n🏃‍♂️ To run this optimized version:")
    print(f"   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
    print(f"   python3 memory_optimized.py")