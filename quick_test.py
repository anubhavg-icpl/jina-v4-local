#!/usr/bin/env python3
"""
Quick Test for Jina Embeddings v4 - Faster Loading

This script provides a streamlined test with progress indicators.

Author: Claude
Date: 2025
"""

import torch
import time
from transformers import AutoModel
import os

def quick_test():
    print("🚀 Quick Test - Jina Embeddings v4")
    print("=" * 50)
    
    print("📋 System Info:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   MPS Available: {torch.backends.mps.is_available()}")
    print(f"   Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    
    print("\n⏳ Loading model (this may take 1-2 minutes first time)...")
    start_time = time.time()
    
    try:
        # Load with minimal configuration
        model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True  # Optimize loading
        )
        
        # Move to appropriate device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps":
            print("📱 Using MPS (Apple Silicon GPU)")
        else:
            print("💻 Using CPU")
            
        model.to(device)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds!")
        
        # Quick text test
        print("\n📝 Testing text encoding...")
        texts = ["Hello Jina v4!", "Quick test"]
        
        with torch.no_grad():
            embeddings = model.encode_text(
                texts=texts,
                task="retrieval",
                prompt_name="query",
                return_numpy=True
            )
        
        print(f"✅ Text embeddings: {embeddings.shape}")
        print(f"   Sample values: [{embeddings[0][:3]}, ...]")
        
        # Test similarity
        import numpy as np
        sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        print(f"   Similarity: {sim:.4f}")
        
        print("\n🎉 SUCCESS! Jina v4 is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure stable internet for model download")
        print("2. Check available memory (need ~8GB)")
        print("3. Try: pip install --upgrade transformers torch")
        return False

def check_cache():
    """Check if model is already cached"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        jina_dirs = [d for d in os.listdir(cache_dir) if "jina" in d.lower()]
        if jina_dirs:
            print("📦 Found cached Jina models:")
            for d in jina_dirs:
                print(f"   - {d}")
            return True
    print("📥 No cached model found - will download (~8GB)")
    return False

if __name__ == "__main__":
    print("🧪 Jina v4 Quick Test")
    print("=" * 30)
    
    # Check cache first
    has_cache = check_cache()
    if not has_cache:
        response = input("\nDownload model now? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            exit()
    
    print("\nStarting test...")
    success = quick_test()
    
    if success:
        print("\n✅ Ready to run full examples:")
        print("   python3 hello_world.py")
        print("   python3 examples/text_similarity.py")
    else:
        print("\n❌ Setup needs attention - check error messages above")