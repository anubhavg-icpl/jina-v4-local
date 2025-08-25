#!/usr/bin/env python3
"""
Offline Usage Example for Jina Embeddings v4

This script demonstrates how to use Jina Embeddings v4 in fully offline mode
after the initial model download.

Author: Claude
Date: 2025
"""

from hello_world import JinaEmbeddingsV4
import os


def check_model_cache():
    """Check if model is cached locally"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Look for jina-embeddings-v4 in cache
    if os.path.exists(cache_dir):
        cached_models = [d for d in os.listdir(cache_dir) if "jina" in d.lower()]
        if cached_models:
            print("✅ Found cached Jina models:")
            for model in cached_models:
                model_path = os.path.join(cache_dir, model)
                size_mb = sum(os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(model_path)
                            for filename in filenames) / (1024 * 1024)
                print(f"   - {model} ({size_mb:.1f} MB)")
            return True
        else:
            print("❌ No Jina models found in cache")
            return False
    else:
        print("❌ HuggingFace cache directory not found")
        return False


def offline_demo():
    """Demonstrate offline usage"""
    
    print("=" * 60)
    print("🌐 OFFLINE MODE DEMONSTRATION")
    print("=" * 60)
    
    # Check cache
    print("\n📁 Checking for cached model...")
    has_cache = check_model_cache()
    
    if not has_cache:
        print("\n⚠️  Model not cached. First run will download ~8GB.")
        print("After download, subsequent runs work offline.")
        response = input("\nDownload model now? (y/n): ")
        
        if response.lower() != 'y':
            print("Exiting. Run again when ready to download.")
            return
        
        print("\n📥 Downloading model (first time only)...")
        # This will download and cache the model
        jina = JinaEmbeddingsV4(offline=False)
        print("✅ Model downloaded and cached!")
        
    else:
        print("\n✅ Model cache found! Running in offline mode...")
        
        try:
            # Load model in offline mode
            jina = JinaEmbeddingsV4(offline=True)
            
            # Test text embedding
            print("\n📝 Testing text embedding (offline)...")
            text = "This is running completely offline!"
            text_emb = jina.encode_text(text)
            print(f"   Text embedding shape: {text_emb.shape}")
            
            # Test with sample image if exists
            sample_image = "assets/sample_image.png"
            if os.path.exists(sample_image):
                print("\n🖼️  Testing image embedding (offline)...")
                img_emb = jina.encode_image(sample_image)
                print(f"   Image embedding shape: {img_emb.shape}")
                
                # Calculate similarity
                similarity = jina.cosine_similarity(text_emb[0], img_emb[0])
                print(f"\n🔗 Cross-modal similarity: {similarity:.4f}")
            
            print("\n🎉 SUCCESS! Running fully offline!")
            print("   No internet connection needed from now on.")
            
        except Exception as e:
            if "local_files_only" in str(e):
                print("\n❌ Model not fully cached. Run with internet once to download.")
            else:
                print(f"\n❌ Error: {e}")


def show_offline_usage():
    """Show how to use offline mode in code"""
    
    print("\n" + "=" * 60)
    print("📖 HOW TO USE OFFLINE MODE")
    print("=" * 60)
    
    print("""
1. FIRST RUN (Internet Required):
   ```python
   from hello_world import JinaEmbeddingsV4
   
   # Download and cache model
   jina = JinaEmbeddingsV4(offline=False)
   ```
   
2. SUBSEQUENT RUNS (Fully Offline):
   ```python
   from hello_world import JinaEmbeddingsV4
   
   # Load from cache - no internet needed!
   jina = JinaEmbeddingsV4(offline=True)
   
   # Use normally
   embeddings = jina.encode_text("Your text here")
   ```

3. CACHE MANAGEMENT:
   - Location: ~/.cache/huggingface/hub/
   - Size: ~8GB
   - To change location: export HF_HOME=/your/custom/path
   
4. VERIFY OFFLINE MODE:
   - Disconnect from internet
   - Run: python offline_example.py
   - Should work without any network calls!
""")


if __name__ == "__main__":
    offline_demo()
    show_offline_usage()