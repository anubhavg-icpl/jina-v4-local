#!/usr/bin/env python3
"""
Jina Embeddings v4 Hello World Example

This script demonstrates basic usage of Jina Embeddings v4 for:
1. Text embedding generation
2. Image embedding generation
3. Similarity calculations
4. Multi-modal retrieval

Author: Claude
Date: 2025
"""

import torch
import numpy as np
from transformers import AutoModel
from PIL import Image
import os
from typing import List, Union
import time


class JinaEmbeddingsV4:
    """Wrapper class for Jina Embeddings v4 model"""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v4", offline: bool = False):
        """Initialize the Jina Embeddings v4 model
        
        Args:
            model_name: HuggingFace model identifier or local path
            offline: If True, only use locally cached model (no internet required)
        """
        print("🚀 Loading Jina Embeddings v4...")
        if offline:
            print("📦 Running in offline mode (using cached model)")
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            local_files_only=offline  # Use cached model if offline=True
        )
        
        # Set device (MPS for Apple Silicon, CPU otherwise)
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🍎 Using Apple Silicon GPU (MPS)")
        else:
            self.device = "cpu"
            print("💻 Using CPU")
            
        self.model.to(self.device)
        print("✅ Model loaded successfully!")
        
    def encode_text(self, texts: Union[str, List[str]], task: str = "retrieval") -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
            
        print(f"📝 Encoding {len(texts)} text(s)...")
        embeddings = self.model.encode_text(
            texts=texts,
            task=task,
            prompt_name="query",
            return_numpy=True
        )
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def encode_image(self, image_paths: Union[str, List[str]], task: str = "retrieval") -> np.ndarray:
        """Encode images into embeddings"""
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        # Validate image paths
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"⚠️  Image not found: {path}")
                
        if not valid_paths:
            print("❌ No valid images found!")
            return np.array([])
            
        print(f"🖼️  Encoding {len(valid_paths)} image(s)...")
        embeddings = self.model.encode_image(
            images=valid_paths,
            task=task,
            return_numpy=True
        )
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_sample_image():
    """Create a simple sample image for testing"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a simple image with text
    img = Image.new('RGB', (400, 200), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        # Try to use default font
        draw.text((50, 80), "Hello Jina v4!", fill='darkblue', anchor="mm")
        draw.text((50, 120), "Sample Image", fill='darkblue', anchor="mm")
    except:
        # Fallback if font issues
        draw.text((50, 80), "Hello Jina v4!", fill='darkblue')
        draw.text((50, 120), "Sample Image", fill='darkblue')
    
    # Save the image
    sample_path = "assets/sample_image.png"
    os.makedirs("assets", exist_ok=True)
    img.save(sample_path)
    print(f"📷 Created sample image: {sample_path}")
    return sample_path


def main():
    """Main hello world demonstration"""
    print("=" * 60)
    print("🌟 JINA EMBEDDINGS V4 - HELLO WORLD 🌟")
    print("=" * 60)
    
    try:
        # Initialize the model
        jina = JinaEmbeddingsV4()
        
        # 1. TEXT EMBEDDINGS DEMO
        print("\n" + "=" * 40)
        print("📝 TEXT EMBEDDING DEMO")
        print("=" * 40)
        
        texts = [
            "Hello World! This is Jina Embeddings v4.",
            "Artificial intelligence and machine learning",
            "Python programming and data science",
            "Hello World! This is Jina Embeddings v4."  # Duplicate for similarity test
        ]
        
        start_time = time.time()
        text_embeddings = jina.encode_text(texts)
        end_time = time.time()
        
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
        print(f"📊 Embedding dimensions: {text_embeddings.shape[1]}")
        
        # Calculate similarity between first and last text (they're identical)
        if len(text_embeddings) >= 2:
            similarity = jina.cosine_similarity(text_embeddings[0], text_embeddings[-1])
            print(f"🔗 Similarity between identical texts: {similarity:.4f}")
            
            # Calculate similarity between different texts
            similarity_diff = jina.cosine_similarity(text_embeddings[0], text_embeddings[1])
            print(f"🔗 Similarity between different texts: {similarity_diff:.4f}")
        
        # 2. IMAGE EMBEDDINGS DEMO
        print("\n" + "=" * 40)
        print("🖼️  IMAGE EMBEDDING DEMO")
        print("=" * 40)
        
        # Create sample image
        sample_image_path = create_sample_image()
        
        start_time = time.time()
        image_embeddings = jina.encode_image([sample_image_path])
        end_time = time.time()
        
        if len(image_embeddings) > 0:
            print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
            print(f"📊 Embedding dimensions: {image_embeddings.shape[1]}")
        
        # 3. CROSS-MODAL SIMILARITY
        if len(text_embeddings) > 0 and len(image_embeddings) > 0:
            print("\n" + "=" * 40)
            print("🔄 CROSS-MODAL SIMILARITY DEMO")
            print("=" * 40)
            
            # Compare text about image with the actual image
            cross_modal_similarity = jina.cosine_similarity(
                text_embeddings[0],  # "Hello World! This is Jina Embeddings v4."
                image_embeddings[0]  # Sample image with "Hello Jina v4!"
            )
            print(f"🔗 Text-Image similarity: {cross_modal_similarity:.4f}")
        
        # 4. EMBEDDING STATISTICS
        print("\n" + "=" * 40)
        print("📈 EMBEDDING STATISTICS")
        print("=" * 40)
        
        if len(text_embeddings) > 0:
            print(f"📝 Text embeddings:")
            print(f"   Shape: {text_embeddings.shape}")
            print(f"   Mean: {np.mean(text_embeddings):.4f}")
            print(f"   Std:  {np.std(text_embeddings):.4f}")
            print(f"   Min:  {np.min(text_embeddings):.4f}")
            print(f"   Max:  {np.max(text_embeddings):.4f}")
        
        if len(image_embeddings) > 0:
            print(f"🖼️  Image embeddings:")
            print(f"   Shape: {image_embeddings.shape}")
            print(f"   Mean: {np.mean(image_embeddings):.4f}")
            print(f"   Std:  {np.std(image_embeddings):.4f}")
            print(f"   Min:  {np.min(image_embeddings):.4f}")
            print(f"   Max:  {np.max(image_embeddings):.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 HELLO WORLD DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Make sure you have installed all requirements:")
        print("   pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()