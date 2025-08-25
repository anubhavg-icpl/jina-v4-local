#!/usr/bin/env python3
"""
Multimodal Search Example using Jina Embeddings v4

This example demonstrates how to use Jina Embeddings v4 for:
- Cross-modal search (text queries on image databases)
- Image-to-image similarity
- Text-to-image matching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jina_embeddings import JinaEmbeddings
from jina_embeddings.utils.image import ImageProcessor
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def create_sample_images():
    """Create sample images for demonstration"""
    
    os.makedirs("assets/samples", exist_ok=True)
    
    images_info = [
        ("cat.png", "🐱", (255, 200, 200), "Cute Cat"),
        ("dog.png", "🐕", (200, 255, 200), "Happy Dog"), 
        ("car.png", "🚗", (200, 200, 255), "Red Car"),
        ("house.png", "🏠", (255, 255, 200), "Nice House"),
        ("tree.png", "🌳", (150, 255, 150), "Green Tree")
    ]
    
    created_paths = []
    
    for filename, emoji, color, text in images_info:
        img = Image.new('RGB', (300, 200), color=color)
        draw = ImageDraw.Draw(img)
        
        # Add emoji and text
        try:
            draw.text((150, 80), emoji, fill='black', anchor="mm", font_size=40)
            draw.text((150, 130), text, fill='black', anchor="mm")
        except:
            draw.text((120, 80), emoji, fill='black')
            draw.text((120, 130), text, fill='black')
        
        filepath = f"assets/samples/{filename}"
        img.save(filepath)
        created_paths.append(filepath)
        
    print(f"📷 Created {len(created_paths)} sample images in assets/samples/")
    return created_paths


def multimodal_search_demo():
    """Demonstrate multimodal search capabilities"""
    
    print("🎭 Multimodal Search Demo with Jina Embeddings v4")
    print("=" * 55)
    
    # Initialize model
    jina = JinaEmbeddings()
    
    # Create sample images
    image_paths = create_sample_images()
    
    # Text queries for searching images
    text_queries = [
        "a cute animal pet",
        "transportation vehicle", 
        "building or architecture",
        "nature and plants"
    ]
    
    print(f"🖼️  Sample Images:")
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        print(f"   {i+1}. {filename}")
    
    print(f"\n🔍 Text Queries:")
    for i, query in enumerate(text_queries):
        print(f"   {i+1}. {query}")
    
    # Generate embeddings
    print("\n📝 Generating image embeddings...")
    image_embeddings = jina.encode_image(image_paths, task="document")
    
    print("📝 Generating text query embeddings...")
    query_embeddings = jina.encode_text(text_queries, task="query")
    
    # Perform cross-modal search
    print(f"\n🔎 Cross-Modal Search Results:")
    print("=" * 60)
    
    for q_idx, query in enumerate(text_queries):
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        # Calculate similarities between query and all images
        similarities = []
        for img_idx, img_emb in enumerate(image_embeddings):
            similarity = jina.cosine_similarity(query_embeddings[q_idx], img_emb)
            similarities.append((img_idx, similarity, os.path.basename(image_paths[img_idx])))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 3 results
        for rank, (img_idx, sim_score, img_name) in enumerate(similarities[:3], 1):
            print(f"   {rank}. {img_name:<12} (Score: {sim_score:.4f})")
    
    # Image-to-image similarity
    print(f"\n🖼️  Image-to-Image Similarity Matrix:")
    print("=" * 50)
    
    similarity_matrix = np.zeros((len(image_paths), len(image_paths)))
    
    for i in range(len(image_embeddings)):
        for j in range(len(image_embeddings)):
            similarity = jina.cosine_similarity(image_embeddings[i], image_embeddings[j])
            similarity_matrix[i][j] = similarity
    
    # Display similarity matrix
    image_names = [os.path.basename(path).replace('.png', '') for path in image_paths]
    
    print("\nSimilarity Matrix:")
    print(f"{'':>8}", end="")
    for name in image_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(image_names):
        print(f"{name:>8}", end="")
        for j in range(len(image_names)):
            print(f"{similarity_matrix[i][j]:>8.3f}", end="")
        print()
    
    # Find most similar image pairs
    print(f"\n🏆 Most Similar Image Pairs:")
    print("-" * 30)
    
    pairs = []
    for i in range(len(image_names)):
        for j in range(i+1, len(image_names)):
            pairs.append((similarity_matrix[i][j], image_names[i], image_names[j]))
    
    pairs.sort(reverse=True)
    
    for sim_score, img1, img2 in pairs[:3]:
        print(f"   {img1} ↔ {img2}: {sim_score:.4f}")


if __name__ == "__main__":
    multimodal_search_demo()