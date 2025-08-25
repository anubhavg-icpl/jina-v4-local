#!/usr/bin/env python3
"""
Jina Embeddings v4 - Hello World Example

Demonstrates basic usage of the refactored Jina Embeddings v4 package.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import numpy as np
from jina_embeddings import JinaEmbeddings, Config
from jina_embeddings.utils.image import ImageProcessor


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demonstrate_text_embeddings(model: JinaEmbeddings):
    """Demonstrate text embedding generation."""
    print_section("TEXT EMBEDDING DEMONSTRATION")
    
    texts = [
        "Hello World! This is Jina Embeddings v4.",
        "Artificial intelligence and machine learning are transforming technology.",
        "Python is a versatile programming language for data science.",
        "Hello World! This is Jina Embeddings v4.",  # Duplicate for similarity test
    ]
    
    print(f"\nEncoding {len(texts)} text samples...")
    start_time = time.time()
    
    embeddings = model.encode_text(
        texts,
        task="retrieval",
        show_progress=False
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.2f} seconds")
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    # Calculate similarities
    print("\nSimilarity Analysis:")
    
    # Identical texts should have similarity ~1.0
    sim_identical = model.cosine_similarity(embeddings[0], embeddings[3])
    print(f"  Identical texts: {sim_identical:.4f}")
    
    # Different texts should have lower similarity
    sim_different = model.cosine_similarity(embeddings[0], embeddings[1])
    print(f"  Different texts: {sim_different:.4f}")
    
    # All pairwise similarities
    print("\nPairwise Similarity Matrix:")
    for i in range(len(texts)):
        similarities = []
        for j in range(len(texts)):
            sim = model.cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(f"{sim:.2f}")
        print(f"  Text {i+1}: [{', '.join(similarities)}]")
    
    return embeddings


def demonstrate_image_embeddings(model: JinaEmbeddings):
    """Demonstrate image embedding generation."""
    print_section("IMAGE EMBEDDING DEMONSTRATION")
    
    # Create sample images
    processor = ImageProcessor()
    
    print("\nCreating sample images...")
    sample_images = []
    
    for i, (text, color) in enumerate([
        ("Jina v4", "lightblue"),
        ("AI Model", "lightgreen"),
        ("Embeddings", "lightyellow"),
    ]):
        img_path = f"outputs/sample_{i}.png"
        path = processor.create_sample_image(
            text=text,
            size=(300, 200),
            bg_color=color,
            output_path=img_path
        )
        sample_images.append(path)
        print(f"  Created: {path}")
    
    print(f"\nEncoding {len(sample_images)} images...")
    start_time = time.time()
    
    embeddings = model.encode_image(
        sample_images,
        task="retrieval",
        show_progress=False
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.2f} seconds")
    print(f"✓ Embedding shape: {embeddings.shape}")
    
    return embeddings, sample_images


def demonstrate_cross_modal(text_embeddings: np.ndarray, image_embeddings: np.ndarray):
    """Demonstrate cross-modal similarity."""
    print_section("CROSS-MODAL SIMILARITY")
    
    if len(text_embeddings) == 0 or len(image_embeddings) == 0:
        print("Skipping: Missing embeddings")
        return
    
    print("\nText-to-Image Similarities:")
    
    for i in range(min(3, len(text_embeddings))):
        for j in range(min(3, len(image_embeddings))):
            similarity = JinaEmbeddings.cosine_similarity(
                text_embeddings[i],
                image_embeddings[j]
            )
            print(f"  Text {i+1} × Image {j+1}: {similarity:.4f}")


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print_section("CONFIGURATION MANAGEMENT")
    
    config = Config()
    
    print("\nCurrent Configuration:")
    print(f"  Model: {config.model.name}")
    print(f"  Device: {config.device.preference}")
    print(f"  Batch Size: {config.performance.batch_size}")
    print(f"  Embedding Dim: {config.embedding.default_dim}")
    
    # Save configuration
    config_path = "outputs/config.json"
    config.save(config_path)
    print(f"\n✓ Configuration saved to: {config_path}")
    
    return config


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("  JINA EMBEDDINGS V4 - HELLO WORLD")
    print("  Professional Package Structure Demo")
    print("=" * 60)
    
    try:
        # Setup configuration
        config = demonstrate_configuration()
        
        # Initialize model
        print_section("MODEL INITIALIZATION")
        print("\nLoading Jina Embeddings v4...")
        
        model = JinaEmbeddings(
            device="cpu",  # Use CPU for demo
            offline_mode=False
        )
        
        print("✓ Model loaded successfully")
        
        # Run demonstrations
        text_embeddings = demonstrate_text_embeddings(model)
        image_embeddings, _ = demonstrate_image_embeddings(model)
        demonstrate_cross_modal(text_embeddings, image_embeddings)
        
        # Summary
        print_section("DEMONSTRATION COMPLETE")
        print("\n✓ Text embeddings generated")
        print("✓ Image embeddings generated")
        print("✓ Cross-modal similarities calculated")
        print("✓ Configuration management demonstrated")
        
        print("\nNext Steps:")
        print("  1. Explore examples/text_similarity.py")
        print("  2. Try examples/multimodal_search.py")
        print("  3. Run tests: make test")
        print("  4. View documentation: make docs")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure dependencies are installed: pip install -e .")
        print("  2. Check model cache: ~/.cache/models")
        print("  3. Verify internet connection for model download")
        raise


if __name__ == "__main__":
    main()