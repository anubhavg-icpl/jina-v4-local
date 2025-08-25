#!/usr/bin/env python3
"""
Text Similarity Example using Jina Embeddings v4

This example demonstrates how to use Jina Embeddings v4 for:
- Text embedding generation
- Semantic similarity search
- Document ranking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jina_embeddings import JinaEmbeddings
import numpy as np


def text_similarity_demo():
    """Demonstrate text similarity using Jina Embeddings v4"""
    
    print("🔍 Text Similarity Demo with Jina Embeddings v4")
    print("=" * 50)
    
    # Initialize model
    print("Loading Jina Embeddings v4...")
    jina = JinaEmbeddings()
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers", 
        "Python is a popular programming language for data science",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret visual information",
        "The weather is nice today with sunny skies",
        "I love eating pizza with extra cheese",
        "Artificial intelligence will transform the future"
    ]
    
    # Query
    query = "What is artificial intelligence and machine learning?"
    
    print(f"📄 Documents ({len(documents)}):")
    for i, doc in enumerate(documents):
        print(f"   {i+1}. {doc}")
    
    print(f"\n🔍 Query: {query}")
    
    # Encode documents and query
    print("\n📝 Generating embeddings...")
    doc_embeddings = jina.encode_text(documents, task="document")
    query_embedding = jina.encode_text([query], task="query")
    
    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        similarity = jina.cosine_similarity(query_embedding[0], doc_emb)
        similarities.append((i, similarity, documents[i]))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print(f"\n🏆 Similarity Rankings:")
    print("-" * 80)
    for rank, (doc_idx, sim_score, doc_text) in enumerate(similarities, 1):
        print(f"{rank:2d}. [Score: {sim_score:.4f}] {doc_text}")
    
    print(f"\n✅ Most similar document: '{similarities[0][2]}'")
    print(f"📊 Similarity score: {similarities[0][1]:.4f}")


if __name__ == "__main__":
    text_similarity_demo()