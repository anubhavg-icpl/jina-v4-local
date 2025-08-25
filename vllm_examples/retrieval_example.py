#!/usr/bin/env python3
"""
Jina Embeddings v4 - vLLM Retrieval Adapter Example

This script demonstrates how to use jina-embeddings-v4-vllm-retrieval
for high-performance document retrieval with vLLM.

vLLM Model: jinaai/jina-embeddings-v4-vllm-retrieval
Use Case: Query-document matching, search engines, RAG systems

Author: Claude  
Date: 2025
"""

import torch
from PIL import Image
import numpy as np
import os

try:
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.inputs.data import TextPrompt
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class JinaV4VLLMRetrieval:
    """Jina v4 vLLM Retrieval Adapter Implementation"""
    
    def __init__(self, dtype="float16"):
        """Initialize the vLLM-compatible Jina v4 retrieval model"""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
            
        print("🚀 Loading Jina v4 vLLM Retrieval Model...")
        print(f"   Model: jinaai/jina-embeddings-v4-vllm-retrieval")
        print(f"   Task: Retrieval (Query ↔ Document matching)")
        print(f"   Precision: {dtype}")
        
        # Initialize vLLM model with retrieval adapter
        self.model = LLM(
            model="jinaai/jina-embeddings-v4-vllm-retrieval",
            task="embed",
            override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False),
            dtype=dtype,
            trust_remote_code=True
        )
        
        # Vision token IDs for image processing
        self.VISION_START_TOKEN_ID = 151652
        self.VISION_END_TOKEN_ID = 151653
        
        print("✅ vLLM Retrieval model loaded successfully!")
    
    def create_text_prompt(self, text: str, prompt_type: str = "Query") -> TextPrompt:
        """Create a text prompt for encoding"""
        return TextPrompt(prompt=f"{prompt_type}: {text}")
    
    def create_image_prompt(self, image_path: str) -> TextPrompt:
        """Create an image prompt for encoding"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path)
        return TextPrompt(
            prompt="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n",
            multi_modal_data={"image": image}
        )
    
    def get_embeddings(self, outputs):
        """Extract and normalize embeddings from vLLM outputs"""
        embeddings = []
        
        for output in outputs:
            if self.VISION_START_TOKEN_ID in output.prompt_token_ids:
                # Process image embeddings - gather only vision tokens
                img_start_pos = torch.where(
                    torch.tensor(output.prompt_token_ids) == self.VISION_START_TOKEN_ID
                )[0][0]
                img_end_pos = torch.where(
                    torch.tensor(output.prompt_token_ids) == self.VISION_END_TOKEN_ID
                )[0][0]
                embeddings_tensor = output.outputs.data.detach().clone()[
                    img_start_pos : img_end_pos + 1
                ]
            else:
                # Process text embeddings - use all tokens
                embeddings_tensor = output.outputs.data.detach().clone()
            
            # Pool embeddings (mean pooling)
            pooled_output = (
                embeddings_tensor.sum(dim=0, dtype=torch.float32) / embeddings_tensor.shape[0]
            )
            
            # L2 normalize
            normalized_embedding = torch.nn.functional.normalize(pooled_output, dim=-1)
            embeddings.append(normalized_embedding)
        
        return embeddings
    
    def encode(self, prompts):
        """Encode prompts and return normalized embeddings"""
        print(f"📝 Encoding {len(prompts)} prompt(s) with vLLM...")
        
        # Get raw outputs from vLLM
        outputs = self.model.encode(prompts)
        
        # Extract and normalize embeddings
        embeddings = self.get_embeddings(outputs)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        return torch.dot(emb1, emb2).item()


def retrieval_demo():
    """Demonstrate retrieval use case with vLLM"""
    
    print("=" * 80)
    print("🔍 JINA V4 vLLM RETRIEVAL DEMO")
    print("=" * 80)
    
    if not VLLM_AVAILABLE:
        print("❌ vLLM not installed!")
        print("💡 Install with: pip install vllm")
        return False
    
    try:
        # Initialize retrieval model
        retrieval_model = JinaV4VLLMRetrieval()
        
        print("\n1️⃣ TEXT-TO-TEXT RETRIEVAL")
        print("-" * 50)
        
        # Define query and passages for retrieval
        query = "Overview of climate change impacts on coastal cities"
        passages = [
            "The impacts of climate change on coastal cities are significant and multifaceted, affecting infrastructure, economy, and residents.",
            "Artificial intelligence and machine learning are revolutionizing various industries through automation and data analysis.",
            "Coastal urban areas face rising sea levels, increased flooding, and severe weather events due to climate change.",
            "The development of renewable energy sources is crucial for reducing carbon emissions and combating global warming."
        ]
        
        print(f"🔍 Query: {query}")
        print(f"📚 Searching through {len(passages)} passages...")
        
        # Create prompts
        query_prompt = retrieval_model.create_text_prompt(query, "Query")
        passage_prompts = [
            retrieval_model.create_text_prompt(passage, "Passage") 
            for passage in passages
        ]
        
        # Encode all prompts
        all_prompts = [query_prompt] + passage_prompts
        embeddings = retrieval_model.encode(all_prompts)
        
        # Calculate similarities
        query_embedding = embeddings[0]
        passage_embeddings = embeddings[1:]
        
        similarities = []
        for i, passage_emb in enumerate(passage_embeddings):
            sim = retrieval_model.compute_similarity(query_embedding, passage_emb)
            similarities.append((i, sim, passages[i]))
        
        # Rank by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Retrieval Results (ranked by relevance):")
        for rank, (idx, sim, passage) in enumerate(similarities, 1):
            print(f"{rank}. [Score: {sim:.4f}] {passage[:80]}...")
        
        print(f"\n✅ Best match: Passage {similarities[0][0] + 1} (Score: {similarities[0][1]:.4f})")
        
        # Test with image if available
        print(f"\n2️⃣ TEXT-TO-IMAGE RETRIEVAL")
        print("-" * 50)
        
        # Look for sample images
        sample_images = []
        for img_name in ["tech_concepts.png", "nature_scene.png", "abstract_art.png"]:
            img_path = f"../assets/{img_name}"
            if os.path.exists(img_path):
                sample_images.append(img_path)
        
        if sample_images:
            print(f"🖼️  Testing with image: {os.path.basename(sample_images[0])}")
            
            # Create image prompt
            image_prompt = retrieval_model.create_image_prompt(sample_images[0])
            
            # Encode query and image
            multimodal_prompts = [query_prompt, image_prompt]
            multimodal_embeddings = retrieval_model.encode(multimodal_prompts)
            
            # Calculate cross-modal similarity
            text_emb, img_emb = multimodal_embeddings
            cross_modal_sim = retrieval_model.compute_similarity(text_emb, img_emb)
            
            print(f"🔗 Text-Image similarity: {cross_modal_sim:.4f}")
            
        else:
            print("⚠️  No sample images found for cross-modal demo")
        
        print(f"\n🎉 vLLM Retrieval Demo Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Ensure vLLM is installed: pip install vllm")
        print(f"2. Check GPU memory availability")
        print(f"3. Try with CPU: set CUDA_VISIBLE_DEVICES=''")
        return False


def show_retrieval_use_cases():
    """Show practical use cases for retrieval adapter"""
    
    print(f"\n" + "=" * 80)
    print(f"📖 RETRIEVAL ADAPTER USE CASES")
    print(f"=" * 80)
    
    use_cases = {
        "Search Engines": {
            "description": "Match user queries with relevant web pages",
            "example": "Query: 'best restaurants NYC' → Document: 'Top NYC dining guide'"
        },
        
        "RAG Systems": {
            "description": "Retrieve relevant documents for LLM context",
            "example": "Question: 'How does photosynthesis work?' → Knowledge base retrieval"
        },
        
        "E-commerce": {
            "description": "Product search and recommendation",
            "example": "Query: 'wireless headphones' → Product descriptions matching"
        },
        
        "Document Q&A": {
            "description": "Find relevant sections in large documents",
            "example": "Question about legal contract → Relevant contract clauses"
        },
        
        "Code Search": {
            "description": "Find relevant code snippets",
            "example": "Query: 'sort array python' → Code examples with sorting"
        },
        
        "Cross-Modal Search": {
            "description": "Text queries matching images and vice versa",
            "example": "Text: 'sunset beach' → Matching landscape photos"
        }
    }
    
    for use_case, details in use_cases.items():
        print(f"\n🎯 {use_case}:")
        print(f"   Description: {details['description']}")
        print(f"   Example: {details['example']}")


if __name__ == "__main__":
    success = retrieval_demo()
    
    if success:
        show_retrieval_use_cases()
        
    print(f"\n📚 More Examples:")
    print(f"   vllm_examples/text_matching_example.py")
    print(f"   vllm_examples/code_search_example.py")