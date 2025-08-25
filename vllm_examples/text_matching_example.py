#!/usr/bin/env python3
"""
Jina Embeddings v4 - vLLM Text Matching Adapter Example

This script demonstrates how to use jina-embeddings-v4-vllm-text-matching
for symmetric text similarity and multilingual matching with vLLM.

vLLM Model: jinaai/jina-embeddings-v4-vllm-text-matching  
Use Case: Text similarity, multilingual matching, content deduplication

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


class JinaV4VLLMTextMatching:
    """Jina v4 vLLM Text Matching Adapter Implementation"""
    
    def __init__(self, dtype="float16"):
        """Initialize the vLLM-compatible Jina v4 text matching model"""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
            
        print("🚀 Loading Jina v4 vLLM Text Matching Model...")
        print(f"   Model: jinaai/jina-embeddings-v4-vllm-text-matching")
        print(f"   Task: Symmetric text similarity & multilingual matching")
        print(f"   Precision: {dtype}")
        
        # Initialize vLLM model with text-matching adapter
        self.model = LLM(
            model="jinaai/jina-embeddings-v4-vllm-text-matching",
            task="embed",
            override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False),
            dtype=dtype,
            trust_remote_code=True
        )
        
        # Vision token IDs for image processing
        self.VISION_START_TOKEN_ID = 151652
        self.VISION_END_TOKEN_ID = 151653
        
        print("✅ vLLM Text Matching model loaded successfully!")
    
    def create_text_prompt(self, text: str, prompt_type: str = "Query") -> TextPrompt:
        """Create a text prompt for encoding (symmetric matching)"""
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
    
    def create_similarity_matrix(self, embeddings):
        """Create a similarity matrix for all embeddings"""
        n = len(embeddings)
        similarity_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = self.compute_similarity(embeddings[i], embeddings[j])
        
        return similarity_matrix


def text_matching_demo():
    """Demonstrate text matching use cases with vLLM"""
    
    print("=" * 80)
    print("🔤 JINA V4 vLLM TEXT MATCHING DEMO")
    print("=" * 80)
    
    if not VLLM_AVAILABLE:
        print("❌ vLLM not installed!")
        print("💡 Install with: pip install vllm")
        return False
    
    try:
        # Initialize text matching model
        matching_model = JinaV4VLLMTextMatching()
        
        print("\n1️⃣ MULTILINGUAL TEXT MATCHING")
        print("-" * 50)
        
        # Multilingual examples (same meaning in different languages)
        multilingual_texts = [
            ("English", "Ein wunderschöner Sonnenuntergang am Strand"),    # German
            ("German", "A beautiful sunset at the beach"),                 # English  
            ("Japanese", "浜辺に沈む美しい夕日"),                         # Japanese
            ("Spanish", "Una hermosa puesta de sol en la playa"),         # Spanish
            ("French", "Un magnifique coucher de soleil sur la plage"),   # French
        ]
        
        print("🌍 Testing multilingual similarity:")
        for lang, text in multilingual_texts:
            print(f"   {lang}: {text}")
        
        # Create prompts for multilingual texts
        multilingual_prompts = [
            matching_model.create_text_prompt(text, "Query") 
            for _, text in multilingual_texts
        ]
        
        # Encode all multilingual texts
        multilingual_embeddings = matching_model.encode(multilingual_prompts)
        
        # Create similarity matrix
        similarity_matrix = matching_model.create_similarity_matrix(multilingual_embeddings)
        
        print(f"\n📊 Multilingual Similarity Matrix:")
        print(f"{'':>10}", end="")
        languages = [lang for lang, _ in multilingual_texts]
        for lang in languages:
            print(f"{lang[:8]:>10}", end="")
        print()
        
        for i, lang in enumerate(languages):
            print(f"{lang[:8]:>10}", end="")
            for j in range(len(languages)):
                print(f"{similarity_matrix[i][j]:.3f}     ", end="")
            print()
        
        # Find highest cross-lingual similarity
        max_sim = 0
        best_pair = None
        for i in range(len(languages)):
            for j in range(i+1, len(languages)):
                sim = similarity_matrix[i][j].item()
                if sim > max_sim:
                    max_sim = sim
                    best_pair = (languages[i], languages[j])
        
        if best_pair:
            print(f"\n🏆 Highest cross-lingual similarity: {best_pair[0]} ↔ {best_pair[1]} ({max_sim:.4f})")
        
        print(f"\n2️⃣ CONTENT SIMILARITY DETECTION") 
        print("-" * 50)
        
        # Similar content examples
        content_variations = [
            "Machine learning is transforming artificial intelligence",
            "AI is being revolutionized by machine learning techniques",
            "The weather is nice today with sunny skies",
            "ML algorithms are advancing the field of artificial intelligence",
            "Today has beautiful weather and sunshine"
        ]
        
        print("🔍 Testing content similarity:")
        for i, text in enumerate(content_variations, 1):
            print(f"   {i}. {text}")
        
        # Create prompts for content variations
        content_prompts = [
            matching_model.create_text_prompt(text, "Query") 
            for text in content_variations
        ]
        
        # Encode content variations
        content_embeddings = matching_model.encode(content_prompts)
        
        # Find similar content pairs
        similarity_threshold = 0.7
        similar_pairs = []
        
        for i in range(len(content_embeddings)):
            for j in range(i+1, len(content_embeddings)):
                sim = matching_model.compute_similarity(content_embeddings[i], content_embeddings[j])
                if sim > similarity_threshold:
                    similar_pairs.append((i+1, j+1, sim, content_variations[i], content_variations[j]))
        
        print(f"\n🔗 Similar content pairs (similarity > {similarity_threshold}):")
        for idx1, idx2, sim, text1, text2 in similar_pairs:
            print(f"   {idx1} ↔ {idx2}: {sim:.4f}")
            print(f"      \"{text1[:50]}...\"")
            print(f"      \"{text2[:50]}...\"")
            print()
        
        print(f"\n3️⃣ TEXT-IMAGE MATCHING")
        print("-" * 50)
        
        # Test with image if available
        sample_images = []
        for img_name in ["nature_scene.png", "tech_concepts.png"]:
            img_path = f"../assets/{img_name}"
            if os.path.exists(img_path):
                sample_images.append(img_path)
        
        if sample_images:
            print(f"🖼️  Testing text-image matching with: {os.path.basename(sample_images[0])}")
            
            # Text descriptions
            descriptions = [
                "A beautiful natural landscape with mountains and trees",
                "Technology and artificial intelligence concepts",
                "Ocean waves and blue water"
            ]
            
            # Create prompts
            image_prompt = matching_model.create_image_prompt(sample_images[0])
            text_prompts = [matching_model.create_text_prompt(desc, "Query") for desc in descriptions]
            
            # Encode image and texts
            multimodal_prompts = [image_prompt] + text_prompts
            multimodal_embeddings = matching_model.encode(multimodal_prompts)
            
            # Calculate similarities
            image_emb = multimodal_embeddings[0]
            text_embs = multimodal_embeddings[1:]
            
            print(f"\n🎯 Text-Image similarity scores:")
            for i, (desc, text_emb) in enumerate(zip(descriptions, text_embs)):
                sim = matching_model.compute_similarity(image_emb, text_emb)
                print(f"   {i+1}. [{sim:.4f}] {desc}")
            
        else:
            print("⚠️  No sample images found for multimodal demo")
        
        print(f"\n🎉 vLLM Text Matching Demo Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Ensure vLLM is installed: pip install vllm")
        print(f"2. Check GPU memory availability") 
        print(f"3. Try with CPU: set CUDA_VISIBLE_DEVICES=''")
        return False


def show_text_matching_use_cases():
    """Show practical use cases for text matching adapter"""
    
    print(f"\n" + "=" * 80)
    print(f"📖 TEXT MATCHING ADAPTER USE CASES")
    print(f"=" * 80)
    
    use_cases = {
        "Content Deduplication": {
            "description": "Identify and remove duplicate content",
            "example": "Detect similar news articles from different sources"
        },
        
        "Multilingual Support": {
            "description": "Match content across different languages",
            "example": "German 'Hallo Welt' ↔ English 'Hello World'"
        },
        
        "Paraphrase Detection": {
            "description": "Find semantically similar but differently worded text",
            "example": "'ML is powerful' ↔ 'Machine learning is effective'"
        },
        
        "Content Clustering": {
            "description": "Group similar content together",
            "example": "Organize articles by topic similarity"
        },
        
        "Plagiarism Detection": {
            "description": "Identify potentially copied content",
            "example": "Academic paper similarity checking"
        },
        
        "Recommendation Systems": {
            "description": "Recommend similar content to users",
            "example": "If you liked article A, you might like article B"
        },
        
        "Translation Quality": {
            "description": "Assess translation accuracy by similarity",
            "example": "Compare original text with translated version"
        }
    }
    
    for use_case, details in use_cases.items():
        print(f"\n🎯 {use_case}:")
        print(f"   Description: {details['description']}")
        print(f"   Example: {details['example']}")


if __name__ == "__main__":
    success = text_matching_demo()
    
    if success:
        show_text_matching_use_cases()
        
    print(f"\n📚 More Examples:")
    print(f"   vllm_examples/retrieval_example.py")
    print(f"   vllm_examples/code_search_example.py")