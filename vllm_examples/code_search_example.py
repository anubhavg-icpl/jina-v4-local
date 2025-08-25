#!/usr/bin/env python3
"""
Jina Embeddings v4 - vLLM Code Search Adapter Example

This script demonstrates how to use jina-embeddings-v4-vllm-code
for natural language to code search and code similarity with vLLM.

vLLM Model: jinaai/jina-embeddings-v4-vllm-code
Use Case: Natural language ↔ code search, code similarity, documentation

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


class JinaV4VLLMCodeSearch:
    """Jina v4 vLLM Code Search Adapter Implementation"""
    
    def __init__(self, dtype="float16"):
        """Initialize the vLLM-compatible Jina v4 code search model"""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
            
        print("🚀 Loading Jina v4 vLLM Code Search Model...")
        print(f"   Model: jinaai/jina-embeddings-v4-vllm-code")
        print(f"   Task: Natural language ↔ Code search & similarity")
        print(f"   Precision: {dtype}")
        
        # Initialize vLLM model with code adapter
        self.model = LLM(
            model="jinaai/jina-embeddings-v4-vllm-code",
            task="embed",
            override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False),
            dtype=dtype,
            trust_remote_code=True
        )
        
        # Vision token IDs for image processing
        self.VISION_START_TOKEN_ID = 151652
        self.VISION_END_TOKEN_ID = 151653
        
        print("✅ vLLM Code Search model loaded successfully!")
    
    def create_text_prompt(self, text: str, prompt_type: str = "Query") -> TextPrompt:
        """Create a text/code prompt for encoding"""
        return TextPrompt(prompt=f"{prompt_type}: {text}")
    
    def create_image_prompt(self, image_path: str) -> TextPrompt:
        """Create an image prompt for encoding (for code screenshots)"""
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
                # Process text/code embeddings - use all tokens
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
        print(f"💻 Encoding {len(prompts)} code prompt(s) with vLLM...")
        
        # Get raw outputs from vLLM
        outputs = self.model.encode(prompts)
        
        # Extract and normalize embeddings
        embeddings = self.get_embeddings(outputs)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        return torch.dot(emb1, emb2).item()


def code_search_demo():
    """Demonstrate code search use cases with vLLM"""
    
    print("=" * 80)
    print("💻 JINA V4 vLLM CODE SEARCH DEMO")
    print("=" * 80)
    
    if not VLLM_AVAILABLE:
        print("❌ vLLM not installed!")
        print("💡 Install with: pip install vllm")
        return False
    
    try:
        # Initialize code search model
        code_model = JinaV4VLLMCodeSearch()
        
        print("\n1️⃣ NATURAL LANGUAGE TO CODE SEARCH")
        print("-" * 50)
        
        # Natural language query
        query = "Find a function that prints a greeting message to the console"
        print(f"🔍 Query: {query}")
        
        # Code snippets database
        code_snippets = [
            {
                "language": "Python",
                "code": "def hello_world():\n    print('Hello, World!')\n    return True"
            },
            {
                "language": "JavaScript", 
                "code": "function greetUser(name) {\n    console.log(`Hello, ${name}!`);\n}"
            },
            {
                "language": "Python",
                "code": "import math\n\ndef calculate_area(radius):\n    return math.pi * radius ** 2"
            },
            {
                "language": "Java",
                "code": "public class Calculator {\n    public static int add(int a, int b) {\n        return a + b;\n    }\n}"
            },
            {
                "language": "Python", 
                "code": "def print_greeting(name='World'):\n    message = f'Hello, {name}!'\n    print(message)"
            }
        ]
        
        print(f"📚 Searching through {len(code_snippets)} code snippets...")
        
        # Create prompts
        query_prompt = code_model.create_text_prompt(query, "Query")
        code_prompts = [
            code_model.create_text_prompt(snippet["code"], "Passage") 
            for snippet in code_snippets
        ]
        
        # Encode all prompts
        all_prompts = [query_prompt] + code_prompts
        embeddings = code_model.encode(all_prompts)
        
        # Calculate similarities
        query_embedding = embeddings[0]
        code_embeddings = embeddings[1:]
        
        similarities = []
        for i, code_emb in enumerate(code_embeddings):
            sim = code_model.compute_similarity(query_embedding, code_emb)
            similarities.append((i, sim, code_snippets[i]))
        
        # Rank by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Code Search Results (ranked by relevance):")
        for rank, (idx, sim, snippet) in enumerate(similarities, 1):
            code_preview = snippet["code"].replace('\n', '\\n')[:60] + "..."
            print(f"{rank}. [Score: {sim:.4f}] [{snippet['language']}] {code_preview}")
        
        print(f"\n✅ Best match:")
        best_snippet = similarities[0][2]
        print(f"   Language: {best_snippet['language']}")
        print(f"   Code:\n{best_snippet['code']}")
        
        print(f"\n2️⃣ CODE-TO-CODE SIMILARITY")
        print("-" * 50)
        
        # Similar function implementations in different languages
        similar_functions = [
            {
                "name": "Python Hello World",
                "code": "def hello():\n    print('Hello, World!')"
            },
            {
                "name": "JavaScript Hello World", 
                "code": "function hello() {\n    console.log('Hello, World!');\n}"
            },
            {
                "name": "Python Add Function",
                "code": "def add(a, b):\n    return a + b"
            },
            {
                "name": "Java Add Method",
                "code": "public int add(int a, int b) {\n    return a + b;\n}"
            }
        ]
        
        print("🔍 Testing cross-language code similarity:")
        for func in similar_functions:
            print(f"   {func['name']}: {func['code'].replace(chr(10), ' ')}")
        
        # Create prompts for similar functions
        function_prompts = [
            code_model.create_text_prompt(func["code"], "Query") 
            for func in similar_functions
        ]
        
        # Encode all functions
        function_embeddings = code_model.encode(function_prompts)
        
        # Calculate cross-language similarities
        print(f"\n📊 Cross-Language Code Similarity Matrix:")
        names = [func["name"] for func in similar_functions]
        
        print(f"{'':>20}", end="")
        for name in names:
            print(f"{name[:15]:>17}", end="")
        print()
        
        for i, name_i in enumerate(names):
            print(f"{name_i[:18]:>20}", end="")
            for j, name_j in enumerate(names):
                sim = code_model.compute_similarity(function_embeddings[i], function_embeddings[j])
                print(f"{sim:.3f}            ", end="")
            print()
        
        # Find most similar cross-language pair
        max_sim = 0
        best_pair = None
        for i in range(len(function_embeddings)):
            for j in range(i+1, len(function_embeddings)):
                sim = code_model.compute_similarity(function_embeddings[i], function_embeddings[j])
                if sim > max_sim:
                    max_sim = sim
                    best_pair = (names[i], names[j])
        
        if best_pair:
            print(f"\n🏆 Most similar cross-language pair:")
            print(f"   {best_pair[0]} ↔ {best_pair[1]} (Similarity: {max_sim:.4f})")
        
        print(f"\n3️⃣ DOCUMENTATION SEARCH")
        print("-" * 50)
        
        # Documentation queries
        doc_query = "How to sort a list in ascending order"
        
        documentation = [
            "sort() method sorts the list in place in ascending order by default",
            "Use reverse=True parameter in sort() to sort in descending order", 
            "sorted() function returns a new sorted list without modifying the original",
            "join() method concatenates list elements into a single string",
            "append() adds an element to the end of the list"
        ]
        
        print(f"🔍 Documentation Query: {doc_query}")
        print(f"📖 Searching through {len(documentation)} documentation entries...")
        
        # Create documentation prompts
        doc_query_prompt = code_model.create_text_prompt(doc_query, "Query")
        doc_prompts = [
            code_model.create_text_prompt(doc, "Passage") 
            for doc in documentation
        ]
        
        # Encode documentation
        doc_all_prompts = [doc_query_prompt] + doc_prompts
        doc_embeddings = code_model.encode(doc_all_prompts)
        
        # Find best documentation match
        doc_query_emb = doc_embeddings[0]
        doc_content_embs = doc_embeddings[1:]
        
        doc_similarities = []
        for i, doc_emb in enumerate(doc_content_embs):
            sim = code_model.compute_similarity(doc_query_emb, doc_emb)
            doc_similarities.append((i, sim, documentation[i]))
        
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📚 Documentation Search Results:")
        for rank, (idx, sim, doc_text) in enumerate(doc_similarities, 1):
            print(f"{rank}. [Score: {sim:.4f}] {doc_text}")
        
        # Test with code screenshot if available
        print(f"\n4️⃣ CODE IMAGE ANALYSIS")
        print("-" * 50)
        
        # Look for code-related images
        code_images = []
        for img_name in ["doc_python.png", "doc_ml.png", "tech_concepts.png"]:
            img_path = f"../assets/{img_name}"
            if os.path.exists(img_path):
                code_images.append(img_path)
        
        if code_images:
            print(f"🖼️  Analyzing code image: {os.path.basename(code_images[0])}")
            
            # Create image prompt
            image_prompt = code_model.create_image_prompt(code_images[0])
            
            # Code descriptions for matching
            code_descriptions = [
                "Python function definition with print statement",
                "Machine learning import statements and model code", 
                "JavaScript function with console.log output"
            ]
            
            # Create text prompts
            desc_prompts = [code_model.create_text_prompt(desc, "Query") for desc in code_descriptions]
            
            # Encode image and descriptions
            image_code_prompts = [image_prompt] + desc_prompts
            image_code_embeddings = code_model.encode(image_code_prompts)
            
            # Calculate similarities
            image_emb = image_code_embeddings[0]
            desc_embs = image_code_embeddings[1:]
            
            print(f"\n🎯 Code Image-Text similarity scores:")
            for i, (desc, desc_emb) in enumerate(zip(code_descriptions, desc_embs)):
                sim = code_model.compute_similarity(image_emb, desc_emb)
                print(f"   {i+1}. [{sim:.4f}] {desc}")
            
        else:
            print("⚠️  No code-related images found for analysis")
        
        print(f"\n🎉 vLLM Code Search Demo Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Ensure vLLM is installed: pip install vllm")
        print(f"2. Check GPU memory availability")
        print(f"3. Try with CPU: set CUDA_VISIBLE_DEVICES=''")
        return False


def show_code_search_use_cases():
    """Show practical use cases for code search adapter"""
    
    print(f"\n" + "=" * 80)
    print(f"📖 CODE SEARCH ADAPTER USE CASES")
    print(f"=" * 80)
    
    use_cases = {
        "Code Search Engines": {
            "description": "Search codebases with natural language queries",
            "example": "Query: 'function to validate email' → Find email validation functions"
        },
        
        "Documentation Assistance": {
            "description": "Match questions with relevant documentation",
            "example": "How to sort arrays? → Documentation about sorting methods"
        },
        
        "Code Completion": {
            "description": "Suggest code based on natural language descriptions",
            "example": "Description: 'connect to database' → Database connection code"
        },
        
        "Cross-Language Translation": {
            "description": "Find equivalent code in different programming languages",
            "example": "Python for loop → JavaScript for loop equivalent"
        },
        
        "API Discovery": {
            "description": "Find relevant APIs and libraries",
            "example": "Query: 'HTTP request library' → requests, axios, fetch APIs"
        },
        
        "Code Review Assistant": {
            "description": "Find similar code patterns for consistency",
            "example": "Check if error handling follows project patterns"
        },
        
        "Learning Resources": {
            "description": "Match programming concepts with code examples",
            "example": "Concept: 'recursion' → Recursive function examples"
        },
        
        "Stack Overflow Search": {
            "description": "Enhanced search for programming Q&A",
            "example": "Problem description → Relevant answered questions"
        }
    }
    
    for use_case, details in use_cases.items():
        print(f"\n🎯 {use_case}:")
        print(f"   Description: {details['description']}")
        print(f"   Example: {details['example']}")


if __name__ == "__main__":
    success = code_search_demo()
    
    if success:
        show_code_search_use_cases()
        
    print(f"\n📚 More Examples:")
    print(f"   vllm_examples/retrieval_example.py")
    print(f"   vllm_examples/text_matching_example.py")