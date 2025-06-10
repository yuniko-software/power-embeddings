import json
import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path
import os

def convert_tokenizer_outputs(tokens, token_indices):
    """Convert tokenizer outputs to model input format"""
    # Pair tokens with their indices and sort by position
    token_pairs = list(zip(token_indices, tokens))
    token_pairs.sort()  # Sort by position (token_indices)
    
    # Get ordered tokens
    ordered_tokens = [pair[1] for pair in token_pairs]
    
    # Create input_ids and attention_mask
    input_ids = np.array([ordered_tokens], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    
    return input_ids, attention_mask

class OnnxE5LargeInstructEmbedder:
    """E5 Large Instruct embedder using ONNX tokenizer and model"""
    
    def __init__(self, tokenizer_path, model_path):
        """Initialize the embedder with ONNX tokenizer and model"""
        # Initialize tokenizer session
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(get_library_path())
        self.tokenizer_session = ort.InferenceSession(
            tokenizer_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Initialize model session
        self.model_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
    
    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        """Format query with instruction as required by E5"""
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def encode(self, texts):
        """Generate embeddings for the input texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            # Tokenize the input
            tokenizer_outputs = self.tokenizer_session.run(None, {"inputs": np.array([text])})
            tokens, _, token_indices = tokenizer_outputs
            
            # Convert to model input format
            input_ids, attention_mask = convert_tokenizer_outputs(tokens, token_indices)
            
            # Generate embeddings
            model_outputs = self.model_session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            
            # Extract normalized embeddings
            embedding = model_outputs[0][0]  # Remove batch dimension
            embeddings.append(embedding.tolist())  # Convert to list for JSON serialization
        
        return embeddings if len(embeddings) > 1 else embeddings[0]

def main():
    """Generate reference embeddings using E5 Large Instruct ONNX models"""
    
    script_dir = os.getcwd()
    onnx_dir = os.path.join(script_dir, "onnx")
    
    tokenizer_path = os.path.join(onnx_dir, "e5_large_instruct_tokenizer.onnx")
    model_path = os.path.join(onnx_dir, "e5_large_instruct_model.onnx")
    output_path = os.path.join(onnx_dir, "e5_large_instruct_reference_embeddings.json")
    
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Using model: {model_path}")
    print(f"Output will be saved to: {output_path}")

    # Verify files exist
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer file not found at {tokenizer_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    # Initialize the E5 Large Instruct embedder
    print("Initializing E5 Large Instruct ONNX embedder...")
    embedder = OnnxE5LargeInstructEmbedder(tokenizer_path, model_path)
    
    # Test data matching the original example
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    test_cases = {
        # Instructed queries (for retrieval)
        "instruct_protein_query": embedder.get_detailed_instruct(task, 'how much protein should a female eat'),
        "instruct_pumpkin_query": embedder.get_detailed_instruct(task, '南瓜的家常做法'),
        
        # Documents (no instruction needed)
        "protein_document": "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "pumpkin_document": "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅",
        
        # Additional test cases
        "simple_text": "This is a simple test text.",
        "empty_text": "",
        "multilingual_text": "English, Español, Русский, 中文, العربية, हिन्दी",
        "long_text": "This is a longer text that should generate a meaningful embedding vector. The embedding model should capture the semantic meaning of this text and provide high-quality representations for various downstream tasks.",
        "technical_text": "ONNX Runtime is a performance-focused engine for ONNX models, enabling cross-platform inference with identical results.",
        
        # Different task instructions
        "classification_query": embedder.get_detailed_instruct("Classify the sentiment of this text", "I love this product!"),
        "summarization_query": embedder.get_detailed_instruct("Summarize the following passage", "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."),
    }
    
    embeddings = {}
    
    print(f"\nGenerating embeddings for {len(test_cases)} test cases...")
    for name, text in test_cases.items():
        print(f"Processing: {name}")
        embedding = embedder.encode(text)
        embeddings[name] = {
            "text": text,
            "embedding": embedding
        }
    
    # Save to JSON file
    print(f"\nSaving reference embeddings to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    
    # Print some statistics
    first_embedding = list(embeddings.values())[0]["embedding"]
    print(f"\nSaved {len(embeddings)} reference embeddings")
    print(f"Embedding dimension: {len(first_embedding)}")
    print(f"Sample embedding (first 5 values): {first_embedding[:5]}")
    print("\nReference embeddings generated successfully!")

if __name__ == "__main__":
    main()