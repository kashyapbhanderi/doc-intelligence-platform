import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test embedding
sentences = [
    "Retrieval augmented generation improves LLM accuracy",
    "Large language models can hallucinate facts",
    "Vector databases store embeddings for semantic search"
]

print("Embedding 3 test sentences...")
vectors = model.encode(sentences)

print(f"\nResults:")
print(f"  Input sentences: {len(sentences)}")
print(f"  Vector shape:    {vectors.shape}")
print(f"  Vector size:     {vectors.shape[1]} dimensions")
print(f"  First 5 values:  {vectors[0][:5].tolist()}")
print("\nEmbedding model working correctly!")
