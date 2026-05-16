import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from embeddings.embedder import DocumentEmbedder

embedder = DocumentEmbedder()
count = embedder.get_document_count()
print(f"Total chunks in Weaviate: {count}")
print("=" * 50)

test_queries = [
    "What is retrieval augmented generation?",
    "How does LoRA fine-tuning work?",
    "What are transformer attention mechanisms?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 40)

    # Vector search
    results = embedder.search_vector(query, top_k=2)
    if results:
        print(f"Vector search top result:")
        print(f"  Source: {results[0]['source']}")
        print(f"  Text: {results[0]['text'][:120]}...")
    else:
        print("  No results found")