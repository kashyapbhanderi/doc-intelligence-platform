import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from embeddings.embedder import DocumentEmbedder

embedder = DocumentEmbedder()
total = embedder.get_document_count()
print(f"Total chunks in Weaviate: {total}")
print("=" * 60)

# Test queries — mix of keyword and semantic
test_cases = [
    {
        "query": "LoRA low rank adaptation",
        "expected_source": "lora.pdf",
        "type": "keyword"
    },
    {
        "query": "BERT bidirectional encoder",
        "expected_source": "bert.pdf",
        "type": "keyword"
    },
    {
        "query": "how to make language models follow instructions",
        "expected_source": "flan.pdf",
        "type": "semantic"
    },
    {
        "query": "reducing hallucination in generated text",
        "expected_source": "rag_original.pdf",
        "type": "semantic"
    },
    {
        "query": "attention mechanism transformer architecture",
        "expected_source": "attention.pdf",
        "type": "keyword"
    },
]

bm25_correct = 0
vector_correct = 0

print(f"\n{'Query':<45} {'BM25':<20} {'Vector':<20}")
print("-" * 85)

for tc in test_cases:
    query = tc["query"]
    expected = tc["expected_source"]

    # BM25 search
    bm25_results = embedder.search_bm25(query, top_k=3)
    bm25_sources = [r.get("source", "") for r in bm25_results]
    bm25_hit = "✅" if expected in bm25_sources else "❌"
    bm25_top = bm25_sources[0] if bm25_sources else "none"

    # Vector search
    vector_results = embedder.search_vector(query, top_k=3)
    vector_sources = [r.get("source", "") for r in vector_results]
    vector_hit = "✅" if expected in vector_sources else "❌"
    vector_top = vector_sources[0] if vector_sources else "none"

    if expected in bm25_sources:
        bm25_correct += 1
    if expected in vector_sources:
        vector_correct += 1

    print(f"{query[:43]:<45} "
          f"{bm25_hit} {bm25_top[:16]:<18} "
          f"{vector_hit} {vector_top[:16]:<18}")

print("\n" + "=" * 60)
print(f"BM25 correct:   {bm25_correct}/{len(test_cases)}")
print(f"Vector correct: {vector_correct}/{len(test_cases)}")
print("\nConclusion:")
if bm25_correct > vector_correct:
    print("BM25 wins on these queries (keyword-heavy)")
elif vector_correct > bm25_correct:
    print("Vector wins on these queries (semantic-heavy)")
else:
    print("Both methods perform equally — hybrid will be best")