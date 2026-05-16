import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from embeddings.embedder import DocumentEmbedder

embedder = DocumentEmbedder()
print(f"Chunks in Weaviate: {embedder.get_document_count()}")
print("=" * 60)
print("COMPARING ALL 3 SEARCH METHODS")
print("=" * 60)

queries = [
    ("LoRA low rank adaptation fine-tuning",
     "lora.pdf"),
    ("how does BERT language model work",
     "bert.pdf"),
    ("retrieval augmented generation reduce hallucination",
     "rag_original.pdf"),
    ("making LLMs follow human instructions",
     "flan.pdf"),
    ("self attention transformer neural network",
     "attention.pdf"),
    ("reinforcement learning from human feedback",
     "scaling_rlhf.pdf"),
    ("sentence embeddings semantic similarity",
     "sentence_bert.pdf"),
    ("tree of thought reasoning LLM",
     "tree_of_thought.pdf"),
]

bm25_score = 0
vector_score = 0
hybrid_score = 0

results_table = []

for query, expected in queries:
    bm25_res = embedder.search_bm25(query, top_k=5) or []
    vector_res = embedder.search_vector(query, top_k=5) or []
    hybrid_res = embedder.search_hybrid(query, top_k=5,
                                        alpha=0.5) or []

    bm25_sources = [r.get("source", "") for r in bm25_res]
    vector_sources = [r.get("source", "") for r in vector_res]
    hybrid_sources = [r.get("source", "") for r in hybrid_res]

    b = "✅" if expected in bm25_sources else "❌"
    v = "✅" if expected in vector_sources else "❌"
    h = "✅" if expected in hybrid_sources else "❌"

    if expected in bm25_sources:
        bm25_score += 1
    if expected in vector_sources:
        vector_score += 1
    if expected in hybrid_sources:
        hybrid_score += 1

    results_table.append((query[:40], b, v, h))

# Print results table
print(f"\n{'Query':<42} {'BM25'} {'Vec'} {'Hybrid'}")
print("-" * 58)
for row in results_table:
    print(f"{row[0]:<42} {row[1]}    {row[2]}    {row[3]}")

total = len(queries)
print("\n" + "=" * 58)
print(f"Method     Correct   Score")
print(f"BM25       {bm25_score}/{total}       "
      f"{bm25_score/total*100:.0f}%")
print(f"Vector     {vector_score}/{total}       "
      f"{vector_score/total*100:.0f}%")
print(f"Hybrid     {hybrid_score}/{total}       "
      f"{hybrid_score/total*100:.0f}%")
print("=" * 58)

best = max(
    [("BM25", bm25_score),
     ("Vector", vector_score),
     ("Hybrid", hybrid_score)],
    key=lambda x: x[1]
)
print(f"\nWinner: {best[0]} with {best[1]}/{total} correct")
print("\nSave these numbers for your README!")