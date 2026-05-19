import os
import sys
import time
sys.path.insert(0, os.path.abspath('.'))

from embeddings.query_engine import (
    build_query_engine,
    query_with_sources
)

print("Testing updated query engine...")
print("=" * 60)

# Build engine — will auto-detect fine-tuned model
engine, client = build_query_engine(top_k=5)

test_questions = [
    "What is retrieval augmented generation?",
    "How does LoRA reduce memory during fine-tuning?",
    "Explain the transformer attention mechanism",
]

for question in test_questions:
    print(f"\nQ: {question}")
    start = time.time()
    result = query_with_sources(engine, question)
    elapsed = time.time() - start

    print(f"A: {result['answer'][:200]}...")
    print(f"Sources: {result['num_sources']} | "
          f"Time: {elapsed:.1f}s")
    for src in result['sources'][:2]:
        print(f"  → {src['source']} "
              f"(page {src['page']})")

print("\n" + "=" * 60)
print("Query engine working with fine-tuned model!")