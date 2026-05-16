import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

from embeddings.embedder import DocumentEmbedder


def run_search_comparison(output_path="eval/search_results.json"):
    """
    Run all 3 search methods on test queries.
    Save results to JSON for NDCG evaluation tomorrow.
    """
    embedder = DocumentEmbedder()
    os.makedirs("eval", exist_ok=True)

    queries = [
        "What is retrieval augmented generation?",
        "How does LoRA fine-tuning work?",
        "Explain transformer attention mechanisms",
        "What is BERT pre-training?",
        "How to reduce hallucination in LLMs?",
        "What is reinforcement learning from human feedback?",
        "How do sentence embeddings work?",
        "What is the tree of thought prompting?",
        "Explain dense passage retrieval",
        "What is instruction tuning for LLMs?",
    ]

    all_results = []

    for query in queries:
        result = {
            "query": query,
            "bm25": embedder.search_bm25(query, top_k=10),
            "vector": embedder.search_vector(query, top_k=10),
            "hybrid": embedder.search_hybrid(query, top_k=10)
        }
        all_results.append(result)
        print(f"Searched: {query[:50]}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} search results to "
          f"{output_path}")
    return all_results


if __name__ == "__main__":
    run_search_comparison()