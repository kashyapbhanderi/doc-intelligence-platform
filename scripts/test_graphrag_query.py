"""
scripts/test_graphrag_query.py
================================
Quick sanity-check script for GraphRAG after building the graph.
Runs 5 test queries and prints what each retrieval method found.

Run:
    python scripts/test_graphrag_query.py

WHERE TO PLACE THIS FILE: scripts/test_graphrag_query.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    # ── Load graph ─────────────────────────────────
    from knowledge_graph.graph_builder import GraphBuilder
    from knowledge_graph.hybrid_graphrag import HybridGraphRAG

    gb = GraphBuilder()
    if not gb.load():
        print("ERROR: No graph found. Run first: python scripts/build_graph.py")
        sys.exit(1)

    stats = gb.get_stats()
    print(f"Graph loaded: {stats['entity_nodes']} entities, {stats['chunk_nodes']} chunks\n")

    # ── Load your existing hybrid_search ───────────
    try:
        from embeddings.search import hybrid_search
    except ImportError:
        print("Could not import embeddings.search.hybrid_search")
        print("Make sure you run from the project root directory.\n")
        # Use a stub so the script can still test the graph part
        def hybrid_search(query: str, top_k: int = 5):
            return []

    graphrag = HybridGraphRAG(gb, hybrid_search, graph_weight=0.4)

    # ── Test queries ────────────────────────────────
    test_queries = [
        "What is the quarterly revenue?",
        "Who is the CEO and what did they announce?",
        "What are the operating margins?",
        "Compare performance across different business segments",
        "What risks are mentioned in the document?",
    ]

    print("=" * 70)
    print("GraphRAG Retrieval Test")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 60)

        explanation = graphrag.explain_retrieval(query, top_k=3)

        print(f"  Graph entities matched : {explanation['graph_entities']}")
        print(f"  Graph contributed      : {explanation['graph_contributed']}")

        if explanation["fused_results"]:
            print(f"  Top result (fused)     : {explanation['fused_results'][0]['text'][:120]}...")
            print(f"  RRF score              : {explanation['fused_results'][0].get('rrf_score', 'N/A')}")

        if not explanation["graph_entities"]:
            print("  ⚠ No graph entities matched — vector-only result")

    print("\n" + "=" * 70)
    print("Test complete. If all 5 queries returned results, GraphRAG is working.")
    print("\nTip: If entities are empty, check that spaCy loaded correctly.")
    print("     Run: python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    main()
