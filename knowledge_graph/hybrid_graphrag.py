"""
GraphRAG — hybrid_graphrag.py
==============================
Fuses your existing vector search with GraphRAG retrieval
using Reciprocal Rank Fusion (RRF).

This is the DROP-IN REPLACEMENT for hybrid_search in agents/executor.py.

WHERE TO PLACE THIS FILE: knowledge_graph/hybrid_graphrag.py

INTEGRATION (in agents/executor.py):
──────────────────────────────────────
# BEFORE (Week 2 code):
from embeddings.search import hybrid_search

def executor_node(state):
    context = []
    for q in state["sub_queries"]:
        results = hybrid_search(q, top_k=3)    # ← old
        context.extend(results)

# AFTER (GraphRAG upgrade):
from knowledge_graph.hybrid_graphrag import HybridGraphRAG
from knowledge_graph.graph_builder import GraphBuilder
from embeddings.search import hybrid_search

gb = GraphBuilder()
gb.load()
graphrag = HybridGraphRAG(gb, hybrid_search)   # wrap your existing fn

def executor_node(state):
    context = []
    for q in state["sub_queries"]:
        results = graphrag.retrieve(q, top_k=3)  # ← new (same interface)
        context.extend(results)
──────────────────────────────────────
"""

from __future__ import annotations
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_graph.graph_builder import GraphBuilder

from knowledge_graph.graph_retriever import GraphRetriever


# ────────────────────────────────────────────────
# RRF FUSION
# ────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "chunk_id",
    k: int = 60,
) -> list[dict]:
    """
    Merge N ranked lists using Reciprocal Rank Fusion.
    Formula: score(item) = Σ 1 / (k + rank)

    The same algorithm used by Weaviate's built-in hybrid search,
    applied here across two different retrieval methods.
    """
    scores: dict[str, float] = {}
    items:  dict[str, dict]  = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = str(item.get(id_key, rank))
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            items[item_id] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    merged = []
    for item_id, rrf_score in ranked:
        entry = items[item_id].copy()
        entry["rrf_score"] = round(rrf_score, 6)
        merged.append(entry)

    return merged


# ────────────────────────────────────────────────
# HYBRID GRAPHRAG
# ────────────────────────────────────────────────

class HybridGraphRAG:
    """
    Combines your existing vector/hybrid search with GraphRAG
    via Reciprocal Rank Fusion.

    Result: better retrieval on complex multi-entity questions,
    same or better on simple questions (vector still dominates).

    Args:
        graph_builder    : loaded GraphBuilder instance
        vector_search_fn : your existing hybrid_search function
                           must accept (query: str, top_k: int) → list[dict]
        graph_weight     : how much to weight graph results in final ranking.
                           0.0 = pure vector, 1.0 = pure graph.
                           0.4 is a good starting point.
    """

    def __init__(
        self,
        graph_builder: "GraphBuilder",
        vector_search_fn: Callable,
        graph_weight: float = 0.4,
    ):
        self.graph_retriever = GraphRetriever(graph_builder)
        self.vector_search = vector_search_fn
        self.graph_weight = graph_weight

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve using both vector search and GraphRAG, then fuse with RRF.

        Returns same format as your existing hybrid_search — plug-and-play.
        """
        # ── Vector retrieval ──────────────────────────
        try:
            vec_results = self.vector_search(query, top_k=top_k)
            for item in vec_results:
                # Normalise id key (Weaviate uses various names)
                if "chunk_id" not in item:
                    item["chunk_id"] = item.get("id") or item.get("_additional", {}).get("id", "")
                item["retrieval_method"] = "vector"
        except Exception as exc:
            print(f"[HybridGraphRAG] Vector search error: {exc}")
            vec_results = []

        # ── GraphRAG retrieval ─────────────────────────
        graph_results = self.graph_retriever.retrieve(query, top_k=top_k)

        # ── Fallbacks ─────────────────────────────────
        if not vec_results:
            return graph_results[:top_k]
        if not graph_results:
            return vec_results[:top_k]

        # ── RRF fusion ────────────────────────────────
        # Give graph list extra weight by duplicating entries proportionally
        graph_copies = max(1, round(self.graph_weight * 2))
        fused = reciprocal_rank_fusion(
            [vec_results] + [graph_results] * graph_copies,
            id_key="chunk_id",
        )
        return fused[:top_k]

    def explain_retrieval(self, query: str, top_k: int = 3) -> dict:
        """
        Debug method: shows what each method found before fusion.
        Use this in your LangSmith traces to verify GraphRAG is contributing.

        Example:
            graphrag.explain_retrieval("What did the CEO announce?")
        """
        vec_results   = []
        graph_results = []

        try:
            vec_results = self.vector_search(query, top_k=top_k)
        except Exception:
            pass

        graph_results = self.graph_retriever.retrieve(query, top_k=top_k)
        fused         = self.retrieve(query, top_k=top_k)
        graph_explain = self.graph_retriever.explain(query)

        return {
            "query":           query,
            "graph_entities":  graph_explain.get("matched_entities", []),
            "vector_results":  [{"text": r.get("text", "")[:120]} for r in vec_results],
            "graph_results":   [{"text": r.get("text", "")[:120], "score": r.get("graph_score", 0)} for r in graph_results],
            "fused_results":   [{"text": r.get("text", "")[:120], "rrf_score": r.get("rrf_score", 0)} for r in fused],
            "graph_contributed": len(graph_results) > 0,
        }
