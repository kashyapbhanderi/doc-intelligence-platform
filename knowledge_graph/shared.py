# knowledge_graph/shared.py
"""
Single shared GraphBuilder + HybridGraphRAG instance.
Both api/main.py and agents/executor.py import from here
instead of each building their own.
"""
from knowledge_graph.graph_builder import GraphBuilder
from knowledge_graph.hybrid_graphrag import HybridGraphRAG
from embeddings.search import hybrid_search

_gb = None
_graphrag = None


def get_graph_builder() -> GraphBuilder:
    global _gb
    if _gb is None:
        _gb = GraphBuilder()
        _gb.load()
    return _gb


def get_hybrid_graphrag() -> HybridGraphRAG:
    global _graphrag
    if _graphrag is None:
        _graphrag = HybridGraphRAG(
            get_graph_builder(), hybrid_search, graph_weight=0.4
        )
    return _graphrag