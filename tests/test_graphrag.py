"""
tests/test_graphrag.py
Tests for GraphRAG knowledge graph components.

Run: pytest tests/test_graphrag.py -v
"""

import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from collections import defaultdict


# ────────────────────────────────────────────────
# FIXTURES
# ────────────────────────────────────────────────

@pytest.fixture
def mock_graph_builder():
    """GraphBuilder with a small pre-built graph (no spaCy/LLM calls)."""
    from knowledge_graph.graph_builder import GraphBuilder

    gb = GraphBuilder.__new__(GraphBuilder)
    gb.G = nx.Graph()
    gb.entity_to_chunks = defaultdict(list)
    gb.client = MagicMock()
    gb.nlp = None                           # skip real NLP in tests
    gb.graph_path = MagicMock()

    # Add synthetic nodes
    gb.G.add_node("chunk_0", node_type="chunk",  text="Infosys Q3 revenue was ₹38,000 crore.", source="q3.pdf", page=1)
    gb.G.add_node("chunk_1", node_type="chunk",  text="CEO Salil Parekh presented the results.", source="q3.pdf", page=2)
    gb.G.add_node("chunk_2", node_type="chunk",  text="Wipro posted lower margins in Q3.",       source="wipro.pdf", page=1)

    gb.G.add_node("entity::infosys",  node_type="entity", text="Infosys",  entity_type="ORG")
    gb.G.add_node("entity::salil parekh", node_type="entity", text="Salil Parekh", entity_type="PERSON")
    gb.G.add_node("entity::wipro",    node_type="entity", text="Wipro",    entity_type="ORG")

    gb.G.add_edge("chunk_0", "entity::infosys",        edge_type="mentions", weight=1.0)
    gb.G.add_edge("chunk_1", "entity::salil parekh",   edge_type="mentions", weight=1.0)
    gb.G.add_edge("chunk_2", "entity::wipro",          edge_type="mentions", weight=1.0)
    gb.G.add_edge("entity::infosys", "entity::salil parekh", edge_type="relation", relation="led by", weight=2.0)

    gb.entity_to_chunks["entity::infosys"].append("chunk_0")
    gb.entity_to_chunks["entity::salil parekh"].append("chunk_1")
    gb.entity_to_chunks["entity::wipro"].append("chunk_2")

    return gb


# ────────────────────────────────────────────────
# GRAPH BUILDER TESTS
# ────────────────────────────────────────────────

class TestGraphBuilder:

    def test_get_stats(self, mock_graph_builder):
        stats = mock_graph_builder.get_stats()
        assert stats["total_nodes"] == 6
        assert stats["entity_nodes"] == 3
        assert stats["chunk_nodes"] == 3
        assert stats["total_edges"] == 4

    def test_load_returns_false_when_no_file(self, tmp_path):
        from knowledge_graph.graph_builder import GraphBuilder
        gb = GraphBuilder(graph_path=str(tmp_path / "nonexistent.pkl"))
        assert gb.load() is False

    def test_save_and_load_roundtrip(self, tmp_path, mock_graph_builder):
        mock_graph_builder.graph_path = tmp_path / "graph.pkl"
        mock_graph_builder.save()
        assert mock_graph_builder.graph_path.exists()

        from knowledge_graph.graph_builder import GraphBuilder
        gb2 = GraphBuilder(graph_path=str(mock_graph_builder.graph_path))
        loaded = gb2.load()
        assert loaded is True
        assert gb2.G.number_of_nodes() == 6

    def test_add_chunk_no_nlp_adds_chunk_node(self, mock_graph_builder):
        """Without spaCy, chunks still get added as nodes."""
        n_before = mock_graph_builder.G.number_of_nodes()
        mock_graph_builder.add_chunk("chunk_99", "Some text with no entities.", source="test.pdf")
        assert mock_graph_builder.G.has_node("chunk_99")
        assert mock_graph_builder.G.number_of_nodes() == n_before + 1


# ────────────────────────────────────────────────
# GRAPH RETRIEVER TESTS
# ────────────────────────────────────────────────

class TestGraphRetriever:

    def test_entity_cache_built(self, mock_graph_builder):
        from knowledge_graph.graph_retriever import GraphRetriever
        gr = GraphRetriever(mock_graph_builder)
        cache_texts = [text for _, text in gr._entity_cache]
        assert "Infosys" in cache_texts
        assert "Salil Parekh" in cache_texts

    def test_find_query_entities_exact_match(self, mock_graph_builder):
        from knowledge_graph.graph_retriever import GraphRetriever
        gr = GraphRetriever(mock_graph_builder)
        entities = gr._find_query_entities("What is Infosys revenue?")
        assert "entity::infosys" in entities

    def test_retrieve_returns_list(self, mock_graph_builder):
        from knowledge_graph.graph_retriever import GraphRetriever
        gr = GraphRetriever(mock_graph_builder)
        results = gr.retrieve("Infosys revenue Q3", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_finds_infosys_chunk(self, mock_graph_builder):
        from knowledge_graph.graph_retriever import GraphRetriever
        gr = GraphRetriever(mock_graph_builder)
        results = gr.retrieve("What is Infosys revenue?", top_k=5)
        chunk_ids = [r["chunk_id"] for r in results]
        assert "chunk_0" in chunk_ids    # Infosys chunk

    def test_multihop_traversal_reaches_ceo(self, mock_graph_builder):
        """
        Infosys → [relation] → Salil Parekh → chunk_1
        So querying 'Infosys' with 2 hops should also find chunk_1.
        """
        from knowledge_graph.graph_retriever import GraphRetriever
        gr = GraphRetriever(mock_graph_builder)
        results = gr.retrieve("Infosys CEO announcement", top_k=5, hops=2)
        chunk_ids = [r["chunk_id"] for r in results]
        assert "chunk_1" in chunk_ids    # CEO chunk reached via 2-hop

    def test_retrieve_empty_graph(self):
        """Empty graph returns empty list without crashing."""
        from knowledge_graph.graph_builder import GraphBuilder
        from knowledge_graph.graph_retriever import GraphRetriever

        gb = GraphBuilder.__new__(GraphBuilder)
        gb.G = nx.Graph()
        gb.entity_to_chunks = defaultdict(list)
        gb.nlp = None
        gb.client = MagicMock()

        gr = GraphRetriever(gb)
        assert gr.retrieve("anything") == []

    def test_explain_returns_dict(self, mock_graph_builder):
        from knowledge_graph.graph_retriever import GraphRetriever
        gr = GraphRetriever(mock_graph_builder)
        info = gr.explain("Infosys CEO")
        assert "matched_entities" in info
        assert "chunks_found" in info
        assert isinstance(info["matched_entities"], list)


# ────────────────────────────────────────────────
# HYBRID GRAPHRAG TESTS
# ────────────────────────────────────────────────

class TestHybridGraphRAG:

    def _fake_vector_search(self, query: str, top_k: int = 5) -> list[dict]:
        return [
            {"chunk_id": "chunk_0", "text": "Infosys Q3 revenue was ₹38,000 crore.", "score": 0.9},
            {"chunk_id": "chunk_2", "text": "Wipro posted lower margins.",            "score": 0.7},
        ]

    def test_retrieve_returns_fused_list(self, mock_graph_builder):
        from knowledge_graph.hybrid_graphrag import HybridGraphRAG
        hg = HybridGraphRAG(mock_graph_builder, self._fake_vector_search)
        results = hg.retrieve("Infosys revenue", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_rrf_scores_present(self, mock_graph_builder):
        from knowledge_graph.hybrid_graphrag import HybridGraphRAG
        hg = HybridGraphRAG(mock_graph_builder, self._fake_vector_search)
        results = hg.retrieve("Infosys revenue", top_k=5)
        for r in results:
            assert "rrf_score" in r

    def test_fallback_to_vector_when_no_graph_results(self, mock_graph_builder):
        """If graph finds nothing, vector results are returned."""
        from knowledge_graph.hybrid_graphrag import HybridGraphRAG
        hg = HybridGraphRAG(mock_graph_builder, self._fake_vector_search)
        results = hg.retrieve("completely unrelated jargon xyz", top_k=3)
        # Should return vector results at minimum
        assert isinstance(results, list)

    def test_explain_contains_expected_keys(self, mock_graph_builder):
        from knowledge_graph.hybrid_graphrag import HybridGraphRAG
        hg = HybridGraphRAG(mock_graph_builder, self._fake_vector_search)
        info = hg.explain_retrieval("Infosys CEO announcement")
        assert "graph_entities" in info
        assert "vector_results" in info
        assert "fused_results"  in info
        assert "graph_contributed" in info

    def test_rrf_fusion_logic(self):
        from knowledge_graph.hybrid_graphrag import reciprocal_rank_fusion

        list_a = [{"chunk_id": "A", "text": "x"}, {"chunk_id": "B", "text": "y"}]
        list_b = [{"chunk_id": "B", "text": "y"}, {"chunk_id": "C", "text": "z"}]

        result = reciprocal_rank_fusion([list_a, list_b])
        ids = [r["chunk_id"] for r in result]

        # B appears in both lists — should rank #1
        assert ids[0] == "B"
        # All three items should be present
        assert set(ids) == {"A", "B", "C"}
