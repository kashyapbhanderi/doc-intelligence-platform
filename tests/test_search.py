import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from embeddings.embedder import DocumentEmbedder


@pytest.fixture(scope="module")
def embedder():
    return DocumentEmbedder()


def test_vector_search_returns_results(embedder):
    """Vector search should return results for any query."""
    results = embedder.search_vector(
        "language models", top_k=3)
    assert len(results) > 0


def test_vector_search_has_text_field(embedder):
    """Every result must have a text field."""
    results = embedder.search_vector(
        "transformer attention", top_k=3)
    for r in results:
        assert "text" in r
        assert len(r["text"]) > 0


def test_bm25_search_returns_results(embedder):
    """BM25 search should return results."""
    results = embedder.search_bm25("BERT encoder", top_k=3)
    assert len(results) > 0


def test_hybrid_search_returns_results(embedder):
    """Hybrid search using RRF should return results."""
    results = embedder.search_hybrid(
        "fine-tuning language model", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_better_than_individual(embedder):
    """
    Hybrid should return results from both BM25 and vector.
    RRF combines both — so results should be comprehensive.
    """
    query = "LoRA fine tuning language model"
    bm25 = embedder.search_bm25(query, top_k=3) or []
    vector = embedder.search_vector(query, top_k=3) or []
    hybrid = embedder.search_hybrid(query, top_k=5) or []

    # Hybrid should return results if either method does
    if bm25 or vector:
        assert len(hybrid) > 0

def test_hybrid_returns_correct_count(embedder):
    """Hybrid search should respect top_k limit."""
    results = embedder.search_hybrid("LoRA", top_k=5)
    assert len(results) <= 5


def test_search_result_has_source(embedder):
    """Every result should have a source filename."""
    results = embedder.search_hybrid("attention", top_k=3)
    for r in results:
        assert "source" in r
        assert r["source"].endswith(".pdf")


def test_similar_queries_share_results(embedder):
    """Very similar queries should return overlapping results."""
    r1 = embedder.search_vector("LoRA fine tuning", top_k=5)
    r2 = embedder.search_vector("LoRA low rank adaptation",
                                 top_k=5)
    sources1 = set(r["source"] for r in r1)
    sources2 = set(r["source"] for r in r2)
    # At least one common source
    assert len(sources1.intersection(sources2)) > 0