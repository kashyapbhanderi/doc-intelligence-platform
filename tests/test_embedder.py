import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from embeddings.embedder import DocumentEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Create one embedder for all tests in this file."""
    return DocumentEmbedder()


def test_embed_text_returns_list(embedder):
    """embed_text should return a list of floats."""
    vector = embedder.embed_text("Hello world")
    assert isinstance(vector, list)
    assert len(vector) == 384  # all-MiniLM-L6-v2 size


def test_embed_text_correct_dimensions(embedder):
    """Vector must be exactly 384 dimensions."""
    vector = embedder.embed_text("Test sentence")
    assert len(vector) == 384


def test_embed_batch_multiple_texts(embedder):
    """Batch embedding should handle multiple texts."""
    texts = ["First sentence", "Second sentence",
             "Third sentence"]
    vectors = embedder.embed_batch(texts)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)


def test_embed_similar_texts_close_vectors(embedder):
    """Similar texts should have similar vectors."""
    v1 = embedder.embed_text("The cat sat on the mat")
    v2 = embedder.embed_text("A cat is sitting on a mat")
    v3 = embedder.embed_text("Quantum physics equations")

    # Calculate cosine similarity manually
    import numpy as np
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    v3_arr = np.array(v3)

    sim_12 = np.dot(v1_arr, v2_arr) / (
        np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr))
    sim_13 = np.dot(v1_arr, v3_arr) / (
        np.linalg.norm(v1_arr) * np.linalg.norm(v3_arr))

    # Similar sentences should score higher
    assert sim_12 > sim_13


def test_weaviate_connection(embedder):
    """Weaviate should be reachable."""
    count = embedder.get_document_count()
    assert isinstance(count, int)
    assert count >= 0


def test_schema_exists(embedder):
    """Document schema should exist in Weaviate."""
    schema = embedder.client.collections.list_all()
    classes =  list(schema.keys())
    assert "Document" in classes