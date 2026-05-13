import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.chunker import chunk_pages


def test_chunk_pages_returns_list():
    """Chunks output should be a list."""
    pages = [{"page": 1, "text": "Hello world. " * 100, "source": "test.pdf"}]
    chunks = chunk_pages(pages)
    assert isinstance(chunks, list)


def test_chunk_has_required_fields():
    """Every chunk must have these fields."""
    pages = [{"page": 1, "text": "Hello world. " * 100, "source": "test.pdf"}]
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "source" in chunk
        assert "page" in chunk


def test_chunk_size_respected():
    """No chunk should exceed 600 characters (512 + small buffer)."""
    pages = [{"page": 1, "text": "word " * 1000, "source": "test.pdf"}]
    chunks = chunk_pages(pages, chunk_size=512)
    for chunk in chunks:
        assert chunk["char_count"] <= 600


def test_empty_pages_skipped():
    """Empty pages should not produce chunks."""
    pages = [{"page": 1, "text": "   ", "source": "test.pdf"}]
    chunks = chunk_pages(pages)
    assert len(chunks) == 0


def test_multiple_pages():
    """Multiple pages should all be chunked."""
    pages = [
        {"page": 1, "text": "First page content. " * 50, "source": "test.pdf"},
        {"page": 2, "text": "Second page content. " * 50, "source": "test.pdf"},
    ]
    chunks = chunk_pages(pages)
    assert len(chunks) > 2