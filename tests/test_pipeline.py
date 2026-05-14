import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.chunker import chunk_pages, save_chunks
from ingestion.pdf_extractor import is_scanned_pdf


def test_is_scanned_pdf_detects_empty():
    """Pages with very little text should be detected as scanned."""
    pages = [{"page": 1, "text": "hi", "source": "test.pdf"}]
    assert is_scanned_pdf(pages) == True


def test_is_scanned_pdf_detects_normal():
    """Pages with good text should NOT be detected as scanned."""
    pages = [{"page": 1, "text": "word " * 100, "source": "test.pdf"}]
    assert is_scanned_pdf(pages) == False


def test_save_chunks_creates_file(tmp_path):
    """save_chunks should create a JSON file."""
    chunks = [{"chunk_id": 0, "text": "hello", "source": "test.pdf",
               "page": 1, "char_count": 5}]
    output = str(tmp_path / "test_chunks.json")
    save_chunks(chunks, output)
    assert os.path.exists(output)


def test_save_chunks_valid_json(tmp_path):
    """Saved file should be valid JSON."""
    chunks = [{"chunk_id": 0, "text": "hello world", "source": "test.pdf",
               "page": 1, "char_count": 11}]
    output = str(tmp_path / "test_chunks.json")
    save_chunks(chunks, output)
    with open(output) as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["text"] == "hello world"


def test_chunk_pages_preserves_source():
    """Source filename must be preserved in every chunk."""
    pages = [{"page": 1, "text": "content " * 200, "source": "myfile.pdf"}]
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert chunk["source"] == "myfile.pdf"


def test_chunk_ids_are_unique():
    """Every chunk must have a unique chunk_id."""
    pages = [
        {"page": 1, "text": "content " * 100, "source": "test.pdf"},
        {"page": 2, "text": "more content " * 100, "source": "test.pdf"},
    ]
    chunks = chunk_pages(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))  # all unique