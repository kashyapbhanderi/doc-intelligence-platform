import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from embeddings.query_engine import query_with_sources, build_query_engine
from eval.answer_eval import keyword_overlap_score, source_accuracy_score


# ── Unit tests — no API or Weaviate needed ───────────────────────────────────

def test_keyword_overlap_perfect_match():
    score = keyword_overlap_score(
        "LoRA reduces trainable parameters",
        "LoRA reduces trainable parameters"
    )
    assert score == 1.0


def test_keyword_overlap_no_match():
    score = keyword_overlap_score(
        "quantum physics equations",
        "LoRA reduces trainable parameters"
    )
    assert score < 0.3


def test_keyword_overlap_partial_match():
    score = keyword_overlap_score(
        "LoRA is a method that reduces parameters",
        "LoRA reduces parameters in fine-tuning"
    )
    assert 0.0 < score < 1.0


def test_source_accuracy_correct():
    sources = [{"source": "lora.pdf"}, {"source": "bert.pdf"}, {"source": "attention.pdf"}]
    assert source_accuracy_score(sources, "lora.pdf") == 1.0


def test_source_accuracy_wrong():
    sources = [{"source": "bert.pdf"}, {"source": "attention.pdf"}]
    assert source_accuracy_score(sources, "lora.pdf") == 0.0


def test_source_accuracy_empty():
    assert source_accuracy_score([], "lora.pdf") == 0.0


# ── Integration tests — needs Weaviate running ────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    try:
        eng, client = build_query_engine(top_k=3)
        return eng
    except Exception:
        pytest.skip("Weaviate not available")


def test_query_returns_answer(engine):
    result = query_with_sources(engine, "What is a transformer?")
    assert "answer" in result
    assert len(result["answer"]) > 0


def test_query_returns_sources(engine):
    result = query_with_sources(engine, "How does attention work?")
    assert "sources" in result
    assert "num_sources" in result


def test_query_answer_is_string(engine):
    result = query_with_sources(engine, "What is BERT?")
    assert isinstance(result["answer"], str)


def test_query_sources_have_filename(engine):
    result = query_with_sources(engine, "What is fine-tuning?")
    for source in result["sources"]:
        assert "source" in source