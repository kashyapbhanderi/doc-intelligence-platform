import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from agents.executor import (
    deduplicate_chunks,
    build_context,
)
from agents.critic import parse_verdict


# ── Executor unit tests (no Weaviate needed) ─────────

def test_deduplicate_removes_exact_duplicates():
    """Same chunk text should appear only once."""
    chunks = [
        {"text": "A" * 150, "source": "a.pdf",
         "page": 1, "score": 0.9, "query": "q1"},
        {"text": "A" * 150, "source": "a.pdf",
         "page": 1, "score": 0.7, "query": "q2"},
        {"text": "B" * 150, "source": "b.pdf",
         "page": 2, "score": 0.8, "query": "q1"},
    ]
    result = deduplicate_chunks(chunks)
    assert len(result) == 2


def test_deduplicate_keeps_higher_score():
    """When duplicate, keep the one with higher score."""
    chunks = [
        {"text": "A" * 150, "source": "a.pdf",
         "page": 1, "score": 0.7, "query": "q1"},
        {"text": "A" * 150, "source": "a.pdf",
         "page": 1, "score": 0.9, "query": "q2"},
    ]
    result = deduplicate_chunks(chunks)
    assert result[0]["score"] == 0.9


def test_deduplicate_sorts_by_score():
    """Results should be sorted highest score first."""
    chunks = [
        {"text": "A" * 150, "source": "a.pdf",
         "page": 1, "score": 0.5, "query": "q"},
        {"text": "B" * 150, "source": "b.pdf",
         "page": 1, "score": 0.9, "query": "q"},
        {"text": "C" * 150, "source": "c.pdf",
         "page": 1, "score": 0.7, "query": "q"},
    ]
    result = deduplicate_chunks(chunks)
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_build_context_format():
    """Context should number each chunk with source."""
    chunks = [
        {"text": "Chunk one text here.",
         "source": "paper.pdf", "page": 3,
         "score": 0.9, "query": "q"},
    ]
    ctx = build_context(chunks)
    assert "[1]" in ctx
    assert "paper.pdf" in ctx
    assert "Chunk one text here." in ctx


def test_build_context_caps_at_max():
    """Context should not exceed max_chunks entries."""
    chunks = [
        {"text": f"Chunk {i} " * 20,
         "source": "p.pdf", "page": i,
         "score": 0.9, "query": "q"}
        for i in range(15)
    ]
    ctx = build_context(chunks, max_chunks=5)
    assert "[6]" not in ctx     # only 5 chunks included
    assert "[5]" in ctx


def test_build_context_empty_chunks():
    """Empty chunk list should return empty string."""
    ctx = build_context([])
    assert ctx == ""


# ── Critic unit tests ────────────────────────────────

def test_parse_verdict_faithful():
    """FAITHFUL verdict should return True."""
    content = "VERDICT: FAITHFUL\nREASON: All claims supported."
    is_faithful, reason = parse_verdict(content)
    assert is_faithful is True
    assert "supported" in reason.lower()


def test_parse_verdict_unfaithful():
    """UNFAITHFUL verdict should return False."""
    content = ("VERDICT: UNFAITHFUL\n"
               "REASON: Claim about inventor not in context.")
    is_faithful, reason = parse_verdict(content)
    assert is_faithful is False
    assert len(reason) > 0


def test_parse_verdict_default_faithful():
    """Unclear response should default to faithful."""
    content = "This is unclear output from the model."
    is_faithful, reason = parse_verdict(content)
    assert is_faithful is True


def test_parse_verdict_extracts_reason():
    """Reason should be extracted from REASON: line."""
    content = "VERDICT: FAITHFUL\nREASON: Evidence found in source 1."
    _, reason = parse_verdict(content)
    assert reason == "Evidence found in source 1."


# ── Graph integration test ───────────────────────────

def test_graph_builds_without_error():
    """The full agent graph should compile cleanly."""
    from agents.graph import build_agent_graph
    app = build_agent_graph()
    assert app is not None


def test_graph_has_three_nodes():
    """Graph should contain planner, executor, critic."""
    from agents.graph import build_agent_graph
    app = build_agent_graph()
    # LangGraph compiled app has a graph attribute
    nodes = list(app.get_graph().nodes.keys())
    assert "planner"  in nodes
    assert "executor" in nodes
    assert "critic"   in nodes