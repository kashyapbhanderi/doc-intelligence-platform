import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from agents.state import (
    AgentState,
    create_initial_state
)
from agents.planner import parse_sub_queries


# ── State Tests ─────────────────────────────────

def test_initial_state_has_question():
    """State should contain original question."""
    state = create_initial_state("Test question?")
    assert state["question"] == "Test question?"


def test_initial_state_empty_fields():
    """All list/string fields should start empty."""
    state = create_initial_state("Test?")
    assert state["sub_queries"] == []
    assert state["retrieved_chunks"] == []
    assert state["context"] == ""
    assert state["answer"] == ""
    assert state["final_answer"] == ""
    assert state["sources"] == []
    assert state["error"] == ""


def test_initial_state_defaults():
    """Default values should be correct."""
    state = create_initial_state("Test?")
    assert state["is_faithful"] == False
    assert state["iterations"] == 0
    assert state["critique"] == ""


def test_state_has_all_required_fields():
    """State must have all 11 required fields."""
    state = create_initial_state("Test?")
    required = [
        "question", "sub_queries",
        "retrieved_chunks", "context",
        "answer", "is_faithful", "critique",
        "final_answer", "sources",
        "iterations", "error"
    ]
    for field in required:
        assert field in state, \
            f"Missing field: {field}"


def test_state_question_preserved():
    """Question should never be modified."""
    q = "What is the meaning of life?"
    state = create_initial_state(q)
    assert state["question"] == q


# ── Planner Parser Tests ────────────────────────

def test_parse_python_list():
    """Should parse valid Python list string."""
    content = '["query 1", "query 2", "query 3"]'
    result = parse_sub_queries(content, "original")
    assert len(result) == 3
    assert result[0] == "query 1"


def test_parse_quoted_strings():
    """Should extract quoted strings."""
    content = '"first query" and "second query"'
    result = parse_sub_queries(content, "original")
    assert len(result) >= 2


def test_parse_fallback_to_original():
    """Should fall back to original if parsing fails."""
    content = "unparseable gibberish!!!"
    result = parse_sub_queries(content, "original q")
    assert len(result) >= 1
    assert result[0] == "original q"


def test_parse_removes_short_queries():
    """Should filter out very short queries."""
    content = '["ok", "this is a real query here"]'
    result = parse_sub_queries(content, "fallback")
    for q in result:
        assert len(q) > 5


def test_parse_max_three_queries():
    """Should return at most 3 sub-queries."""
    content = ('["q1 with words", "q2 with words", '
               '"q3 with words", "q4 with words", '
               '"q5 with words"]')
    result = parse_sub_queries(content, "original")
    assert len(result) <= 3


# ── LangGraph Tests ─────────────────────────────

def test_langgraph_imports():
    """LangGraph should be importable."""
    from langgraph.graph import StateGraph, END
    assert StateGraph is not None
    assert END is not None


def test_simple_graph_runs():
    """A simple 2-node graph should run correctly."""
    from langgraph.graph import StateGraph, END
    from typing import TypedDict

    class TestState(TypedDict):
        value: int
        done: bool

    def add_one(state):
        return {"value": state["value"] + 1}

    def check_done(state):
        return {"done": state["value"] > 3}

    graph = StateGraph(TestState)
    graph.add_node("add", add_one)
    graph.add_node("check", check_done)
    graph.set_entry_point("add")
    graph.add_edge("add", "check")
    graph.add_edge("check", END)

    app = graph.compile()
    result = app.invoke({"value": 1, "done": False})

    assert result["value"] == 2
    assert result["done"] == False