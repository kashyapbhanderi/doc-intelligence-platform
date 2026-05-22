import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from agents.editor_tools.tool_registry import (
    ALL_EDITOR_TOOLS,
    TOOL_NAMES
)
from agents.router import detect_intent


# ── Tool registry tests ───────────────────────────────────

def test_all_tools_registered():
    """Should have at least 10 editor tools."""
    assert len(ALL_EDITOR_TOOLS) >= 10


def test_tool_names_are_strings():
    """Every tool must have a string name."""
    for t in ALL_EDITOR_TOOLS:
        assert isinstance(t.name, str)
        assert len(t.name) > 0


def test_tool_descriptions_exist():
    """Every tool must have a non-empty description."""
    for t in ALL_EDITOR_TOOLS:
        assert hasattr(t, "description")
        assert len(t.description) > 10


def test_expected_tools_present():
    """Key tools must be in the registry."""
    expected = [
        "tool_edit_docx_text",
        "tool_watermark_pdf",
        "tool_resize_image",
        "tool_read_docx",
    ]
    for name in expected:
        assert name in TOOL_NAMES, \
            f"Missing tool: {name}"


def test_no_duplicate_tool_names():
    """All tool names must be unique."""
    assert len(TOOL_NAMES) == len(set(TOOL_NAMES))


# ── Router tests ──────────────────────────────────────────

def test_router_detects_question():
    """Question phrases route to RAG."""
    questions = [
        "What is retrieval augmented generation?",
        "How does LoRA fine-tuning work?",
        "Explain the attention mechanism",
    ]
    for q in questions:
        intent = detect_intent(q)
        assert intent == "question", \
            f"Expected question for: {q}"


def test_router_detects_edit():
    """Edit keywords route to Editor Agent."""
    edits = [
        "Edit the contract.docx file",
        "Add watermark to report.pdf",
        "Replace DRAFT with FINAL in document.docx",
        "Resize the image.png to 800px",
        "Merge two pdf files together",
    ]
    for e in edits:
        intent = detect_intent(e)
        assert intent == "edit", \
            f"Expected edit for: {e}"


def test_router_detects_file_extensions():
    """File extensions in instruction → edit intent."""
    with_ext = [
        "Process the report.docx",
        "Handle this presentation.pdf",
        "Modify logo.png",
    ]
    for req in with_ext:
        intent = detect_intent(req)
        assert intent == "edit", \
            f"Expected edit for: {req}"


def test_router_defaults_to_question():
    """Ambiguous input defaults to question."""
    intent = detect_intent(
        "something completely ambiguous xyz")
    assert intent == "question"


def test_editor_agent_builds():
    """Editor agent should build without errors."""
    from agents.editor_agent import build_editor_agent
    agent = build_editor_agent()
    assert agent is not None


def test_router_file_exists():
    """Router module must exist."""
    assert os.path.exists("agents/router.py")


def test_editor_agent_file_exists():
    """Editor agent module must exist."""
    assert os.path.exists("agents/editor_agent.py")


def test_tool_registry_file_exists():
    """Tool registry must exist."""
    assert os.path.exists(
        "agents/editor_tools/tool_registry.py")