"""
agents/router.py

Intent detection — decides whether a user request
should go to the RAG pipeline or the Editor Agent.

Industry pattern: "intent classification" or
"task routing". Every production AI assistant
has a router that decides which backend to use.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()


# Keywords that signal edit intent
EDIT_KEYWORDS = [
    "edit", "change", "replace", "update",
    "modify", "watermark", "merge", "split",
    "convert", "resize", "add table", "insert",
    "rename", "rewrite", "fix the", "correct",
    "remove background", "add heading",
    "pdf to word", "word to pdf",
]

# Keywords that signal question/RAG intent
QUESTION_KEYWORDS = [
    "what is", "what are", "how does", "how do",
    "explain", "describe", "tell me about",
    "summarize", "compare", "why", "when",
    "who invented", "what paper",
]


def detect_intent(instruction: str) -> str:
    """
    Detect whether instruction is an edit task
    or a question/RAG task.

    Uses keyword matching first (fast, no API).
    Falls back to LLM classification if unclear.

    Returns:
        "edit"     — route to Editor Agent
        "question" — route to RAG pipeline
    """
    lower = instruction.lower()

    # Check edit keywords
    for kw in EDIT_KEYWORDS:
        if kw in lower:
            return "edit"

    # Check question keywords
    for kw in QUESTION_KEYWORDS:
        if lower.startswith(kw) or kw in lower:
            return "question"

    # If file extension mentioned → edit
    for ext in [".docx", ".pdf", ".png",
                ".jpg", ".jpeg", ".xlsx"]:
        if ext in lower:
            return "edit"

    # Default to question for ambiguous input
    return "question"


def route_request(
    instruction: str,
    verbose: bool = True
) -> dict:
    """
    Route a user request to the right agent.

    Args:
        instruction: User's natural language request
        verbose:     Print routing decision

    Returns:
        Dict with intent and result from agent
    """
    intent = detect_intent(instruction)

    if verbose:
        print(f"\n🔀 ROUTER: '{instruction[:50]}...'")
        print(f"   Intent detected: {intent.upper()}")

    if intent == "edit":
        from agents.editor_agent import run_editor
        result = run_editor(
            instruction, verbose=verbose)
        return {
            "intent": "edit",
            "result": result["result"],
            "details": result
        }
    else:
        from agents.graph import ask
        result = ask(instruction, verbose=verbose)
        return {
            "intent": "question",
            "result": result.get(
                "final_answer", ""),
            "details": result
        }


if __name__ == "__main__":
    test_requests = [
        "What is retrieval augmented generation?",
        "Add a DRAFT watermark to "
        "data/test_docs/test_report.docx",
        "How does LoRA reduce memory usage?",
        "Replace FINAL with DRAFT in "
        "data/test_docs/test_report.docx",
    ]

    print("Testing Router")
    print("=" * 60)

    for req in test_requests:
        intent = detect_intent(req)
        icon   = "📝" if intent == "edit" else "❓"
        print(f"{icon} [{intent.upper()}] {req[:60]}")