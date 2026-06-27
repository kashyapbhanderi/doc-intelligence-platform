"""
agents/editor_agent.py

The Editor Agent — fourth agent in the system.

Unlike the RAG pipeline (Planner→Executor→Critic)
which READS and ANSWERS, the Editor Agent:
1. Reads the user instruction
2. Identifies the file and what to do
3. Selects the right tool
4. Executes the edit
5. Returns the result

This uses a ReAct (Reason + Act) pattern:
- Reason: what tool do I need?
- Act: call the tool
- Observe: did it work?
- Repeat if needed
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="create_react_agent has been moved"
)
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from agents.editor_tools.tool_registry import (
    ALL_EDITOR_TOOLS
)


def get_editor_llm():
    """Get LLM for Editor Agent — uses centralized config."""
    from langchain_openai import ChatOpenAI
    from config.llm_config import get_llm_config

    cfg = get_llm_config()
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=0.0,
        max_tokens=512
    )

def build_editor_agent():
    """
    Build the Editor Agent using LangGraph's
    create_react_agent.

    create_react_agent = prebuilt ReAct loop:
    1. LLM decides which tool to call
    2. Tool runs
    3. LLM sees result
    4. Repeat until task done

    Returns compiled agent app.
    """
    llm = get_editor_llm()

    agent = create_react_agent(
        model=llm,
        tools=ALL_EDITOR_TOOLS,
    )
    return agent


def run_editor(instruction: str,
               verbose: bool = True) -> dict:
    """
    Run the Editor Agent on an instruction.

    Args:
        instruction: Natural language edit command
                     e.g. "Add watermark DRAFT to
                     data/test_docs/report.pdf"
        verbose: Print progress

    Returns:
        Dict with result and steps taken
    """
    agent = build_editor_agent()

    if verbose:
        print(f"\n{'='*60}")
        print(f"EDITOR AGENT")
        print(f"Instruction: {instruction}")
        print(f"{'='*60}")

    messages = [HumanMessage(content=instruction)]
    result   = agent.invoke({"messages": messages})

    # Extract final answer
    final_msg = result["messages"][-1]
    answer    = (
        final_msg.content
        if hasattr(final_msg, "content")
        else str(final_msg)
    )

    # Count tool calls
    tool_calls = sum(
        1 for m in result["messages"]
        if hasattr(m, "type") and
        m.type == "tool"
    )

    if verbose:
        print(f"\nResult: {answer[:300]}")
        print(f"Tool calls made: {tool_calls}")
        print(f"{'='*60}")

    return {
        "instruction": instruction,
        "result":      answer,
        "tool_calls":  tool_calls,
        "messages":    result["messages"],
    }


if __name__ == "__main__":
    # Test the editor agent
    test_instructions = [
        (
            "Read the Word document at "
            "data/test_docs/test_report.docx "
            "and tell me how many words it has."
        ),
        (
            "In the file "
            "data/test_docs/test_report.docx, "
            "replace the word 'FINAL' with 'DRAFT'."
        ),
    ]

    print("Testing Editor Agent")
    print("=" * 60)

    for instruction in test_instructions:
        result = run_editor(instruction)
        print()