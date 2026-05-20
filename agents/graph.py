"""
agents/graph.py
Full 3-agent LangGraph pipeline.

Flow:
  User question
       ↓
  [PLANNER] — decomposes into sub-queries
       ↓
  [EXECUTOR] — searches Weaviate, generates answer
       ↓
  [CRITIC] — checks faithfulness
       ↓
  Verified answer returned to user

State flows through all three nodes.
Each node reads what it needs and writes its output.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from langgraph.graph import StateGraph, END
from agents.state import AgentState, create_initial_state
from agents.planner import planner_node
from agents.executor import executor_node
from agents.critic import critic_node
import os
from dotenv import load_dotenv
load_dotenv()

# LangSmith tracing — set in .env to enable
# All agent runs will appear at smith.langchain.com
os.environ.setdefault("LANGCHAIN_TRACING_V2",
    os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY",
    os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT",
    os.getenv("LANGCHAIN_PROJECT",
               "doc-intelligence-platform"))


def build_agent_graph():
    """
    Build and compile the full 3-agent pipeline.

    Returns a compiled LangGraph app ready for invoke().
    """
    graph = StateGraph(AgentState)

    # Add all three agent nodes
    graph.add_node("planner",  planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic",   critic_node)

    # Wire the pipeline: planner → executor → critic → end
    graph.set_entry_point("planner")
    graph.add_edge("planner",  "executor")
    graph.add_edge("executor", "critic")
    graph.add_edge("critic",   END)

    return graph.compile()


def ask(question: str, verbose: bool = True) -> dict:
    """
    Ask a question through the full agent pipeline.

    Args:
        question: Natural language question
        verbose:  Print agent progress if True

    Returns:
        Final AgentState with answer + sources
    """
    app   = build_agent_graph()
    state = create_initial_state(question)

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

    result = app.invoke(state)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"FINAL ANSWER:")
        print(result["final_answer"])
        print(f"\nFaithful: {result['is_faithful']}")
        if result["critique"]:
            print(f"Critique: {result['critique']}")
        print(f"\nSources ({len(result['sources'])}):")
        for s in result["sources"][:3]:
            print(f"  → {s['source']} "
                  f"(page {s['page']}, "
                  f"score: {s['score']})")
        print(f"{'='*60}")

    return result


if __name__ == "__main__":
    # Test the full pipeline
    test_questions = [
        "What is retrieval augmented generation?",
        "How does LoRA reduce memory usage?",
    ]

    print("Testing Full Agent Pipeline")
    print("Planner → Executor → Critic")
    print("=" * 60)

    for question in test_questions:
        result = ask(question)
        print(f"\nSub-queries used: "
              f"{result['sub_queries']}")
        print(f"Chunks retrieved: "
              f"{len(result['retrieved_chunks'])}")
        print()