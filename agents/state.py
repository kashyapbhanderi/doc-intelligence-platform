"""
Agent State — the shared memory of the entire
multi-agent system.

Every agent (Planner, Executor, Critic) reads
from and writes to this state object.

Industry term: This is called a "state machine"
pattern — agents communicate through shared state
rather than directly calling each other.
"""
from typing import TypedDict, Optional
import operator


class AgentState(TypedDict):
    """
    Complete state shared between all agents.

    Fields:
    -------
    question : str
        Original user question. Never modified.

    sub_queries : list[str]
        Planner breaks question into sub-queries.
        Example: "What is RAG and how does it work?"
        → ["What is RAG?", "How does RAG work?"]

    retrieved_chunks : list[dict]
        Executor fills this with search results.
        Each dict has: text, source, page, score

    context : str
        Combined text from all retrieved chunks.
        Passed to LLM as context for generation.

    answer : str
        Generated answer from LLM.
        Filled by Executor after retrieval.

    is_faithful : bool
        Whether Critic approved the answer.
        True = answer is grounded in retrieved context.
        False = answer has hallucinations.

    critique : str
        Critic's explanation if answer is unfaithful.
        Empty string if answer is approved.

    final_answer : str
        The answer returned to user.
        Same as answer if faithful.
        Re-generated if Critic flagged issues.

    sources : list[dict]
        Source documents used for the answer.
        Each: {source, page, score}

    iterations : int
        How many times Critic has sent answer back.
        Prevents infinite loops (max 2 iterations).

    error : str
        Error message if something fails.
        Empty string if no error.
    """
    question: str
    sub_queries: list
    retrieved_chunks: list
    context: str
    answer: str
    is_faithful: bool
    critique: str
    final_answer: str
    sources: list
    iterations: int
    error: str


def create_initial_state(question: str) -> AgentState:
    """
    Create a fresh initial state for a new question.

    All fields start empty/default.
    Only question is set from user input.

    Args:
        question: The user's question

    Returns:
        Fresh AgentState ready for pipeline
    """
    return AgentState(
        question=question,
        sub_queries=[],
        retrieved_chunks=[],
        context="",
        answer="",
        is_faithful=False,
        critique="",
        final_answer="",
        sources=[],
        iterations=0,
        error=""
    )


if __name__ == "__main__":
    # Test state creation
    state = create_initial_state(
        "What is retrieval augmented generation?"
    )

    print("Initial AgentState:")
    print("=" * 50)
    for key, value in state.items():
        print(f"  {key}: {repr(value)}")

    print("\nState fields defined correctly!")
    print(f"Total fields: {len(state)}")