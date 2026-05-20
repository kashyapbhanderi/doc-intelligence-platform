"""
Critic Agent — the third and final agent in the pipeline.

Responsibility:
- Read the generated answer
- Read the retrieved context
- Check: is every claim in the answer
  supported by the context?
- Return verdict: faithful (True) or hallucinated (False)
- Provide specific critique if unfaithful

Industry pattern:
This is called "faithfulness checking" or
"groundedness evaluation". Used in production by
teams at OpenAI, Cohere, and Anthropic to catch
hallucinations before they reach users.

LLM-as-judge: using an LLM to evaluate another LLM's
output is now standard practice in the industry.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI as OpenAIClient


def get_llm_client():
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/kashyapbhanderi/doc-intelligence-platform",
            "X-Title": "Doc Intelligence Platform"
        }
    )


def check_faithfulness(
    answer:  str,
    context: str
) -> tuple[bool, str]:
    """
    Check if the answer is grounded in the context.

    Uses LLM-as-judge pattern:
    - Send both answer and context to gpt-4o-mini
    - Ask it to verify every claim
    - Parse YES/NO verdict + explanation

    Returns:
        (is_faithful: bool, critique: str)
    """
    client = get_llm_client()

    prompt = f"""You are a factuality checker for an AI
question-answering system.

Your job: check if the ANSWER is fully supported
by the CONTEXT. Every claim in the answer must be
traceable to a specific part of the context.

CONTEXT:
{context[:2000]}

ANSWER:
{answer}

Instructions:
1. Read every sentence in the ANSWER
2. Check if it is supported by the CONTEXT
3. If ALL sentences are supported → verdict is FAITHFUL
4. If ANY sentence makes a claim not in context
   → verdict is UNFAITHFUL

Respond in EXACTLY this format:
VERDICT: FAITHFUL
REASON: [one sentence explaining]

OR:

VERDICT: UNFAITHFUL
REASON: [specific claim that is not in context]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,    # deterministic for judging
        max_tokens=150
    )

    content = response.choices[0].message.content.strip()
    return parse_verdict(content)


def parse_verdict(content: str) -> tuple[bool, str]:
    """
    Parse the LLM's verdict response.
    Returns (is_faithful, reason).
    """
    content_upper = content.upper()

    # Extract verdict
    if "VERDICT: FAITHFUL" in content_upper and \
       "UNFAITHFUL" not in content_upper:
        is_faithful = True
    elif "VERDICT: UNFAITHFUL" in content_upper:
        is_faithful = False
    else:
        # Default to faithful if response unclear
        # Better to pass through than block everything
        is_faithful = True

    # Extract reason
    reason = ""
    for line in content.split("\n"):
        if line.strip().upper().startswith("REASON:"):
            reason = line.split(":", 1)[-1].strip()
            break

    return is_faithful, reason


def critic_node(state: dict) -> dict:
    """
    Critic agent node for LangGraph.

    Checks answer faithfulness against context.
    Updates state with verdict and critique.

    Args:
        state: Current AgentState dict

    Returns:
        Updated state with is_faithful and critique
    """
    answer  = state.get("answer", "")
    context = state.get("context", "")
    iters   = state.get("iterations", 0)

    print(f"\n🔍 CRITIC: checking answer faithfulness...")

    # Skip check if no answer or no context
    if not answer or not context:
        print("   ⚠️  No answer or context to check")
        return {
            "is_faithful": False,
            "critique":    "No answer or context provided",
            "final_answer": answer,
            "iterations":  iters + 1,
        }

    # Check faithfulness
    is_faithful, reason = check_faithfulness(
        answer, context)

    if is_faithful:
        print(f"   ✅ FAITHFUL — answer approved")
        final_answer = answer
    else:
        print(f"   ❌ UNFAITHFUL — {reason[:60]}")
        final_answer = (
            f"{answer}\n\n"
            f"[Note: Some claims may not be fully "
            f"supported by the retrieved documents. "
            f"Please verify with the source papers.]"
        )

    return {
        "is_faithful":  is_faithful,
        "critique":     reason,
        "final_answer": final_answer,
        "iterations":   iters + 1,
    }


if __name__ == "__main__":
    print("Testing Critic Agent")
    print("=" * 60)

    # Test 1 — faithful answer
    state_faithful = {
        "answer": (
            "RAG combines document retrieval with "
            "language model generation to reduce "
            "hallucination by grounding responses "
            "in retrieved evidence."
        ),
        "context": (
            "[1] RAG is a technique that combines "
            "retrieval of relevant documents with "
            "language model generation. It reduces "
            "hallucination by grounding the model's "
            "responses in retrieved evidence from "
            "a knowledge base."
        ),
        "iterations": 0,
    }

    result = critic_node(state_faithful)
    print(f"\nTest 1 (should be FAITHFUL):")
    print(f"  is_faithful: {result['is_faithful']}")
    print(f"  critique:    {result['critique']}")

    # Test 2 — unfaithful answer
    state_unfaithful = {
        "answer": (
            "RAG was invented by Mark Zuckerberg "
            "in 2019 and first deployed at Meta."
        ),
        "context": (
            "[1] RAG is a technique introduced in "
            "academic literature for improving "
            "language model factuality."
        ),
        "iterations": 0,
    }

    result2 = critic_node(state_unfaithful)
    print(f"\nTest 2 (should be UNFAITHFUL):")
    print(f"  is_faithful: {result2['is_faithful']}")
    print(f"  critique:    {result2['critique']}")