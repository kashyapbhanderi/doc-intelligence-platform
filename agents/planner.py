"""
Planner Agent — the first agent in the pipeline.

Responsibility:
- Receive the user's question
- Decompose it into 2-3 focused sub-queries
- Each sub-query targets a specific aspect

Why decompose?
Complex questions like "How does RAG compare to
fine-tuning for domain adaptation?" need multiple
searches — one for RAG, one for fine-tuning, one
for comparison. A single search misses aspects.

Industry pattern: This is called
"query decomposition" or "query expansion".
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()


# def get_llm():
#     """Get LLM client — supports OpenAI + OpenRouter."""
#     from langchain_openai import ChatOpenAI

#     api_key = os.getenv("OPENAI_API_KEY")
#     base_url = os.getenv(
#         "OPENAI_BASE_URL",
#         "https://api.openai.com/v1"
#     )

#     # Use cheaper model for planning
#     if "openrouter" in base_url:
#         model = "openai/gpt-4o-mini"
#     else:
#         model = "gpt-4o-mini"

#     return ChatOpenAI(
#         model=model,
#         api_key=api_key,
#         base_url=base_url,
#         temperature=0.1,
#         max_tokens=150  #<-- was 300, planner only needs a short list
#     )

def get_llm():
    """Get LLM — auto-detects OpenRouter vs Ollama."""
    from langchain_openai import ChatOpenAI
    from config.llm_config import get_llm_config

    cfg = get_llm_config()
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=0.1,
        max_tokens=300
    )

def planner_node(state: dict) -> dict:
    """
    Planner agent node for LangGraph.

    Takes the user question from state and
    decomposes it into focused sub-queries.

    Args:
        state: Current AgentState dict

    Returns:
        Updated state with sub_queries filled
    """
    question = state["question"]
    print(f"\n🗺️  PLANNER: {question[:60]}...")

    try:
        llm = get_llm()

        prompt = f"""You are a query decomposition expert.

Break this question into 2-3 focused sub-queries 
for searching a research paper database.

Question: {question}

Rules:
- Each sub-query should target one specific aspect
- Sub-queries should be shorter and more focused
- Together they should cover the full question
- Return ONLY a Python list of strings

Example:
Question: "How does LoRA compare to full fine-tuning 
in terms of memory and performance?"

Output: ["What is LoRA fine-tuning?", 
"How does LoRA reduce memory usage?",
"How does LoRA performance compare to full fine-tuning?"]

Now decompose:
Question: "{question}"

Output (Python list only, no explanation):"""

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse the list from response
        sub_queries = parse_sub_queries(
            content, question)

        print(f"   Sub-queries generated: "
              f"{len(sub_queries)}")
        for i, q in enumerate(sub_queries, 1):
            print(f"   {i}. {q[:60]}")

        return {"sub_queries": sub_queries}

    except Exception as e:
        print(f"   Planner error: {e}")
        # Fallback: use original question as one query
        return {"sub_queries": [question]}


def parse_sub_queries(
    content: str,
    original_question: str
) -> list:
    """
    Parse LLM response into list of sub-queries.
    Handles various output formats robustly.
    """
    import ast
    import re

    content = content.strip()

    # Try direct Python list parsing
    try:
        if content.startswith("["):
            queries = ast.literal_eval(content)
            if (isinstance(queries, list) and
                    all(isinstance(q, str)
                        for q in queries)):
                clean = [q for q in queries
                         if len(q.strip()) > 5]
                return clean[:3]              # ← fix: cap at 3
    except Exception:
        pass

    # Try extracting quoted strings
    patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if len(matches) >= 2:
            return [m for m in matches
                    if len(m.strip()) > 5][:3]

    # Try line-by-line parsing
    # Only use this if we find 2+ valid lines
    # (single lines fall through to fallback below)
    lines = content.split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^[\d\.\-\*\•]\s*', '', line)
        line = line.strip('"\'')
        if len(line) > 10:
            queries.append(line)

    if len(queries) >= 2:             # ← fix: need 2+ lines to trust this
        return queries[:3]

    # Final fallback — return original question
    return [original_question]        # ← fix: always return original, not content


if __name__ == "__main__":
    # Test planner
    test_questions = [
        "What is retrieval augmented generation?",
        "How does LoRA fine-tuning reduce memory?",
        "Compare dense and sparse retrieval methods",
    ]

    print("Testing Planner Agent")
    print("=" * 60)

    for question in test_questions:
        state = {
            "question": question,
            "sub_queries": []
        }
        result = planner_node(state)
        print(f"\nQ: {question}")
        print(f"Sub-queries:")
        for q in result["sub_queries"]:
            print(f"  → {q}")
        print()