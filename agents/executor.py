"""
Executor Agent — the second agent in the pipeline.

Responsibility:
- Receive sub-queries from Planner
- Run each sub-query against Weaviate (hybrid search)
- Aggregate and deduplicate chunks
- Build context string
- Generate answer from LLM using context

Industry pattern: This is called "retrieval + synthesis"
The executor does both — retrieves evidence then
synthesises an answer grounded in that evidence.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

import weaviate
from sentence_transformers import SentenceTransformer
from openai import OpenAI as OpenAIClient

WEAVIATE_URL = os.getenv("WEAVIATE_URL",
                          "http://localhost:8080")
COLLECTION   = "Document"
TOP_K        = 5    # chunks per sub-query

_weaviate_client = None
_embedder        = None


def get_weaviate():
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client =client = weaviate.connect_to_local(
                host="weaviate",
                port=8080,
                grpc_port=50051,
                skip_init_checks=True
            )
    return _weaviate_client


def get_embedder():
    global _embedder
    if _embedder is None:
        # Use fine-tuned model if available
        ft_path = "models/finetuned/best"
        fallback = "models/finetuned/final"
        if os.path.exists(ft_path):
            _embedder = SentenceTransformer(ft_path)
        elif os.path.exists(fallback):
            _embedder = SentenceTransformer(fallback)
        else:
            _embedder = SentenceTransformer(
                "all-MiniLM-L6-v2")
    return _embedder


def search_single_query(
    query: str,
    top_k: int = TOP_K
) -> list:
    """
    Run one query against Weaviate using hybrid search.
    Hybrid = BM25 keyword + vector semantic combined.

    Returns list of chunk dicts.
    """
    client = get_weaviate()
    model  = get_embedder()

    # Vector search
    query_vector = model.encode(query).tolist()

    result = (
        client.query
        .get(COLLECTION, ["text", "source", "page"])
        .with_near_vector({"vector": query_vector})
        .with_additional(["distance"])
        .with_limit(top_k)
        .do()
    )

    docs = (
        result.get("data", {})
              .get("Get", {})
              .get(COLLECTION, [])
    ) or []

    chunks = []
    for doc in docs:
        dist  = float(
            doc.get("_additional", {})
               .get("distance", 1.0)
        )
        chunks.append({
            "text":   doc.get("text", ""),
            "source": doc.get("source", ""),
            "page":   doc.get("page", 0),
            "score":  round(1.0 - dist, 4),
            "query":  query,    # track which query found it
        })

    return chunks


def deduplicate_chunks(chunks: list) -> list:
    """
    Remove duplicate chunks based on first 100 chars.

    When multiple sub-queries retrieve the same chunk,
    keep the one with the highest score.
    Deduplication is critical — without it the same
    text appears multiple times in the LLM context.
    """
    seen    = {}
    for chunk in chunks:
        key = chunk["text"][:100].strip()
        if key not in seen:
            seen[key] = chunk
        elif chunk["score"] > seen[key]["score"]:
            seen[key] = chunk   # keep higher score

    deduped = list(seen.values())
    # Sort by score descending
    deduped.sort(key=lambda x: x["score"], reverse=True)
    return deduped


def build_context(chunks: list, max_chunks: int = 8) -> str:
    """
    Build a context string from the top chunks.
    Passed to the LLM as the knowledge base.

    max_chunks=8 keeps context under ~3000 tokens —
    safe for gpt-4o-mini's context window.
    """
    top = chunks[:max_chunks]
    parts = []
    for i, chunk in enumerate(top, 1):
        parts.append(
            f"[{i}] Source: {chunk['source']} "
            f"(page {chunk['page']})\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    context:  str
) -> str:
    """
    Generate a grounded answer using the LLM.
    Instructs the model to ONLY use the provided context.
    """
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/kashyapbhanderi/doc-intelligence-platform",
            "X-Title": "Doc Intelligence Platform"
        }
    )

    prompt = f"""You are an AI assistant answering questions
about AI/ML research papers.

Answer the question using ONLY the context below.
If the context does not contain enough information,
say "I could not find sufficient information in
the retrieved documents."
Always mention which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()


def executor_node(state: dict) -> dict:
    question    = state["question"]
    sub_queries = state.get("sub_queries", [question])

    print(f"\n⚙️  EXECUTOR: running "
          f"{len(sub_queries)} sub-queries...")

    # Reduce top_k when many sub-queries to stay fast
    # 3 queries × 3 chunks = 9 unique chunks (enough)
    # 3 queries × 5 chunks = 15 (slower, more dupes)
    per_query_k = 3 if len(sub_queries) >= 3 else TOP_K

    all_chunks = []
    for i, query in enumerate(sub_queries, 1):
        print(f"   [{i}/{len(sub_queries)}] "
              f"{query[:55]}...")
        chunks = search_single_query(query, per_query_k)
        all_chunks.extend(chunks)
        print(f"   → {len(chunks)} chunks retrieved")

    # Deduplicate
    deduped = deduplicate_chunks(all_chunks)
    print(f"   Deduped: {len(all_chunks)} → "
          f"{len(deduped)} unique chunks")

    if not deduped:
        return {
            "retrieved_chunks": [],
            "context":          "",
            "answer":  "No relevant documents found.",
            "sources":          [],
        }

    # Build context
    context = build_context(deduped, max_chunks=8)

    # Generate answer
    print(f"   Generating answer from "
          f"{min(len(deduped), 8)} chunks...")
    answer = generate_answer(question, context)

    # Build sources list
    sources = [
        {
            "source": c["source"],
            "page":   c["page"],
            "score":  c["score"],
        }
        for c in deduped[:5]
    ]

    print(f"   ✓ Answer generated "
          f"({len(answer)} chars)")

    return {
        "retrieved_chunks": deduped,
        "context":          context,
        "answer":           answer,
        "sources":          sources,
    }


if __name__ == "__main__":
    # Test executor standalone
    test_state = {
        "question": "What is retrieval augmented generation?",
        "sub_queries": [
            "What is retrieval augmented generation?",
            "How does RAG improve LLM accuracy?",
            "What are the components of a RAG system?",
        ]
    }

    print("Testing Executor Agent")
    print("=" * 60)

    result = executor_node(test_state)

    print(f"\nAnswer ({len(result['answer'])} chars):")
    print(result["answer"][:400])
    print(f"\nSources used: {len(result['sources'])}")
    for s in result["sources"][:3]:
        print(f"  → {s['source']} "
              f"(page {s['page']}, "
              f"score: {s['score']})")