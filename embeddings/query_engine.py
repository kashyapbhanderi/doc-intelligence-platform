import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath('.'))

from sentence_transformers import SentenceTransformer
import weaviate
from openai import OpenAI as OpenAIClient

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "Document"

_embedder       = None
_weaviate_client = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_weaviate_client(url="http://localhost:8080"):
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client = weaviate.Client(url)
    return _weaviate_client


def retrieve_chunks(question: str, top_k: int = 5,
                    weaviate_url: str = "http://localhost:8080") -> list:
    """Embed question and search Weaviate for the most relevant chunks."""
    embedder = get_embedder()
    client   = get_weaviate_client(weaviate_url)

    query_vector = embedder.encode(question).tolist()

    raw = (
        client.query
        .get(COLLECTION, ["text", "source", "page"])
        .with_near_vector({"vector": query_vector})
        .with_additional(["distance"])
        .with_limit(top_k)
        .do()
    )

    docs = raw.get("data", {}).get("Get", {}).get(COLLECTION, []) or []

    chunks = []
    for doc in docs:
        distance = float(doc.get("_additional", {}).get("distance", 1.0))
        chunks.append({
            "text":     doc.get("text", ""),
            "source":   doc.get("source", ""),
            "page":     doc.get("page", 0),
            # "chunk_id": doc.get("chunk_id", ""),
            "score":    round(1.0 - distance, 4),
        })

    return chunks


def generate_answer(question: str, chunks: list) -> str:
    """Build a grounded prompt from retrieved chunks and call the LLM."""
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/kashyapbhanderi/doc-intelligence-platform",
            "X-Title": "Doc Intelligence Platform"
        }
    )

    context = "\n\n".join(
        f"[{i}] Source: {c['source']} (page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks[:5], 1)
    )

    prompt = f"""Answer the question using ONLY the context below.
If the context does not contain enough information, say so clearly.
Always mention which source you used.

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

    return response.choices[0].message.content


def build_query_engine(weaviate_url: str = "http://localhost:8080",
                       top_k: int = 5):
    """Initialise and warm up the query engine components."""
    print("Connecting to Weaviate...")
    get_weaviate_client(weaviate_url)

    get_embedder()   # warm up — loads model weights once

    print("Query engine ready!")

    class QueryEngine:
        def __init__(self, url, k):
            self.weaviate_url = url
            self.top_k        = k

    return QueryEngine(weaviate_url, top_k), get_weaviate_client(weaviate_url)


def query_with_sources(query_engine, question: str) -> dict:
    """Ask a question and return answer + source citations."""
    chunks = retrieve_chunks(question, query_engine.top_k,
                             query_engine.weaviate_url)

    if not chunks:
        return {"question": question, "answer": "No relevant documents found.",
                "sources": [], "num_sources": 0}

    answer  = generate_answer(question, chunks)
    sources = [{"source": c["source"], "page": c["page"],
                "score": c["score"], "text_preview": c["text"][:150]}
               for c in chunks]

    return {"question": question, "answer": answer,
            "sources": sources, "num_sources": len(sources)}


if __name__ == "__main__":
    print("Building query engine...")
    engine, client = build_query_engine(top_k=5)

    test_q = "What is retrieval augmented generation?"
    print(f"\nTest question: {test_q}")
    print("-" * 50)

    result = query_with_sources(engine, test_q)
    print(f"Answer: {result['answer'][:300]}")
    print(f"\nSources used: {result['num_sources']}")
    for s in result['sources'][:3]:
        print(f"  - {s['source']} (page {s['page']}, score: {s['score']})")