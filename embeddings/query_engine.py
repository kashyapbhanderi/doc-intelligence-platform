import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath('.'))

from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.query import MetadataQuery
from openai import OpenAI as OpenAIClient

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "Document"

_embedder = None
_weaviate_client = None


def get_embedder():
    global _embedder

    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL)

    return _embedder


def get_weaviate_client():
    """
    Create/reuse Weaviate v4 client.
    """
    global _weaviate_client

    if _weaviate_client is None:
        _weaviate_client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )

    return _weaviate_client


def retrieve_chunks(
    question: str,
    top_k: int = 5
) -> list:
    """
    Embed question and search Weaviate for relevant chunks.
    """

    embedder = get_embedder()
    client = get_weaviate_client()

    query_vector = embedder.encode(question).tolist()

    collection = client.collections.get(COLLECTION)

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True)
    )

    chunks = []

    for obj in response.objects:
        props = obj.properties
        distance = obj.metadata.distance or 1.0

        chunks.append({
            "text": props.get("text", ""),
            "source": props.get("source", ""),
            "page": props.get("page", 0),
            "chunk_id": props.get("chunk_id", ""),
            "score": round(1.0 - float(distance), 4),
        })

    return chunks


# def generate_answer(question: str, chunks: list) -> str:
#     """
#     Build grounded prompt and generate answer.
#     """

#     api_key = os.getenv("OPENAI_API_KEY")
#     base_url = os.getenv(
#         "OPENAI_BASE_URL",
#         "https://api.openai.com/v1"
#     )

#     client = OpenAIClient(
#         api_key=api_key,
#         base_url=base_url,
#         default_headers={
#             "HTTP-Referer":
#                 "https://github.com/kashyapbhanderi/doc-intelligence-platform",
#             "X-Title":
#                 "Doc Intelligence Platform"
#         }
#     )

#     context = "\n\n".join([
#     c["text"][:120] for c in chunks[:2]
#     ])

#     prompt = f"""
# You are a strict RAG assistant.

# Answer ONLY using the provided context.

# If the answer is not explicitly stated in the context,
# say:
# "I could not find this information in the provided documents."

# Keep answers concise and factual.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = client.chat.completions.create(
#         model="openai/gpt-5.4-mini",
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.1,
#         max_tokens=5

#      )

#     return response.choices[0].message.content
def generate_answer(question, chunks):
    """
    Generate concise answer from retrieved chunks.
    """

    from openai import OpenAI
    import os

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    # ULTRA-SHORT context
    context = ""

    if chunks:
        context = chunks[0]["text"][:40]

    # VERY SHORT PROMPT
    prompt = f"Q:{question}\nC:{context}\nA:"

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=16,
        temperature=0
    )

    return response.choices[0].message.content.strip()

def build_query_engine(top_k: int = 5):
    """
    Initialize query engine.
    """

    print("Connecting to Weaviate...")
    client = get_weaviate_client()

    print("Loading embedder...")
    get_embedder()

    print("Query engine ready!")

    class QueryEngine:
        def __init__(self, k):
            self.top_k = k

    return QueryEngine(top_k), client


def query_with_sources(
    query_engine,
    question: str
) -> dict:
    """
    Ask question and return answer + sources.
    """

    chunks = retrieve_chunks(
        question,
        query_engine.top_k
    )

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documents found.",
            "sources": [],
            "num_sources": 0
        }

    answer = generate_answer(question, chunks)

    sources = [
        {
            "source": c["source"],
            "page": c["page"],
            "score": c["score"],
            "text_preview": c["text"][:150]
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources)
    }


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
        print(
            f"  - {s['source']} "
            f"(page {s['page']}, score: {s['score']})"
        )

    # IMPORTANT FOR V4
    client.close()