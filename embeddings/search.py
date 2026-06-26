"""
embeddings/search.py

Standalone search functions for Weaviate v3.
Function-based alternative to DocumentEmbedder's
search methods — useful when you just need to
search without loading/managing a full embedder
instance (e.g. quick scripts, other modules).

All functions are pure v3 client syntax —
no weaviate.classes, no connect_to_local().
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import weaviate
from sentence_transformers import SentenceTransformer

CLASS_NAME = "Document"

_client = None
_model  = None


def get_client(weaviate_url: str = None) -> "weaviate.Client":
    """
    Lazy-load a single shared Weaviate v3 client.
    Reads WEAVIATE_URL from env if not passed explicitly —
    matches the same env var used in docker-compose.
    """
    global _client
    if _client is None:
        url = weaviate_url or os.getenv(
            "WEAVIATE_URL", "http://localhost:8080")
        _client = weaviate.Client(url)
    return _client


def get_model(model_path: str = None) -> SentenceTransformer:
    """
    Lazy-load a single shared embedding model.
    Auto-detects fine-tuned model if no path given.
    """
    global _model
    if _model is None:
        ft_path = "models/finetuned/final"
        if model_path and os.path.exists(model_path):
            _model = SentenceTransformer(model_path)
        elif os.path.exists(ft_path):
            _model = SentenceTransformer(ft_path)
        else:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_document_count(weaviate_url: str = None) -> int:
    """Total chunks currently indexed in Weaviate."""
    client = get_client(weaviate_url)
    try:
        result = (
            client.query
            .aggregate(CLASS_NAME)
            .with_meta_count()
            .do()
        )
        return (
            result["data"]["Aggregate"][CLASS_NAME]
            [0]["meta"]["count"]
        )
    except Exception:
        return 0


def search_vector(
    query: str,
    top_k: int = 5,
    weaviate_url: str = None,
    model_path:   str = None
) -> list:
    """
    Semantic vector search — finds chunks similar
    in MEANING to the query.
    """
    client = get_client(weaviate_url)
    model  = get_model(model_path)
    vector = model.encode(query).tolist()

    try:
        result = (
            client.query
            .get(CLASS_NAME,
                 ["text", "source", "page", "chunkId"])
            .with_near_vector({"vector": vector})
            .with_limit(top_k)
            .with_additional(["certainty", "distance"])
            .do()
        )
        hits = (
            result.get("data", {})
                  .get("Get", {})
                  .get(CLASS_NAME, [])
        )
        return hits if hits is not None else []
    except Exception as e:
        print(f"  Vector search error: {e}")
        return []


def search_bm25(
    query: str,
    top_k: int = 5,
    weaviate_url: str = None
) -> list:
    """
    BM25 keyword search — finds chunks containing
    the exact words from the query.
    """
    client = get_client(weaviate_url)

    try:
        result = (
            client.query
            .get(CLASS_NAME,
                 ["text", "source", "page", "chunkId"])
            .with_bm25(query=query, properties=["text"])
            .with_limit(top_k)
            .with_additional(["score"])
            .do()
        )
        hits = (
            result.get("data", {})
                  .get("Get", {})
                  .get(CLASS_NAME, [])
        )
        return hits if hits is not None else []
    except Exception as e:
        print(f"  BM25 search error: {e}")
        return []


_embedder_singleton = None


def hybrid_search(query: str, top_k: int = 10) -> list:
    """
    Adapter: wraps DocumentEmbedder.search_hybrid() as a plain function,
    normalizing Weaviate's nested _additional score into a flat "score"
    key so agents/executor.py's deduplicate_chunks/build_context work
    unchanged.
    """
    global _embedder_singleton
    if _embedder_singleton is None:
        from embeddings.embedder import DocumentEmbedder
        _embedder_singleton = DocumentEmbedder(
            model_path="models/finetuned/best"
        )

    raw_results = _embedder_singleton.search_hybrid(query, top_k=top_k)

    normalized = []
    for r in raw_results:
        additional = r.get("_additional", {}) or {}

        if "score" in additional:
            try:
                score = float(additional["score"])
            except (TypeError, ValueError):
                score = 0.0
        elif "certainty" in additional:
            score = float(additional["certainty"])
        elif "distance" in additional:
            score = round(1.0 - float(additional["distance"]), 4)
        else:
            score = 0.0

        normalized.append({
            "text":   r.get("text", ""),
            "source": r.get("source", ""),
            "page":   r.get("page", 0),
            "score":  score,
            "query":  query,
        })

    return normalized

if __name__ == "__main__":
    print(f"Chunks indexed: {get_document_count()}")
    print("\nTest hybrid search:")
    results = hybrid_search(
        "retrieval augmented generation", top_k=3)
    for r in results:
        print(f"  Source: {r.get('source')}")
        print(f"  Text:   {r.get('text','')[:80]}...")