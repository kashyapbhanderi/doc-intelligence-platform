"""
embeddings/embedder.py

DocumentEmbedder class — handles:
- Loading the embedding model (base or fine-tuned)
- Connecting to Weaviate v3
- Creating the Document schema
- Embedding text → vectors
- Vector search, BM25 search, Hybrid RRF search
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from sentence_transformers import SentenceTransformer
import weaviate

# ── Configuration ──────────────────────────────────────────
MODEL_NAME   = "all-MiniLM-L6-v2"
WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME   = "Document"
VECTOR_DIM   = 384


class DocumentEmbedder:
    """
    Handles embedding documents and storing/searching
    in Weaviate v3 client.
    """

    def __init__(
        self,
        model_name:  str = MODEL_NAME,
        weaviate_url: str = WEAVIATE_URL,
        model_path:  str = None
    ):
        ft_default = "models/finetuned/final"

        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model: {model_path}")
            self.model = SentenceTransformer(model_path)
            self.model_name = model_path

        elif os.path.exists(ft_default):
            print(f"Loading fine-tuned model: {ft_default}")
            self.model = SentenceTransformer(ft_default)
            self.model_name = ft_default

        else:
            print(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name

        print(f"Connecting to Weaviate at {weaviate_url}")
        self.client = weaviate.Client(weaviate_url)

        try:
            self.client.schema.get()
            print("Weaviate connected successfully!")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Weaviate at {weaviate_url}.\n"
                f"Make sure Docker is running:\n"
                f"  docker-compose up -d weaviate\n"
                f"Error: {e}"
            )

    # ── Schema ────────────────────────────────────────────────

    def create_schema(self, force_recreate: bool = False):
        existing = self.client.schema.get()
        classes  = [
            c["class"]
            for c in existing.get("classes", [])
        ]

        if CLASS_NAME in classes:
            if force_recreate:
                print(f"Deleting existing {CLASS_NAME} schema...")
                self.client.schema.delete_class(CLASS_NAME)
            else:
                print(f"Schema '{CLASS_NAME}' already exists.")
                return

        print(f"Creating schema for '{CLASS_NAME}'...")

        class_obj = {
            "class":       CLASS_NAME,
            "description": "A chunk of text from a document",
            "vectorizer":  "none",
            "vectorIndexConfig": {
                "distance": "cosine"
            },
            "invertedIndexConfig": {
                "bm25": {"b": 0.75, "k1": 1.2}
            },
            "properties": [
                {
                    "name":     "text",
                    "dataType": ["text"],
                    "invertedIndexConfig": {
                        "indexSearchable": True
                    }
                },
                {"name": "source",    "dataType": ["text"]},
                {"name": "page",      "dataType": ["int"]},
                {"name": "chunkId",   "dataType": ["text"]},
                {"name": "chunkType", "dataType": ["text"]},
                {"name": "charCount", "dataType": ["int"]},
            ]
        }

        self.client.schema.create_class(class_obj)
        print("Schema created successfully!")

    # ── Embedding ─────────────────────────────────────────────

    def embed_text(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list) -> list:
        vectors = self.model.encode(
            texts, batch_size=32, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    # ── Insert ────────────────────────────────────────────────

    def insert_chunk(self, chunk: dict) -> bool:
        try:
            vector = self.embed_text(chunk["text"])
            properties = {
                "text":      chunk.get("text", ""),
                "source":    chunk.get("source", ""),
                "page":      chunk.get("page", 0),
                "chunkId":   str(chunk.get("chunk_id", "")),
                "chunkType": chunk.get("chunk_type", "text"),
                "charCount": chunk.get("char_count", 0),
            }
            self.client.data_object.create(
                data_object=properties,
                class_name=CLASS_NAME,
                vector=vector
            )
            return True
        except Exception as e:
            print(f"  Error inserting chunk: {e}")
            return False

    # ── Count ─────────────────────────────────────────────────

    def get_document_count(self) -> int:
        try:
            result = (
                self.client.query
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

    # ── Vector search ─────────────────────────────────────────

    def search_vector(self, query: str, top_k: int = 5) -> list:
        query_vector = self.embed_text(query)
        try:
            result = (
                self.client.query
                .get(CLASS_NAME,
                     ["text", "source", "page", "chunkId"])
                .with_near_vector({"vector": query_vector})
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

    # ── BM25 search ───────────────────────────────────────────

    def search_bm25(self, query: str, top_k: int = 5) -> list:
        try:
            result = (
                self.client.query
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

    # ── Hybrid RRF search ─────────────────────────────────────

    def search_hybrid(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> list:
        try:
            fetch_k = top_k * 3
            bm25_results   = self.search_bm25(query, top_k=fetch_k) or []
            vector_results = self.search_vector(query, top_k=fetch_k) or []

            scores    = {}
            chunk_map = {}

            for rank, result in enumerate(bm25_results):
                key = result.get("text", "")[:100]
                if key not in scores:
                    scores[key]    = 0
                    chunk_map[key] = result
                scores[key] += 1 / (rank + 60)

            for rank, result in enumerate(vector_results):
                key = result.get("text", "")[:100]
                if key not in scores:
                    scores[key]    = 0
                    chunk_map[key] = result
                scores[key] += 1 / (rank + 60)

            sorted_keys = sorted(
                scores.keys(), key=lambda k: scores[k], reverse=True
            )
            return [chunk_map[k] for k in sorted_keys[:top_k]]

        except Exception as e:
            print(f"  Hybrid search error: {e}")
            return []


if __name__ == "__main__":
    embedder = DocumentEmbedder()
    embedder.create_schema()
    count = embedder.get_document_count()
    print(f"\nDocuments in Weaviate: {count}")
    print("\nEmbedder ready!")