import os
import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import weaviate


# ── Configuration ──────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "Document"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
# ───────────────────────────────────────────────────────────────


class DocumentEmbedder:
    """
    Handles embedding documents and storing them in Weaviate.

    Industry term: This is called a 'vector store' wrapper.
    It abstracts the embedding model + database into one class.
    """

    def __init__(self,
                 model_name: str = MODEL_NAME,
                 weaviate_url: str = WEAVIATE_URL):

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        print(f"Connecting to Weaviate at {weaviate_url}")
        self.client = weaviate.Client(weaviate_url)

        # Test connection
        try:
            self.client.schema.get()
            print("Weaviate connected successfully!")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Weaviate at {weaviate_url}.\n"
                f"Make sure Docker is running and run:\n"
                f"docker-compose up -d weaviate\n"
                f"Error: {e}"
            )

    def create_schema(self, force_recreate: bool = False):
        """
        Create the Document class schema in Weaviate.

        Schema defines what fields each document has.
        Like creating a table in SQL.

        Args:
            force_recreate: If True, deletes existing schema first
        """
        # Check if class already exists
        existing = self.client.schema.get()
        classes = [c["class"] for c in existing.get("classes", [])]

        if CLASS_NAME in classes:
            if force_recreate:
                print(f"Deleting existing {CLASS_NAME} schema...")
                self.client.schema.delete_class(CLASS_NAME)
            else:
                print(f"Schema '{CLASS_NAME}' already exists. Skipping.")
                return

        print(f"Creating schema for '{CLASS_NAME}'...")

        class_obj = {
            "class": CLASS_NAME,
            "description": "A chunk of text from a document",
            "vectorizer": "none",  # we provide our own vectors
            "vectorIndexConfig": {
                "distance": "cosine"  # cosine similarity for search
            },
            "invertedIndexConfig": {
                "bm25": {
                    "b": 0.75,   # BM25 parameter
                    "k1": 1.2    # BM25 parameter
                }
            },
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The chunk text content",
                    "invertedIndexConfig": {
                        "indexSearchable": True  # enables BM25
                    }
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                    "description": "Source PDF filename"
                },
                {
                    "name": "page",
                    "dataType": ["int"],
                    "description": "Page number in source PDF"
                },
                {
                    "name": "chunkId",
                    "dataType": ["text"],
                    "description": "Unique chunk identifier"
                },
                {
                    "name": "chunkType",
                    "dataType": ["text"],
                    "description": "text or vision"
                },
                {
                    "name": "charCount",
                    "dataType": ["int"],
                    "description": "Character count of chunk"
                }
            ]
        }

        self.client.schema.create_class(class_obj)
        print(f"Schema created successfully!")

    def embed_text(self, text: str) -> list:
        """
        Convert a text string into a vector embedding.

        Args:
            text: Any text string

        Returns:
            List of floats (the vector)
        """
        vector = self.model.encode(text)
        return vector.tolist()

    def embed_batch(self, texts: list) -> list:
        """
        Embed multiple texts at once (faster than one by one).

        Args:
            texts: List of text strings

        Returns:
            List of vectors
        """
        vectors = self.model.encode(texts, batch_size=32,
                                    show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def insert_chunk(self, chunk: dict) -> bool:
        """
        Insert a single chunk into Weaviate with its vector.

        Args:
            chunk: Dict with text, source, page, etc.

        Returns:
            True if successful
        """
        try:
            vector = self.embed_text(chunk["text"])

            properties = {
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "chunkId": str(chunk.get("chunk_id", "")),
                "chunkType": chunk.get("chunk_type", "text"),
                "charCount": chunk.get("char_count", 0)
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

    def get_document_count(self) -> int:
        """Get total number of documents in Weaviate."""
        try:
            result = (
                self.client.query
                .aggregate(CLASS_NAME)
                .with_meta_count()
                .do()
            )
            count = (result["data"]["Aggregate"][CLASS_NAME]
                     [0]["meta"]["count"])
            return count
        except Exception:
            return 0

    def search_vector(self, query: str, top_k: int = 5) -> list:
        """
        Semantic vector search — finds chunks similar in MEANING
        to the query, even if they don't share keywords.

        Args:
            query: Natural language question
            top_k: Number of results to return

        Returns:
            List of matching chunks with scores
        """
        query_vector = self.embed_text(query)

        result = (
            self.client.query
            .get(CLASS_NAME, ["text", "source", "page",
                              "chunkId"])
            .with_near_vector({"vector": query_vector})
            .with_limit(top_k)
            .with_additional(["certainty", "distance"])
            .do()
        )

        hits = result.get("data", {}).get("Get", {}).get(
            CLASS_NAME, [])
        return hits

    def search_bm25(self, query: str, top_k: int = 5) -> list:
        """
        BM25 keyword search — finds chunks that contain
        the exact words from the query.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matching chunks
        """
        result = (
            self.client.query
            .get(CLASS_NAME, ["text", "source", "page",
                              "chunkId"])
            .with_bm25(query=query, properties=["text"])
            .with_limit(top_k)
            .with_additional(["score"])
            .do()
        )

        hits = result.get("data", {}).get("Get", {}).get(
            CLASS_NAME, [])
        return hits

    def search_hybrid(self, query: str, top_k: int = 5,
                      alpha: float = 0.5) -> list:
        """
        Hybrid search — combines BM25 + vector search.

        alpha=0.0 → pure BM25 (keyword only)
        alpha=0.5 → balanced (recommended)
        alpha=1.0 → pure vector (semantic only)

        This is better than either alone because:
        - BM25 catches exact keyword matches
        - Vector search catches semantic meaning
        - Together they cover both cases

        Args:
            query: Search query
            top_k: Number of results
            alpha: Balance between BM25 and vector (0-1)

        Returns:
            List of matching chunks
        """
        result = (
            self.client.query
            .get(CLASS_NAME, ["text", "source", "page",
                              "chunkId"])
            .with_hybrid(query=query, alpha=alpha)
            .with_limit(top_k)
            .with_additional(["score"])
            .do()
        )

        hits = result.get("data", {}).get("Get", {}).get(
            CLASS_NAME, [])
        return hits


if __name__ == "__main__":
    # Quick test
    embedder = DocumentEmbedder()
    embedder.create_schema()

    count = embedder.get_document_count()
    print(f"\nDocuments in Weaviate: {count}")
    print("\nEmbedder ready!")