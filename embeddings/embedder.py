import os
import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import weaviate
import weaviate.classes 


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
             weaviate_url: str = WEAVIATE_URL,
             model_path: str = None):
        """
        model_path: if provided, loads fine-tuned model
                    from local path instead of HuggingFace
        """
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model: {model_path}")
            self.model = SentenceTransformer(model_path)
            self.model_name = model_path
        else:
            print(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name

        # Connect to Weaviate (always, regardless of model source)
        # Connect to Weaviate v4
        print(f"Connecting to Weaviate at {weaviate_url}")

        self.client = weaviate.connect_to_local(
            host="localhost",      # use "localhost" if running outside Docker
            port=8080,
            grpc_port=50051,
            skip_init_checks=True
        )

        # Test connection
        try:
            if self.client.is_ready():
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
        existing = self.client.collections.list_all()
        classes = list(existing.keys())

        if CLASS_NAME in classes:
            if force_recreate:
                print(f"Deleting existing {CLASS_NAME} schema...")
                self.client.collections.delete(CLASS_NAME)
            else:
                print(f"Schema '{CLASS_NAME}' already exists. Skipping.")
                return

        print(f"Creating schema for '{CLASS_NAME}'...")

        class_obj = {
            "class": CLASS_NAME,
            "description": "A chunk of text from a document",
            "vectorizer": "none",  # we provide our own vectors
            "vectorIndexConfig": {
                "distance": "cosine",
                "ef": 100,
                "maxConnections": 64  # cosine similarity for search
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

        self.client.collections.create(
            name=CLASS_NAME,
            properties=[
                weaviate.classes.config.Property(
                    name="text",
                    data_type=weaviate.classes.config.DataType.TEXT
                ),
                weaviate.classes.config.Property(
                    name="source",
                    data_type=weaviate.classes.config.DataType.TEXT
                ),
                weaviate.classes.config.Property(
                    name="page",
                    data_type=weaviate.classes.config.DataType.INT
                ),
                weaviate.classes.config.Property(
                    name="chunkId",
                    data_type=weaviate.classes.config.DataType.TEXT
                ),
                weaviate.classes.config.Property(
                    name="chunkType",
                    data_type=weaviate.classes.config.DataType.TEXT
                ),
                weaviate.classes.config.Property(
                    name="charCount",
                    data_type=weaviate.classes.config.DataType.INT
                ),
            ]
        )
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

        query_vector = self.embed_text(query)

        collection = self.client.collections.get(CLASS_NAME)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=["distance"]
        )

        results = []

        for obj in response.objects:
            results.append({
                "text": obj.properties.get("text", ""),
                "source": obj.properties.get("source", ""),
                "page": obj.properties.get("page", 0),
                "chunkId": obj.properties.get("chunkId", ""),
                "distance": getattr(obj.metadata, "distance", None)
            })

        return results

    def search_bm25(self, query: str, top_k: int = 5) -> list:

        collection = self.client.collections.get(CLASS_NAME)

        response = collection.query.bm25(
            query=query,
            limit=top_k
        )

        results = []

        for obj in response.objects:
            results.append({
                "text": obj.properties.get("text", ""),
                "source": obj.properties.get("source", ""),
                "page": obj.properties.get("page", 0),
                "chunkId": obj.properties.get("chunkId", "")
            })

        return results
    def search_hybrid(self, query: str, top_k: int = 5,
                  alpha: float = 0.5) -> list:
        """
        Hybrid search using manual RRF (Reciprocal Rank Fusion).

        Why manual RRF instead of Weaviate's built-in?
        More control, works with all Weaviate versions,
        and RRF is the industry standard fusion algorithm.

        RRF formula: score = sum(1 / (rank + 60))
        Higher score = more relevant result.

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Balance (unused in RRF but kept for API compat)

        Returns:
            List of top_k most relevant chunks
        """
        try:
            # Get more results from each method to fuse
            fetch_k = top_k * 3

            bm25_results = self.search_bm25(query,
                                            top_k=fetch_k) or []
            vector_results = self.search_vector(query,
                                                top_k=fetch_k) or []

            # Build score map using chunk text as unique key
            scores = {}
            chunk_map = {}

            # Score BM25 results
            for rank, result in enumerate(bm25_results):
                key = result.get("text", "")[:100]
                if key not in scores:
                    scores[key] = 0
                    chunk_map[key] = result
                # RRF formula
                scores[key] += 1 / (rank + 60)

            # Score vector results
            for rank, result in enumerate(vector_results):
                key = result.get("text", "")[:100]
                if key not in scores:
                    scores[key] = 0
                    chunk_map[key] = result
                # RRF formula
                scores[key] += 1 / (rank + 60)

            # Sort by combined score
            sorted_keys = sorted(
                scores.keys(),
                key=lambda k: scores[k],
                reverse=True
            )

            # Return top_k results
            return [chunk_map[k] for k in sorted_keys[:top_k]]

        except Exception as e:
            print(f"  Hybrid search error: {e}")
            return []

if __name__ == "__main__":
    # Quick test
    embedder = DocumentEmbedder()
    embedder.create_schema()

    count = embedder.get_document_count()
    print(f"\nDocuments in Weaviate: {count}")
    print("\nEmbedder ready!")
