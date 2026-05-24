import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import MetadataQuery

# ── Configuration ──────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
CLASS_NAME = "Document"
VECTOR_DIM = 384

_embedder = None


class DocumentEmbedder:
    """
    Handles embedding documents and storing/searching
    in Weaviate v4.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_path: str = None
    ):

        ft_default = "models/finetuned/final"

        # Load fine-tuned model if available
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

        # Connect to Weaviate v4
        print("Connecting to Weaviate...")

        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )

        try:
            self.client.is_ready()
            print("Weaviate connected successfully!")

        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Weaviate.\n"
                f"Make sure Docker is running.\n"
                f"Error: {e}"
            )

    # ── Schema ────────────────────────────────────────────────

    def create_schema(self, force_recreate: bool = False):

        existing = self.client.collections.list_all()

        if CLASS_NAME in existing:

            if force_recreate:
                print(f"Deleting existing collection: {CLASS_NAME}")
                self.client.collections.delete(CLASS_NAME)

            else:
                print(f"Collection '{CLASS_NAME}' already exists.")
                return

        print(f"Creating collection '{CLASS_NAME}'...")

        self.client.collections.create(
            name=CLASS_NAME,

            vectorizer_config=None,

            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="source",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="page",
                    data_type=DataType.INT
                ),

                Property(
                    name="chunkId",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="chunkType",
                    data_type=DataType.TEXT
                ),

                Property(
                    name="charCount",
                    data_type=DataType.INT
                ),
            ]
        )

        print("Collection created successfully!")

    # ── Embedding ─────────────────────────────────────────────

    def embed_text(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list) -> list:

        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        )

        return [v.tolist() for v in vectors]

    # ── Insert ────────────────────────────────────────────────

    def insert_chunk(self, chunk: dict) -> bool:

        try:
            vector = self.embed_text(chunk["text"])

            collection = self.client.collections.get(CLASS_NAME)

            collection.data.insert(
                properties={
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", ""),
                    "page": chunk.get("page", 0),
                    "chunkId": str(chunk.get("chunk_id", "")),
                    "chunkType": chunk.get("chunk_type", "text"),
                    "charCount": chunk.get("char_count", 0),
                },
                vector=vector
            )

            return True

        except Exception as e:
            print(f"Insert error: {e}")
            return False

    # ── Count ─────────────────────────────────────────────────

    def get_document_count(self) -> int:

        try:
            collection = self.client.collections.get(CLASS_NAME)

            result = collection.aggregate.over_all(
                total_count=True
            )

            return result.total_count or 0

        except Exception:
            return 0

    # ── Vector Search ─────────────────────────────────────────

    def search_vector(
        self,
        query: str,
        top_k: int = 5
    ) -> list:

        query_vector = self.embed_text(query)

        try:
            collection = self.client.collections.get(CLASS_NAME)

            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(
                    distance=True
                )
            )

            results = []

            for obj in response.objects:

                props = obj.properties
                distance = obj.metadata.distance or 1.0

                results.append({
                    "text": props.get("text", ""),
                    "source": props.get("source", ""),
                    "page": props.get("page", 0),
                    "chunkId": props.get("chunkId", ""),
                    "_additional": {
                        "distance": distance,
                        "certainty": 1.0 - distance
                    }
                })

            return results

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    # ── BM25 Search ───────────────────────────────────────────

    def search_bm25(
        self,
        query: str,
        top_k: int = 5
    ) -> list:

        try:
            collection = self.client.collections.get(CLASS_NAME)

            response = collection.query.bm25(
                query=query,
                limit=top_k,
                return_metadata=MetadataQuery(score=True)
            )

            results = []

            for obj in response.objects:

                props = obj.properties
                score = obj.metadata.score

                results.append({
                    "text": props.get("text", ""),
                    "source": props.get("source", ""),
                    "page": props.get("page", 0),
                    "chunkId": props.get("chunkId", ""),
                    "_additional": {
                        "score": score
                    }
                })

            return results

        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

    # ── Hybrid Search ─────────────────────────────────────────

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5
    ) -> list:

        try:
            collection = self.client.collections.get(CLASS_NAME)

            # MANUALLY EMBED QUERY
            query_vector = self.embed_text(query)

            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=0.5,
                limit=top_k,
                return_metadata=MetadataQuery(
                    score=True,
                    distance=True
                )
            )

            results = []

            for obj in response.objects:

                props = obj.properties

                results.append({
                    "text": props.get("text", ""),
                    "source": props.get("source", ""),
                    "page": props.get("page", 0),
                    "chunkId": props.get("chunkId", ""),
                    "_additional": {
                        "score": obj.metadata.score,
                        "distance": obj.metadata.distance
                    }
                })

            return results

        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []


# ── Quick Test ──────────────────────────────────────────────

if __name__ == "__main__":

    embedder = DocumentEmbedder()

    embedder.create_schema()

    count = embedder.get_document_count()

    print(f"\nDocuments in Weaviate: {count}")

    if count > 0:

        print("\nTest search:")

        results = embedder.search_hybrid(
            "retrieval augmented generation",
            top_k=3
        )

        for r in results:

            print(f"\nSource: {r.get('source')}")
            print(f"Text: {r.get('text', '')[:80]}...")

    else:
        print("\nNo documents found.")
        print("Run embeddings/ingest.py")

    embedder.close()

    print("\nEmbedder ready!")