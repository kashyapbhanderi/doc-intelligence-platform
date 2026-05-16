import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))

from embeddings.embedder import DocumentEmbedder


def load_chunks_from_file(json_path: str) -> list:
    """Load chunks from a processed JSON file."""
    with open(json_path, encoding='utf-8') as f:
        doc = json.load(f)
    return doc.get("chunks", [])


def ingest_documents(
    processed_dir: str = "data/processed",
    limit: int = None,
    batch_size: int = 50
):
    """
    Load all processed documents and insert into Weaviate.

    Args:
        processed_dir: Folder with *_processed.json files
        limit: Max documents to process (None = all)
        batch_size: How many chunks to embed at once
    """
    # Get all processed files
    json_files = list(
        Path(processed_dir).glob("*_processed.json")
    )

    if limit:
        json_files = json_files[:limit]

    print(f"Found {len(json_files)} processed documents")

    # Setup embedder
    embedder = DocumentEmbedder()
    embedder.create_schema()

    # Count what's already in Weaviate
    existing = embedder.get_document_count()
    print(f"Existing chunks in Weaviate: {existing}")

    total_inserted = 0
    total_skipped = 0
    total_errors = 0
    start_time = time.time()

    for json_file in tqdm(json_files, desc="Ingesting docs"):
        try:
            chunks = load_chunks_from_file(str(json_file))

            if not chunks:
                continue

            # Embed in batches for speed
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c["text"] for c in batch]

                # Embed all texts at once
                vectors = embedder.embed_batch(texts)

                # Insert each chunk with its vector
                with embedder.client.batch as weaviate_batch:
                    weaviate_batch.batch_size = batch_size

                    for chunk, vector in zip(batch, vectors):
                        properties = {
                            "text": chunk.get("text", ""),
                            "source": chunk.get("source", ""),
                            "page": chunk.get("page", 0),
                            "chunkId": str(
                                chunk.get("chunk_id", "")
                            ),
                            "chunkType": chunk.get(
                                "chunk_type", "text"
                            ),
                            "charCount": chunk.get(
                                "char_count", 0
                            )
                        }

                        weaviate_batch.add_data_object(
                            data_object=properties,
                            class_name="Document",
                            vector=vector
                        )
                        total_inserted += 1

        except Exception as e:
            print(f"\nError processing {json_file.name}: {e}")
            total_errors += 1

    elapsed = time.time() - start_time
    final_count = embedder.get_document_count()

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"Documents processed: {len(json_files)}")
    print(f"Chunks inserted:     {total_inserted}")
    print(f"Errors:              {total_errors}")
    print(f"Total in Weaviate:   {final_count}")
    print(f"Time taken:          {elapsed:.1f}s")
    print(f"Speed:               "
          f"{total_inserted / max(elapsed, 1):.0f} chunks/sec")

    return final_count


if __name__ == "__main__":
    # Start with first 10 documents to verify
    print("Testing with first 10 documents...")
    count = ingest_documents(
        processed_dir="data/processed",
        limit=10
    )
    print(f"\nFinal chunk count in Weaviate: {count}")