"""
embeddings/reembed.py

Re-embed all documents using fine-tuned model.
Clears old Weaviate data and re-ingests fresh.
"""
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))


def reembed_all_documents(
    processed_dir: str = "data/processed",
    model_path:    str = "models/finetuned/final",
    weaviate_url:  str = None,
    batch_size:    int = 50
):
    from sentence_transformers import SentenceTransformer
    import weaviate

    weaviate_url = weaviate_url or os.getenv(
        "WEAVIATE_URL", "http://localhost:8080")

    # Step 1 — Load model
    if not os.path.exists(model_path):
        print(f"Fine-tuned model not found: {model_path}")
        print("Using base model instead...")
        model_path = "all-MiniLM-L6-v2"
    else:
        print(f"Fine-tuned model found: {model_path}")

    print(f"\nLoading model: {model_path}")
    model = SentenceTransformer(model_path)
    dim   = model.get_sentence_embedding_dimension()
    print(f"Vector dimensions: {dim}")

    # Step 2 — Connect to Weaviate (v3 client)
    print(f"\nConnecting to Weaviate: {weaviate_url}")
    client = weaviate.Client(weaviate_url)

    # Step 3 — Delete old schema + data
    print("\nClearing old embeddings...")
    try:
        client.schema.delete_class("Document")
        print("  Deleted old Document class")
    except Exception as e:
        print(f"  Nothing to delete: {e}")

    # Step 4 — Recreate schema
    print("Creating fresh schema...")
    class_obj = {
        "class":       "Document",
        "description": "Re-embedded with fine-tuned model",
        "vectorizer":  "none",
        "vectorIndexConfig": {"distance": "cosine"},
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
    client.schema.create_class(class_obj)
    print("Schema created!")

    # Step 5 — Load all processed documents
    print("\nLoading processed documents...")
    json_files = list(
        Path(processed_dir).glob("*_processed.json"))
    print(f"Found {len(json_files)} documents")

    # Step 6 — Re-embed and insert
    print("\nRe-embedding with fine-tuned model...")
    print("This takes 5-15 minutes. Please wait.\n")

    total_inserted = 0
    total_errors   = 0
    start_time     = time.time()

    for json_file in tqdm(json_files, desc="Re-embedding docs"):
        try:
            with open(json_file, encoding='utf-8') as f:
                doc = json.load(f)

            chunks = doc.get("chunks", [])
            if not chunks:
                continue

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c.get("text", "") for c in batch]

                vectors = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False
                )

                with client.batch as wb:
                    wb.batch_size = batch_size
                    for chunk, vector in zip(batch, vectors):
                        props = {
                            "text":      chunk.get("text", ""),
                            "source":    chunk.get("source", ""),
                            "page":      chunk.get("page", 0),
                            "chunkId":   str(chunk.get(
                                "chunk_id", "")),
                            "chunkType": chunk.get(
                                "chunk_type", "text"),
                            "charCount": chunk.get(
                                "char_count", 0),
                        }
                        wb.add_data_object(
                            data_object=props,
                            class_name="Document",
                            vector=vector.tolist()
                        )
                        total_inserted += 1

        except Exception as e:
            print(f"\nError: {json_file.name}: {e}")
            total_errors += 1

    elapsed = time.time() - start_time

    # Final count
    result = (
        client.query
        .aggregate("Document")
        .with_meta_count()
        .do()
    )
    final_count = (
        result["data"]["Aggregate"]["Document"]
        [0]["meta"]["count"]
    )

    print("\n" + "=" * 60)
    print("RE-EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {len(json_files)}")
    print(f"Chunks inserted:     {total_inserted}")
    print(f"Errors:              {total_errors}")
    print(f"Total in Weaviate:   {final_count}")
    print(f"Time taken:          {elapsed/60:.1f} min")
    print(f"Model used:          {model_path}")

    return final_count


if __name__ == "__main__":
    count = reembed_all_documents()
    print(f"\nFinal chunk count: {count}")
    print("Ready for NDCG evaluation!")