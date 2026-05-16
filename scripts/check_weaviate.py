import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from embeddings.embedder import DocumentEmbedder

embedder = DocumentEmbedder()
count = embedder.get_document_count()

print(f"Current chunks in Weaviate: {count}")
print(f"Expected total: ~11089")

if count < 1000:
    print("Need to ingest more documents — run ingest_all.py")
elif count < 8000:
    print("Partially ingested — run ingest_all.py to complete")
else:
    print("All documents ingested!")