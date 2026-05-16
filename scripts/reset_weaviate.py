import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from embeddings.embedder import DocumentEmbedder

embedder = DocumentEmbedder()

# Delete existing schema and all data
print("Deleting all data from Weaviate...")
try:
    embedder.client.schema.delete_class("Document")
    print("Deleted Document class successfully")
except Exception as e:
    print(f"Nothing to delete: {e}")

# Recreate fresh schema
print("Creating fresh schema...")
embedder.create_schema(force_recreate=False)

count = embedder.get_document_count()
print(f"Weaviate reset complete!")
print(f"Current chunks: {count}")
print("Ready for fresh ingestion.")
