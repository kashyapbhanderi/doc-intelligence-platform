"""
api/routers/ingest.py
Document ingestion endpoints.
"""
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('.'))

from fastapi import (
    APIRouter, UploadFile, File,
    BackgroundTasks, HTTPException
)
from api.models import IngestResponse

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_document_background(
    file_path: str
):
    """
    Background task — processes PDF and
    ingests into Weaviate.
    Runs async so API returns immediately.
    """
    try:
        from ingestion.merger import (
            process_pdf_with_vision,
            save_document
        )
        from embeddings.reembed import (
            reembed_all_documents
        )

        print(f"Processing: {file_path}")
        doc = process_pdf_with_vision(
            file_path,
            use_vision=False
        )
        save_document(doc, "data/processed")
        print(f"Done: {doc['total_chunks']} chunks")

    except Exception as e:
        print(f"Background processing failed: {e}")


@router.post("/ingest",
             response_model=IngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF document for ingestion.
    Processing runs in the background — API
    returns immediately with job status.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files supported"
        )

    # Save uploaded file
    save_path = f"{UPLOAD_DIR}/{file.filename}"
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {e}"
        )

    # Queue background processing
    background_tasks.add_task(
        process_document_background,
        save_path
    )

    return IngestResponse(
        filename=file.filename,
        status="processing",
        chunks=0,
        message=(
            f"File uploaded successfully. "
            f"Processing in background. "
            f"File saved to {save_path}"
        )
    )


@router.get("/ingest/status")
async def ingest_status():
    """Check how many documents are ingested."""
    try:
        import weaviate
        client = weaviate.Client(
            "http://localhost:8080")
        result = (
            client.query
            .aggregate("Document")
            .with_meta_count()
            .do()
        )
        count = (
            result["data"]["Aggregate"]
            ["Document"][0]["meta"]["count"]
        )
        return {
            "total_chunks": count,
            "status": "ready"
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "status": f"error: {str(e)[:50]}"
        }