"""
api/routers/ingest.py
Document ingestion endpoints with task tracking.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from fastapi import (
    APIRouter, UploadFile, File,
    BackgroundTasks, HTTPException
)
from api.task_queue import (
    create_task, get_task,
    update_task, list_tasks, TaskStatus
)

router    = APIRouter()
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_document_background(
    file_path: str,
    task_id:   str
):
    """
    Background task — processes PDF and
    ingests into Weaviate.
    Updates task status throughout.
    """
    try:
        update_task(task_id, TaskStatus.PROCESSING)

        from ingestion.merger import (
            process_pdf_with_vision,
            save_document
        )

        print(f"[Task {task_id}] Processing: "
              f"{file_path}")

        # Process document
        doc = process_pdf_with_vision(
            file_path,
            use_vision=False
        )
        save_document(doc, "data/processed")

        # Embed and ingest
        from embeddings.embedder import (
            DocumentEmbedder
        )
        import json

        embedder = DocumentEmbedder(
            model_path="models/finetuned/final"
        )

        # Load chunks
        stem   = os.path.splitext(
            os.path.basename(file_path))[0]
        chunks_path = (
            f"data/processed/{stem}_processed.json"
        )

        if os.path.exists(chunks_path):
            with open(chunks_path,
                      encoding='utf-8') as f:
                doc_data = json.load(f)
            chunks = doc_data.get("chunks", [])

            # Embed and insert
            with embedder.client.batch as batch:
                batch.batch_size = 50
                for chunk in chunks:
                    vector = embedder.embed_text(
                        chunk["text"])
                    batch.add_data_object(
                        data_object={
                            "text":      chunk["text"],
                            "source":    chunk["source"],
                            "page":      chunk["page"],
                            "chunkId":   str(chunk.get(
                                "chunk_id", "")),
                            "chunkType": chunk.get(
                                "chunk_type", "text"),
                            "charCount": chunk.get(
                                "char_count", 0),
                        },
                        class_name="Document",
                        vector=vector
                    )

            update_task(
                task_id,
                TaskStatus.DONE,
                chunks=len(chunks)
            )
            print(f"[Task {task_id}] Done: "
                  f"{len(chunks)} chunks")
        else:
            update_task(
                task_id,
                TaskStatus.DONE,
                chunks=0
            )

    except Exception as e:
        print(f"[Task {task_id}] Failed: {e}")
        update_task(
            task_id,
            TaskStatus.FAILED,
            error=str(e)[:200]
        )


@router.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF for ingestion.
    Returns task_id to track progress.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files supported"
        )

    # Save file
    save_path = f"{UPLOAD_DIR}/{file.filename}"
    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {e}"
        )

    # Create tracking task
    task_id = create_task(file.filename)

    # Queue background processing
    background_tasks.add_task(
        process_document_background,
        save_path,
        task_id
    )

    return {
        "filename": file.filename,
        "task_id":  task_id,
        "status":   "pending",
        "message":  (
            f"Upload successful. "
            f"Track progress: "
            f"GET /api/v1/ingest/tasks/{task_id}"
        )
    }


@router.get("/ingest/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific ingestion task."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    return task


@router.get("/ingest/tasks")
async def list_all_tasks():
    """List all ingestion tasks."""
    return {
        "tasks": list_tasks(),
        "total": len(list_tasks())
    }


@router.get("/ingest/status")
async def ingest_status():
    """Check total chunks in Weaviate."""
    try:
        import weaviate
        client = weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
            skip_init_checks=True
        )
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
            "status":       "ready"
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "status":       f"error: {str(e)[:50]}"
        }