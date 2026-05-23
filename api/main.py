"""
api/main.py
FastAPI application entry point.

Registers all routers and middleware.
Handles startup/shutdown events.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from api.routers import query, ingest, edit

# ── App creation ──────────────────────────────────────────
app = FastAPI(
    title="Doc Intelligence Platform",
    description=(
        "Production-grade Multimodal Agentic RAG System "
        "with Document Editing Agent"
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc",    # ReDoc at /redoc
)

# ── CORS middleware ───────────────────────────────────────
# Allows browser frontends to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ───────────────────────────────────────
app.include_router(
    query.router,
    prefix="/api/v1",
    tags=["Query"]
)
app.include_router(
    ingest.router,
    prefix="/api/v1",
    tags=["Ingest"]
)
app.include_router(
    edit.router,
    prefix="/api/v1",
    tags=["Edit"]
)


# ── Root endpoint ─────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "Doc Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# ── Health endpoint ───────────────────────────────────────
@app.get("/api/v1/health",
         response_model=dict)
async def health():
    """
    Check system health — Weaviate, chunk count.
    Used by Docker health checks and monitoring.
    """
    weaviate_status = "unknown"
    total_chunks = 0

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
        total_chunks = (
            result["data"]["Aggregate"]
            ["Document"][0]["meta"]["count"]
        )
        weaviate_status = "healthy"
    except Exception as e:
        weaviate_status = f"error: {str(e)[:50]}"

    return {
        "status": "healthy",
        "version": "1.0.0",
        "weaviate": weaviate_status,
        "total_chunks": total_chunks
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )