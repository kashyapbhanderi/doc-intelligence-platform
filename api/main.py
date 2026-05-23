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
import uuid 
from fastapi import Request

@app.middleware("http")
async def add_request_id(
    request: Request,
    call_next
):
    """
    Adds X-Request-ID header to every response.
    Essential for debugging in production —
    you can trace a specific request through logs.
    """
    request_id = str(uuid.uuid4())[:8]
    response   = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

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
import time as _time

# Simple cache for health check
_health_cache = {"data": None, "expires": 0}


@app.get("/api/v1/health")
async def health():
    """
    Check system health.
    Cached for 30 seconds — prevents hammering
    Weaviate on every health check.
    """
    global _health_cache

    # Return cached if still fresh
    if _time.time() < _health_cache["expires"]:
        return _health_cache["data"]

    weaviate_status = "unknown"
    total_chunks    = 0

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
        total_chunks = (
            result["data"]["Aggregate"]
            ["Document"][0]["meta"]["count"]
        )
        weaviate_status = "healthy"
    except Exception as e:
        weaviate_status = f"error: {str(e)[:50]}"

    data = {
        "status":       "healthy",
        "version":      "1.0.0",
        "weaviate":     weaviate_status,
        "total_chunks": total_chunks
    }

    # Cache for 30 seconds
    _health_cache = {
        "data":    data,
        "expires": _time.time() + 30
    }

    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )