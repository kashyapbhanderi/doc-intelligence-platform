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
from prometheus_fastapi_instrumentator import Instrumentator
from api.metrics import WEAVIATE_CHUNKS
import threading
import time as _time

# from memory.memory_agent import MemoryEnabledAgent
# from agents.graph import ask 
# memory_agent = MemoryEnabledAgent(ask) 

load_dotenv()

from api.routers import query, ingest, edit
# ── Prometheus instrumentation ────────────────────────────
# Auto-instruments every endpoint with request count
# and latency histograms.
# Metrics exposed at GET /metrics

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
Instrumentator().instrument(app).expose(app)


# ── Background chunk counter ──────────────────────────────
def update_weaviate_gauge():
    """
    Updates Weaviate chunk count gauge every 60s.
    Prometheus scrapes /metrics — this keeps the
    gauge current without blocking requests.
    """
    weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
    while True:
        try:
            import weaviate
            client = weaviate.connect_to_local(
                host=weaviate_host,
                port=8080,
                grpc_port=50051,
                skip_init_checks=True
            )
            collection = client.collections.get("Document")
            result = collection.aggregate.over_all(total_count=True)
            count = result.total_count or 0
            WEAVIATE_CHUNKS.set(count)
            client.close()
        except Exception:
            pass
        _time.sleep(60)


# Start background thread
threading.Thread(
    target=update_weaviate_gauge,
    daemon=True
).start()


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

# ── GraphRAG + Memory wiring ───────────────────────────────
from agents.graph import build_agent_graph   # your existing LangGraph builder
from memory.memory_agent import MemoryEnabledAgent
from knowledge_graph.graph_builder import GraphBuilder
from knowledge_graph.hybrid_graphrag import HybridGraphRAG
from embeddings.search import hybrid_search
from api.memory_routes import (
    memory_router, graphrag_router,
    set_memory_agent, set_graphrag
)

# Build the underlying agent graph (same one your /query uses)
_agent_graph = build_agent_graph()
memory_agent = MemoryEnabledAgent(_agent_graph)

# Build/load the knowledge graph
_gb = GraphBuilder()
_gb.load()
_graphrag = HybridGraphRAG(_gb, hybrid_search, graph_weight=0.4)

# Register routes
app.include_router(memory_router,   prefix="/memory",   tags=["Long-Term Memory"])
app.include_router(graphrag_router, prefix="/graphrag",  tags=["GraphRAG"])

# Share instances with the routes
set_memory_agent(memory_agent)
set_graphrag(_graphrag)


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
    global _health_cache
    if _time.time() < _health_cache["expires"]:
        return _health_cache["data"]

    weaviate_status = "unknown"
    total_chunks    = 0

    try:
        import weaviate
        weaviate_url = os.getenv(
            "WEAVIATE_URL", "http://localhost:8080")
        client = weaviate.Client(weaviate_url)

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
    _health_cache = {"data": data, "expires": _time.time() + 30}
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )