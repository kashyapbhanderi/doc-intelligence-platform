"""
api/memory_routes.py
=====================
New FastAPI routes for the Long-term Memory and GraphRAG features.

HOW TO ADD TO YOUR EXISTING api/main.py:
──────────────────────────────────────────
  # At the top of main.py, add:
  from api.memory_routes import memory_router, graphrag_router

  # After creating your `app = FastAPI(...)`, add:
  app.include_router(memory_router,  prefix="/memory",  tags=["Long-Term Memory"])
  app.include_router(graphrag_router, prefix="/graphrag", tags=["GraphRAG"])
──────────────────────────────────────────

WHERE TO PLACE THIS FILE: api/memory_routes.py
"""

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ── Your existing imports (adjust paths if needed) ────────────────
from memory.memory_agent import MemoryEnabledAgent
from knowledge_graph.graph_builder import GraphBuilder
from knowledge_graph.hybrid_graphrag import HybridGraphRAG


# ═══════════════════════════════════════════════════════════════════
# SHARED STATE
# These are initialised at startup (in main.py lifespan or on first use).
# ═══════════════════════════════════════════════════════════════════

# Replace `None` with your actual instances after initialisation in main.py:
#   from api.memory_routes import set_memory_agent, set_graphrag
#   set_memory_agent(MemoryEnabledAgent(agent_graph))
#   set_graphrag(HybridGraphRAG(gb, hybrid_search))

_memory_agent: Optional[MemoryEnabledAgent] = None
_graphrag:     Optional[HybridGraphRAG]     = None


def set_memory_agent(agent: MemoryEnabledAgent) -> None:
    global _memory_agent
    _memory_agent = agent


def set_graphrag(hg: HybridGraphRAG) -> None:
    global _graphrag
    _graphrag = hg


def _require_memory() -> MemoryEnabledAgent:
    if _memory_agent is None:
        raise HTTPException(503, "Memory agent not initialised.")
    return _memory_agent


def _require_graphrag() -> HybridGraphRAG:
    if _graphrag is None:
        raise HTTPException(503, "GraphRAG not initialised.")
    return _graphrag


# ═══════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════

class QueryWithMemoryRequest(BaseModel):
    question: str
    user_id:  str  = "default"
    session_id: Optional[str] = None


class EndSessionRequest(BaseModel):
    user_id: str = "default"


class GraphRAGQueryRequest(BaseModel):
    query:  str
    top_k:  int = 5
    hops:   int = 2    # traversal depth: 1 = direct, 2 = friends-of-friends


class MemoriesResponse(BaseModel):
    user_id:       str
    semantic_facts: list[dict]
    recent_episodes: list[dict]
    stats:          dict


# ═══════════════════════════════════════════════════════════════════
# MEMORY ROUTES
# ═══════════════════════════════════════════════════════════════════

memory_router = APIRouter()


@memory_router.post("/query")
async def query_with_memory(req: QueryWithMemoryRequest):
    """
    Answer a question using the memory-enabled agent.
    Automatically injects long-term memory context from past sessions.

    Example:
        POST /memory/query
        {"question": "What is the EPS?", "user_id": "user_123"}
    """
    agent = _require_memory()
    result = agent.invoke(
        req.question,
        user_id=req.user_id,
        session_id=req.session_id,
    )
    return {
        "answer":               result.get("answer", ""),
        "memory_context_used":  result.get("memory_context_used", False),
        "user_id":              req.user_id,
    }


@memory_router.post("/session/end")
async def end_session(req: EndSessionRequest):
    """
    End a user session and extract + store memories from the conversation.
    Call this when the user logs out or when the session times out.

    Example:
        POST /memory/session/end
        {"user_id": "user_123"}
    """
    agent = _require_memory()
    counts = agent.end_session(req.user_id)
    return {
        "user_id":           req.user_id,
        "episodic_stored":   counts["episodic_stored"],
        "semantic_stored":   counts["semantic_stored"],
        "message":           f"Session ended. Stored {counts['episodic_stored']} episode and {counts['semantic_stored']} facts.",
    }


@memory_router.get("/user/{user_id}")
async def get_user_memories(user_id: str):
    """
    Get all stored memories for a user.
    Useful for a "What does the AI know about me?" page in your frontend.

    Example:
        GET /memory/user/user_123
    """
    agent = _require_memory()
    memories = agent.get_user_memories(user_id)
    return MemoriesResponse(
        user_id=user_id,
        semantic_facts=memories["semantic_facts"],
        recent_episodes=memories["recent_episodes"],
        stats=memories["stats"],
    )


@memory_router.delete("/user/{user_id}")
async def clear_user_memories(user_id: str):
    """
    Delete all memories for a user (GDPR / reset).

    Example:
        DELETE /memory/user/user_123
    """
    agent = _require_memory()
    agent.clear_user_memories(user_id)
    return {"user_id": user_id, "message": "All memories cleared."}


# ═══════════════════════════════════════════════════════════════════
# GRAPHRAG ROUTES
# ═══════════════════════════════════════════════════════════════════

graphrag_router = APIRouter()


@graphrag_router.post("/retrieve")
async def graphrag_retrieve(req: GraphRAGQueryRequest):
    """
    Retrieve document chunks using GraphRAG (entity graph + vector fusion).
    Returns ranked chunks with RRF scores and retrieval method labels.

    Example:
        POST /graphrag/retrieve
        {"query": "What did the CEO say about acquisitions?", "top_k": 5}
    """
    hg = _require_graphrag()
    results = hg.retrieve(req.query, top_k=req.top_k)
    return {
        "query":    req.query,
        "results":  results,
        "count":    len(results),
    }


@graphrag_router.post("/explain")
async def graphrag_explain(req: GraphRAGQueryRequest):
    """
    Debug endpoint: shows exactly what vector search and GraphRAG each found,
    and how they were fused using Reciprocal Rank Fusion.

    Use this during development to verify GraphRAG is contributing value.

    Example:
        POST /graphrag/explain
        {"query": "What is the revenue?"}
    """
    hg = _require_graphrag()
    explanation = hg.explain_retrieval(req.query, top_k=req.top_k)
    return explanation


@graphrag_router.get("/entity/{entity_name}")
async def get_entity_neighbors(entity_name: str):
    """
    Inspect a named entity's connections in the knowledge graph.
    Great for demos — shows the graph is working correctly.

    Example:
        GET /graphrag/entity/Infosys
    """
    hg = _require_graphrag()
    neighbors = hg.graph_retriever.get_entity_neighbors(entity_name)
    if not neighbors.get("found"):
        raise HTTPException(404, f"Entity '{entity_name}' not found in knowledge graph.")
    return neighbors


@graphrag_router.get("/stats")
async def get_graph_stats():
    """
    Return knowledge graph statistics.
    Shows in your Grafana dashboard or project README.

    Example:
        GET /graphrag/stats
    """
    hg = _require_graphrag()
    return hg.graph_retriever.gb.get_stats()
