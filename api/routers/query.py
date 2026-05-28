"""
api/routers/query.py
Query endpoints — ask questions to the RAG pipeline.
"""
import os
import sys
import time
sys.path.insert(0, os.path.abspath('.'))

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.models import QueryRequest, QueryResponse

router = APIRouter()


import time as _time
from api.metrics import (
    QUERY_TOTAL,
    QUERY_LATENCY,
    ACTIVE_QUERIES,
    FAITHFUL_ANSWERS,
    UNFAITHFUL_ANSWERS,
)

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Ask a question — Planner → Executor → Critic.
    Records Prometheus metrics for every request.
    """
    ACTIVE_QUERIES.inc()
    start = _time.time()

    try:
        from agents.graph import ask

        result  = ask(request.question, verbose=False)
        elapsed = _time.time() - start

        # Record metrics
        QUERY_LATENCY.observe(elapsed)
        QUERY_TOTAL.labels(status="success").inc()

        is_faithful = result.get("is_faithful", False)
        if is_faithful:
            FAITHFUL_ANSWERS.inc()
        else:
            UNFAITHFUL_ANSWERS.inc()

        return QueryResponse(
            question=request.question,
            answer=result.get("final_answer", ""),
            sources=result.get("sources", []),
            is_faithful=is_faithful,
            sub_queries=result.get("sub_queries", []),
            latency_seconds=round(elapsed, 2)
        )

    except Exception as e:
        QUERY_TOTAL.labels(status="error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
    finally:
        ACTIVE_QUERIES.dec()

@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Ask a question with streaming response.
    Returns tokens as they are generated (SSE format).
    Used for real-time chat interfaces.
    """
    async def generate():
        try:
            from agents.graph import ask

            # Send sub-queries first
            yield "data: Analyzing question...\n\n"

            result = ask(
                request.question,
                verbose=False
            )

            answer = result.get("final_answer", "")

            # Stream answer word by word
            words = answer.split()
            for i, word in enumerate(words):
                chunk = word + (
                    " " if i < len(words) - 1 else "")
                yield f"data: {chunk}\n\n"

            # Send sources at end
            sources = result.get("sources", [])
            if sources:
                yield f"data: \n\nSources:\n\n"
                for s in sources[:3]:
                    yield (f"data: • "
                           f"{s['source']}\n\n")

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )