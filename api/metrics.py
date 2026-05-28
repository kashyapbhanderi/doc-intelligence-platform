"""
api/metrics.py
Prometheus metrics for the Doc Intelligence Platform.

Tracks:
- Request count per endpoint
- Request latency (p50, p95, p99)
- Active requests
- RAG pipeline specific counters

Industry context:
Prometheus + Grafana is the industry standard
monitoring stack. Used at Uber, Airbnb, Shopify.
Grafana ingests Prometheus data and renders dashboards.
"""
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
)

# ── Request counters ──────────────────────────────────────

QUERY_TOTAL = Counter(
    "rag_queries_total",
    "Total number of RAG queries received",
    ["status"]          # labels: success / error
)

INGEST_TOTAL = Counter(
    "ingest_jobs_total",
    "Total document ingestion jobs",
    ["status"]          # labels: success / failed
)

EDIT_TOTAL = Counter(
    "edit_requests_total",
    "Total document edit requests",
    ["file_type"]       # labels: docx / pdf / image
)

# ── Latency histograms ────────────────────────────────────
# Buckets in seconds — gives p50, p95, p99

QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "RAG query end-to-end latency",
    buckets=[0.5, 1.0, 2.0, 5.0,
             10.0, 20.0, 30.0, 60.0]
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_duration_seconds",
    "Weaviate search latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

LLM_LATENCY = Histogram(
    "llm_call_duration_seconds",
    "LLM API call latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# ── Gauges (current state) ────────────────────────────────

WEAVIATE_CHUNKS = Gauge(
    "weaviate_chunks_total",
    "Total chunks currently stored in Weaviate"
)

ACTIVE_QUERIES = Gauge(
    "active_rag_queries",
    "Number of RAG queries currently being processed"
)

# ── Faithfulness tracking ─────────────────────────────────

FAITHFULNESS_SCORE = Summary(
    "critic_faithfulness_score",
    "Faithfulness scores from the Critic agent"
)

FAITHFUL_ANSWERS = Counter(
    "faithful_answers_total",
    "Answers approved by Critic as faithful"
)

UNFAITHFUL_ANSWERS = Counter(
    "unfaithful_answers_total",
    "Answers flagged by Critic as unfaithful"
)