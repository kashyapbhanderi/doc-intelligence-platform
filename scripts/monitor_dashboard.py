"""
scripts/monitor_dashboard.py
Live terminal dashboard querying Prometheus.

Shows real-time system health:
- Request rates
- Latency percentiles
- Error rates
- Weaviate chunk count

Run with Prometheus + API running:
  docker-compose up -d
  python scripts/monitor_dashboard.py
"""
import os
import sys
import time
import requests

PROMETHEUS_URL = "http://localhost:9090"
API_URL        = "http://localhost:8000"
REFRESH_SECS   = 5


def query_prometheus(promql: str) -> float | None:
    """Query Prometheus and return scalar value."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=3
        )
        data    = r.json()
        results = data.get(
            "data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
        return 0.0
    except Exception:
        return None


def get_api_health() -> dict:
    """Get health data directly from API."""
    try:
        r = requests.get(
            f"{API_URL}/api/v1/health",
            timeout=3
        )
        return r.json()
    except Exception:
        return {}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def render_dashboard():
    """Render one frame of the terminal dashboard."""
    health      = get_api_health()
    prom_ok     = query_prometheus("up") is not None

    # Prometheus queries
    total_req   = query_prometheus(
        'sum(http_requests_total)')
    req_rate    = query_prometheus(
        'sum(rate(http_requests_total[1m]))')
    error_rate  = query_prometheus(
        'sum(rate(http_requests_total'
        '{status="5xx"}[1m])) or vector(0)')
    p95_latency = query_prometheus(
        'histogram_quantile(0.95, '
        'rate(http_request_duration_seconds'
        '_bucket[5m]))')
    chunks      = query_prometheus(
        'weaviate_chunks_total')
    queries_ok  = query_prometheus(
        'rag_queries_total{status="success"}')
    queries_err = query_prometheus(
        'rag_queries_total{status="error"}')
    faithful    = query_prometheus(
        'faithful_answers_total')
    unfaithful  = query_prometheus(
        'unfaithful_answers_total')

    clear()
    ts = time.strftime("%H:%M:%S")

    print(f"╔{'═'*58}╗")
    print(f"║  DOC INTELLIGENCE PLATFORM — LIVE MONITOR  "
          f"  {ts}  ║")
    print(f"╠{'═'*58}╣")

    # System status
    api_ok  = health.get("status") == "healthy"
    wv_ok   = health.get("weaviate") == "healthy"
    api_ico = "🟢" if api_ok  else "🔴"
    wv_ico  = "🟢" if wv_ok   else "🔴"
    pr_ico  = "🟢" if prom_ok else "🔴"

    print(f"║  SERVICES                                         ║")
    print(f"║    {api_ico} FastAPI API          "
          f"{wv_ico} Weaviate DB          ║")
    print(f"║    {pr_ico} Prometheus           "
          f"  Chunks: "
          f"{int(chunks or 0):>6}                ║")
    print(f"╠{'═'*58}╣")

    # Request metrics
    r_rate = req_rate or 0.0
    e_rate = error_rate or 0.0
    p95    = p95_latency or 0.0

    print(f"║  REQUESTS                                         ║")
    print(f"║    Total:      {int(total_req or 0):>8}    "
          f"Rate:    {r_rate:>6.2f} req/s        ║")
    print(f"║    Errors:     {int(e_rate):>8}    "
          f"P95 lat: {p95*1000:>6.0f}ms           ║")
    print(f"╠{'═'*58}╣")

    # RAG quality
    ok_q   = int(queries_ok  or 0)
    err_q  = int(queries_err or 0)
    total_q = ok_q + err_q
    success_pct = (
        ok_q / total_q * 100 if total_q > 0 else 0)

    faith   = int(faithful   or 0)
    unfaith = int(unfaithful or 0)
    total_f = faith + unfaith
    faith_pct = (
        faith / total_f * 100 if total_f > 0 else 0)

    print(f"║  RAG QUALITY                                      ║")
    print(f"║    Queries:    {ok_q:>4} OK  "
          f"{err_q:>4} err   "
          f"({success_pct:>5.1f}% success)   ║")
    print(f"║    Faithful:   {faith:>4} yes "
          f"{unfaith:>4} no    "
          f"({faith_pct:>5.1f}% faithful)   ║")
    print(f"╠{'═'*58}╣")
    print(f"║  Prometheus: http://localhost:9090               ║")
    print(f"║  API Docs:   http://localhost:8000/docs          ║")
    print(f"║  Refresh every {REFRESH_SECS}s  |  Ctrl+C to exit"
          f"              ║")
    print(f"╚{'═'*58}╝")


if __name__ == "__main__":
    print("Starting live monitor...")
    print("Make sure docker-compose up -d is running.")
    print("Press Ctrl+C to exit.\n")
    time.sleep(1)

    try:
        while True:
            render_dashboard()
            time.sleep(REFRESH_SECS)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")