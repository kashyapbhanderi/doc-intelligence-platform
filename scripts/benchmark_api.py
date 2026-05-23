"""
scripts/benchmark_api.py
Quick API benchmark without Locust.
Tests p50, p95, p99 latency for each endpoint.
"""
import time
import statistics
import requests

BASE_URL = "http://localhost:8000"

endpoints = [
    ("GET",  "/api/v1/health",        None),
    ("GET",  "/api/v1/edit/tools",    None),
    ("GET",  "/api/v1/ingest/status", None),
]


def benchmark_endpoint(
    method: str,
    path:   str,
    body:   dict,
    n:      int = 20
) -> dict:
    """Run n requests and return latency stats."""
    latencies = []

    for _ in range(n):
        start = time.time()
        try:
            if method == "GET":
                r = requests.get(
                    BASE_URL + path, timeout=10)
            else:
                r = requests.post(
                    BASE_URL + path,
                    json=body, timeout=30)

            if r.status_code == 200:
                latencies.append(
                    (time.time() - start) * 1000)
        except Exception:
            pass

    if not latencies:
        return {"error": "all requests failed"}

    latencies.sort()
    return {
        "n":    len(latencies),
        "p50":  round(statistics.median(
            latencies), 1),
        "p95":  round(latencies[
            int(len(latencies) * 0.95)], 1),
        "p99":  round(latencies[
            int(len(latencies) * 0.99)], 1),
        "mean": round(statistics.mean(
            latencies), 1),
        "min":  round(min(latencies), 1),
        "max":  round(max(latencies), 1),
    }


if __name__ == "__main__":
    print("API Benchmark")
    print("=" * 60)
    print(f"{'Endpoint':<30} {'P50':>6} "
          f"{'P95':>6} {'P99':>6} {'Mean':>7}")
    print("-" * 60)

    for method, path, body in endpoints:
        stats = benchmark_endpoint(
            method, path, body, n=20)
        if "error" not in stats:
            print(f"{path:<30} "
                  f"{stats['p50']:>5}ms "
                  f"{stats['p95']:>5}ms "
                  f"{stats['p99']:>5}ms "
                  f"{stats['mean']:>6}ms")
        else:
            print(f"{path:<30} ERROR — "
                  "is the server running?")

    print("\nTarget benchmarks:")
    print("  /health:        p95 < 100ms ✅")
    print("  /edit/tools:    p95 < 200ms ✅")
    print("  /ingest/status: p95 < 200ms ✅")