"""
scripts/smoke_test_prod.py
Production smoke tests for the live deployment.
Works against localhost or a remote IP.

Usage:
  python scripts/smoke_test_prod.py localhost
  python scripts/smoke_test_prod.py 54.162.45.123

Exit codes:
  0 = all tests passed
  1 = one or more tests failed
"""
import sys
import json
import time
import requests

if len(sys.argv) < 2:
    print("Usage: python smoke_test_prod.py <IP_or_localhost>")
    sys.exit(1)

HOST     = sys.argv[1]
BASE_URL = f"http://{HOST}:8000"
PASSED   = 0
FAILED   = 0

print(f"Running production smoke tests")
print(f"Target: {BASE_URL}")
print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 55)


def check(name: str, ok: bool, detail: str = ""):
    global PASSED, FAILED
    icon = "✅" if ok else "❌"
    print(f"  {icon} {name}")
    if detail:
        print(f"     {detail}")
    if ok:
        PASSED += 1
    else:
        FAILED += 1


# 1. API reachable
try:
    r = requests.get(f"{BASE_URL}/", timeout=10)
    check("API reachable", r.status_code == 200,
          f"HTTP {r.status_code}")
except Exception as e:
    check("API reachable", False, str(e))
    print("\n❌ Cannot reach API. Is docker-compose up?")
    sys.exit(1)

# 2. Health endpoint
try:
    r    = requests.get(
        f"{BASE_URL}/api/v1/health", timeout=10)
    data = r.json()
    check("Health endpoint returns 200",
          r.status_code == 200)
    check("Weaviate connected",
          data.get("weaviate") == "healthy",
          f"weaviate={data.get('weaviate')}")
    check("Data ingested",
          data.get("total_chunks", 0) > 1000,
          f"chunks={data.get('total_chunks', 0)}")
    check("Version present",
          "version" in data,
          f"version={data.get('version')}")
except Exception as e:
    check("Health check", False, str(e))

# 3. Docs accessible
try:
    r = requests.get(f"{BASE_URL}/docs", timeout=10)
    check("Swagger UI accessible",
          r.status_code == 200)
except Exception as e:
    check("Swagger UI", False, str(e))

# 4. Tools endpoint
try:
    r    = requests.get(
        f"{BASE_URL}/api/v1/edit/tools", timeout=10)
    data = r.json()
    check("Tools endpoint works",
          r.status_code == 200)
    check("At least 10 tools registered",
          data.get("total", 0) >= 10,
          f"tools={data.get('total')}")
except Exception as e:
    check("Tools endpoint", False, str(e))

# 5. Input validation
try:
    r = requests.post(
        f"{BASE_URL}/api/v1/query",
        json={"question": "hi"},
        timeout=10
    )
    check("Input validation works (422 on short q)",
          r.status_code == 422,
          f"HTTP {r.status_code}")
except Exception as e:
    check("Input validation", False, str(e))

# 6. Prometheus metrics
try:
    r = requests.get(
        f"{BASE_URL}/metrics", timeout=10)
    check("Prometheus /metrics exposed",
          r.status_code == 200)
    check("Metrics contain http_requests",
          "http_requests_total" in r.text)
except Exception as e:
    check("Prometheus metrics", False, str(e))

# Summary
print("\n" + "=" * 55)
total = PASSED + FAILED
print(f"Results: {PASSED}/{total} passed")
if FAILED == 0:
    print("✅ All smoke tests passed!")
    print(f"   Production is healthy.")
    print(f"   Live URL: {BASE_URL}/docs")
    sys.exit(0)
else:
    print(f"❌ {FAILED} test(s) failed.")
    sys.exit(1)