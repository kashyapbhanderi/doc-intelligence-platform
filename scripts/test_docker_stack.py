"""
scripts/test_docker_stack.py
End-to-end test of the full Docker stack.
Runs outside Docker, hits the API container.
Tests that API can reach Weaviate container.
"""
import sys
import requests
import json
import time

BASE_URL = "http://localhost:8000"

results  = []
failures = 0


def test(name: str, method: str,
         path: str, body: dict = None,
         expected_status: int = 200,
         check_field: str = None):
    """Run one test and print result."""
    global failures

    try:
        if method == "GET":
            r = requests.get(
                BASE_URL + path, timeout=15)
        else:
            r = requests.post(
                BASE_URL + path,
                json=body, timeout=30)

        passed = r.status_code == expected_status

        if passed and check_field:
            data   = r.json()
            passed = check_field in data

        icon = "✅" if passed else "❌"
        status_ok = (f"HTTP {r.status_code}"
                     if not passed
                     else "OK")
        print(f"  {icon} {name:<40} {status_ok}")

        if not passed:
            failures += 1
            print(f"     Response: "
                  f"{r.text[:100]}")

        results.append({
            "name":   name,
            "passed": passed
        })

    except Exception as e:
        print(f"  ❌ {name:<40} ERROR: {e}")
        failures += 1


print("=" * 60)
print("FULL DOCKER STACK TEST")
print("=" * 60)
print(f"API URL: {BASE_URL}\n")

# Check API is reachable
print("1. API Connectivity:")
test("Root endpoint",
     "GET", "/",
     check_field="name")
test("Swagger docs accessible",
     "GET", "/docs")

# Check health — proves API → Weaviate works
print("\n2. Health + Weaviate Connection:")
test("Health returns 200",
     "GET", "/api/v1/health",
     check_field="status")

# Check response body
try:
    r = requests.get(
        f"{BASE_URL}/api/v1/health", timeout=10)
    data = r.json()
    weaviate_ok = data.get("weaviate") == "healthy"
    chunks_ok   = data.get("total_chunks", 0) > 0

    icon1 = "✅" if weaviate_ok else "❌"
    icon2 = "✅" if chunks_ok   else "⚠️"
    print(f"  {icon1} Weaviate status: "
          f"{data.get('weaviate')}")
    print(f"  {icon2} Chunks loaded:   "
          f"{data.get('total_chunks', 0)}")

    if not chunks_ok:
        print("     ⚠️  0 chunks — run "
              "python embeddings/reembed.py "
              "then re-test")
        failures += 1
except Exception as e:
    print(f"  ❌ Could not parse health: {e}")
    failures += 1

# Check all routers
print("\n3. API Endpoints:")
test("GET /api/v1/edit/tools",
     "GET", "/api/v1/edit/tools",
     check_field="tools")
test("GET /api/v1/ingest/status",
     "GET", "/api/v1/ingest/status",
     check_field="status")
test("GET /api/v1/ingest/tasks",
     "GET", "/api/v1/ingest/tasks",
     check_field="tasks")

# Validation
print("\n4. Input Validation:")
test("Short question returns 422",
     "POST", "/api/v1/query",
     body={"question": "hi"},
     expected_status=422)

# Summary
print("\n" + "=" * 60)
passed = len(results) - failures
print(f"Results: {passed}/{len(results)} passed")
if failures == 0:
    print("✅ Full Docker stack is healthy!")
    print("   API ↔ Weaviate networking works.")
    print("   Ready for cloud deployment.")
else:
    print(f"❌ {failures} test(s) failed.")
    print("   Check docker ps and docker logs.")
print("=" * 60)