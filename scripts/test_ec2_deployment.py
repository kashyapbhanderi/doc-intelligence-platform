"""
scripts/test_ec2_deployment.py
Smoke tests for the EC2 deployment.
Run from your LOCAL machine with the EC2 IP.

Usage:
  python scripts/test_ec2_deployment.py 54.162.45.123
"""
import sys
import requests
import json

if len(sys.argv) < 2:
    print("Usage: python test_ec2_deployment.py <EC2_IP>")
    print("Example: python test_ec2_deployment.py 54.162.45.123")
    sys.exit(1)

EC2_IP   = sys.argv[1]
BASE_URL = f"http://{EC2_IP}:8000"

print(f"Testing EC2 deployment at {BASE_URL}")
print("=" * 55)

tests   = []
passed  = 0


def test(name: str, url: str,
         method: str = "GET",
         body: dict = None,
         expect: int = 200):
    global passed
    try:
        if method == "GET":
            r = requests.get(url, timeout=15)
        else:
            r = requests.post(url, json=body,
                              timeout=30)
        ok   = r.status_code == expect
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name:<40} "
              f"HTTP {r.status_code}")
        if ok:
            passed += 1
        else:
            print(f"     Response: {r.text[:80]}")
        tests.append(ok)
    except requests.ConnectionError:
        print(f"  ❌ {name:<40} CONNECTION REFUSED")
        print(f"     Is the EC2 stack running?")
        tests.append(False)
    except requests.Timeout:
        print(f"  ⚠️  {name:<40} TIMEOUT (slow start)")
        tests.append(False)


print("\n1. Core Endpoints:")
test("Root /",
     f"{BASE_URL}/")
test("Health check",
     f"{BASE_URL}/api/v1/health")
test("Swagger docs",
     f"{BASE_URL}/docs")
test("Prometheus metrics",
     f"{BASE_URL}/metrics")

print("\n2. API Functionality:")
test("List editor tools",
     f"{BASE_URL}/api/v1/edit/tools")
test("Ingest status",
     f"{BASE_URL}/api/v1/ingest/status")
test("Tasks list",
     f"{BASE_URL}/api/v1/ingest/tasks")

print("\n3. Validation:")
test("Short question → 422",
     f"{BASE_URL}/api/v1/query",
     method="POST",
     body={"question": "hi"},
     expect=422)

# Health details
print("\n4. System Details:")
try:
    r    = requests.get(
        f"{BASE_URL}/api/v1/health", timeout=10)
    data = r.json()
    wv   = data.get("weaviate", "unknown")
    ch   = data.get("total_chunks", 0)
    v    = data.get("version", "?")

    wv_icon = "✅" if wv == "healthy" else "⚠️"
    ch_icon = "✅" if ch > 0 else "⚠️"

    print(f"  {wv_icon} Weaviate:    {wv}")
    print(f"  {ch_icon} Chunks:      {ch} "
          f"{'(run reembed)' if ch == 0 else ''}")
    print(f"  ✅ Version:    {v}")
except Exception as e:
    print(f"  ❌ Could not fetch health: {e}")

# Summary
print("\n" + "=" * 55)
print(f"Results: {passed}/{len(tests)} passed")
if passed == len(tests):
    print(f"✅ EC2 deployment is healthy!")
    print(f"   Live URL: {BASE_URL}/docs")
    print(f"   Add this to your resume + README!")
else:
    print(f"⚠️  {len(tests)-passed} test(s) need attention.")
print("=" * 55)
