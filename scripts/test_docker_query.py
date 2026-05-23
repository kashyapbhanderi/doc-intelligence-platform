"""
scripts/test_docker_query.py
Test a real query through the Docker API.
Proves the full RAG pipeline works in containers.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

question = "What is retrieval augmented generation?"
print(f"Sending query through Docker API...")
print(f"Q: {question}\n")

start = time.time()
try:
    r = requests.post(
        f"{BASE_URL}/api/v1/query",
        json={"question": question, "top_k": 3},
        timeout=60
    )

    elapsed = time.time() - start

    if r.status_code == 200:
        data = r.json()
        print(f"✅ Query answered in {elapsed:.1f}s")
        print(f"\nAnswer: {data['answer'][:300]}")
        print(f"\nSources ({data['num_sources']}):")
        for s in data.get("sources", [])[:3]:
            print(f"  → {s['source']} "
                  f"(page {s['page']})")
    else:
        print(f"❌ Error: HTTP {r.status_code}")
        print(r.text[:200])

except requests.Timeout:
    print("❌ Request timed out after 60s")
    print("   LLM call may be slow via OpenRouter")
except Exception as e:
    print(f"❌ Error: {e}")