"""
scripts/locustfile.py
Load test for the Doc Intelligence Platform API.

Simulates real users:
- Checking health
- Asking questions
- Listing tools

Run with:
  locust -f scripts/locustfile.py --host http://localhost:8000

Then open: http://localhost:8089
"""
from locust import HttpUser, task, between
import random


QUESTIONS = [
    "What is retrieval augmented generation?",
    "How does LoRA fine-tuning work?",
    "What is the attention mechanism?",
    "How do agents use tools?",
    "What is contrastive learning?",
]


class APIUser(HttpUser):
    """
    Simulates a typical API user.
    wait_time: between 1-3 seconds between requests
    (realistic human usage pattern)
    """
    wait_time = between(1, 3)

    @task(5)
    def check_health(self):
        """
        Weight=5: Health check is most frequent.
        Simulates monitoring systems pinging health.
        """
        self.client.get("/api/v1/health")

    @task(3)
    def list_tools(self):
        """
        Weight=3: Browsing available tools.
        """
        self.client.get("/api/v1/edit/tools")

    @task(2)
    def check_ingest_status(self):
        """
        Weight=2: Checking database status.
        """
        self.client.get("/api/v1/ingest/status")

    @task(1)
    def ask_question(self):
        """
        Weight=1: Asking questions (expensive — LLM call)
        Low weight because it uses API credits.
        """
        question = random.choice(QUESTIONS)
        self.client.post(
            "/api/v1/query",
            json={
                "question": question,
                "top_k":    5
            },
            timeout=30  # LLM calls can be slow
        )


class LightUser(HttpUser):
    """
    Lightweight user — only hits fast endpoints.
    Use this for pure throughput testing.
    """
    wait_time = between(0.5, 1)
    weight    = 3  # 3x more light users

    @task
    def health_only(self):
        self.client.get("/api/v1/health")