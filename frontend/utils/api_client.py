"""
frontend/utils/api_client.py
Handles all HTTP calls to the FastAPI backend.
"""
import requests
import os

BASE_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


def get_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def ask_question(question: str, top_k: int = 5):
    try:
        r = requests.post(
            f"{BASE_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=60
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def upload_document(file_bytes, filename: str):
    try:
        r = requests.post(
            f"{BASE_URL}/ingest",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=30
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def get_ingest_status():
    try:
        r = requests.get(f"{BASE_URL}/ingest/status", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_task_status(task_id: str):
    try:
        r = requests.get(f"{BASE_URL}/ingest/tasks/{task_id}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def list_tasks():
    try:
        r = requests.get(f"{BASE_URL}/ingest/tasks", timeout=5)
        return r.json() if r.status_code == 200 else {"tasks": []}
    except Exception:
        return {"tasks": []}


def edit_document(instruction: str, file_path: str):
    try:
        r = requests.post(
            f"{BASE_URL}/edit",
            json={"instruction": instruction, "file_path": file_path},
            timeout=60
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def list_tools():
    try:
        r = requests.get(f"{BASE_URL}/edit/tools", timeout=5)
        return r.json() if r.status_code == 200 else {"tools": []}
    except Exception:
        return {"tools": []}