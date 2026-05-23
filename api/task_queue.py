"""
api/task_queue.py

Simple in-memory task queue for tracking
background jobs.

In production this would use Redis + Celery.
For this project we use a simple dict-based
queue that works without extra infrastructure.

Industry term: "Job queue" or "Task queue"
Every production API has one for long-running ops.
"""
import uuid
import time
from typing import Dict
from enum import Enum


class TaskStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class TaskInfo:
    def __init__(self, task_id: str,
                 filename: str):
        self.task_id   = task_id
        self.filename  = filename
        self.status    = TaskStatus.PENDING
        self.chunks    = 0
        self.error     = ""
        self.created   = time.time()
        self.completed = None

    def to_dict(self) -> dict:
        return {
            "task_id":   self.task_id,
            "filename":  self.filename,
            "status":    self.status,
            "chunks":    self.chunks,
            "error":     self.error,
            "created":   self.created,
            "completed": self.completed,
            "elapsed":   round(
                time.time() - self.created, 1)
        }


# Global in-memory task store
# In production: use Redis
_tasks: Dict[str, TaskInfo] = {}


def create_task(filename: str) -> str:
    """Create a new task and return its ID."""
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = TaskInfo(task_id, filename)
    return task_id


def get_task(task_id: str) -> dict | None:
    """Get task status by ID."""
    task = _tasks.get(task_id)
    return task.to_dict() if task else None


def update_task(
    task_id: str,
    status: TaskStatus,
    chunks: int = 0,
    error: str = ""
):
    """Update task status."""
    task = _tasks.get(task_id)
    if task:
        task.status = status
        task.chunks = chunks
        task.error  = error
        if status in (TaskStatus.DONE,
                      TaskStatus.FAILED):
            task.completed = time.time()


def list_tasks() -> list:
    """List all tasks sorted by creation time."""
    return sorted(
        [t.to_dict() for t in _tasks.values()],
        key=lambda x: x["created"],
        reverse=True
    )