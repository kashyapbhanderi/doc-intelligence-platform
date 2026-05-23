import pytest
import sys
import os
import time
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from api.task_queue import (
    create_task,
    get_task,
    update_task,
    list_tasks,
    TaskStatus,
)


def test_create_task_returns_id():
    """Task creation should return an ID string."""
    task_id = create_task("test.pdf")
    assert isinstance(task_id, str)
    assert len(task_id) > 0


def test_create_task_stored():
    """Created task should be retrievable."""
    task_id = create_task("test.pdf")
    task    = get_task(task_id)
    assert task is not None
    assert task["filename"] == "test.pdf"


def test_new_task_status_pending():
    """New task should start as pending."""
    task_id = create_task("test.pdf")
    task    = get_task(task_id)
    assert task["status"] == TaskStatus.PENDING


def test_update_task_to_processing():
    """Task should update to processing."""
    task_id = create_task("test.pdf")
    update_task(task_id, TaskStatus.PROCESSING)
    task = get_task(task_id)
    assert task["status"] == TaskStatus.PROCESSING


def test_update_task_to_done():
    """Task should update to done with chunks."""
    task_id = create_task("test.pdf")
    update_task(task_id, TaskStatus.DONE,
                chunks=150)
    task = get_task(task_id)
    assert task["status"]  == TaskStatus.DONE
    assert task["chunks"]  == 150


def test_update_task_to_failed():
    """Task should update to failed with error."""
    task_id = create_task("test.pdf")
    update_task(task_id, TaskStatus.FAILED,
                error="File not found")
    task = get_task(task_id)
    assert task["status"] == TaskStatus.FAILED
    assert "File not found" in task["error"]


def test_done_task_has_completed_time():
    """Done tasks should have completion time."""
    task_id = create_task("test.pdf")
    update_task(task_id, TaskStatus.DONE)
    task = get_task(task_id)
    assert task["completed"] is not None


def test_pending_task_no_completed_time():
    """Pending tasks should not have completion time."""
    task_id = create_task("test.pdf")
    task    = get_task(task_id)
    assert task["completed"] is None


def test_task_has_elapsed_time():
    """Every task should report elapsed time."""
    task_id = create_task("test.pdf")
    task    = get_task(task_id)
    assert "elapsed" in task
    assert task["elapsed"] >= 0


def test_get_nonexistent_task():
    """Getting unknown task ID returns None."""
    result = get_task("nonexistent123")
    assert result is None


def test_list_tasks_returns_list():
    """list_tasks should return a list."""
    create_task("test1.pdf")
    create_task("test2.pdf")
    tasks = list_tasks()
    assert isinstance(tasks, list)
    assert len(tasks) >= 2


def test_list_tasks_sorted_newest_first():
    """Most recent tasks should appear first."""
    id1 = create_task("first.pdf")
    time.sleep(0.01)
    id2 = create_task("second.pdf")
    tasks = list_tasks()
    ids   = [t["task_id"] for t in tasks]
    assert ids.index(id2) < ids.index(id1)