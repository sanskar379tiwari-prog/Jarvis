"""
Task state management, logging, and LLM usage tracking.

This module is 100% deterministic — it NEVER calls the LLM.
All logging is plain string-based and stored in memory.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskState:
    task_id: str
    status: str = "pending"          # pending | running | completed | failed
    current_step: str = ""
    progress: float = 0.0           # 0–100
    logs: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    llm_calls_used: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "current_step": self.current_step,
            "progress": self.progress,
            "logs": self.logs,
            "result": self.result,
            "llm_calls_used": self.llm_calls_used,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# In-memory task store (thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_tasks: dict[str, TaskState] = {}


def create_task() -> TaskState:
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    state = TaskState(task_id=task_id)
    with _lock:
        _tasks[task_id] = state
    return state


def get_task(task_id: str) -> TaskState | None:
    with _lock:
        return _tasks.get(task_id)


def list_tasks(limit: int = 20) -> list[dict[str, Any]]:
    with _lock:
        recent = sorted(_tasks.values(), key=lambda t: t.created_at, reverse=True)[:limit]
    return [t.to_dict() for t in recent]


# ---------------------------------------------------------------------------
# Deterministic logging (NO LLM)
# ---------------------------------------------------------------------------
def log_event(state: TaskState, message: str) -> None:
    entry = {"time": time.time(), "message": message}
    state.logs.append(entry)
    print(f"[{state.task_id}] {message}")


def update_step(state: TaskState, step_label: str, progress: float) -> None:
    state.current_step = step_label
    state.progress = min(progress, 100.0)
    log_event(state, step_label)


def mark_running(state: TaskState) -> None:
    state.status = "running"
    log_event(state, "Task started")


def mark_completed(state: TaskState, result: dict[str, Any]) -> None:
    state.status = "completed"
    state.progress = 100.0
    state.result = result
    state.completed_at = time.time()
    log_event(state, f"Task completed — LLM calls used: {state.llm_calls_used}")


def mark_failed(state: TaskState, error: str) -> None:
    state.status = "failed"
    state.result = {"ok": False, "error": error}
    state.completed_at = time.time()
    log_event(state, f"Task failed: {error}")


# ---------------------------------------------------------------------------
# LLM call counter
# ---------------------------------------------------------------------------
_llm_counter_lock = threading.Lock()
_global_llm_calls = 0


def increment_llm_counter(state: TaskState | None = None) -> int:
    global _global_llm_calls
    with _llm_counter_lock:
        _global_llm_calls += 1
        total = _global_llm_calls
    if state:
        state.llm_calls_used += 1
    return total


def get_global_llm_calls() -> int:
    with _llm_counter_lock:
        return _global_llm_calls
