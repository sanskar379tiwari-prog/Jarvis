import threading
import time
import uuid
from typing import Any
from dotenv import load_dotenv

load_dotenv()  # Initialize environment variables from .env

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jarvis import run_execute
from task_state import create_task, get_task, list_tasks, get_global_llm_calls

app = FastAPI(title="Jarvis Brain", version="2.0.0")


# ───────────────────────────────────────────────────────────
# Request / Response models
# ───────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    text: str


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = "jarvis"
    messages: list[Message]
    stream: bool | None = False


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────

def _message_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def _run_task_in_background(task_state, user_text: str) -> None:
    """Worker function that runs in a background thread."""
    try:
        run_execute(user_text, state=task_state)
    except Exception as exc:
        from task_state import mark_failed
        mark_failed(task_state, str(exc))


# ───────────────────────────────────────────────────────────
# Endpoints — backward compatible
# ───────────────────────────────────────────────────────────

@app.post("/execute")
def execute(req: ExecuteRequest):
    """Synchronous execution — backward compatible with v1."""
    data = run_execute(req.text)
    return {"response": data.get("reply", "Done.")}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    """OpenAI-compatible chat endpoint — backward compatible."""
    user_input = _message_to_text(req.messages[-1].content)

    result = run_execute(user_input)
    content = result.get("reply", "Done.")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "jarvis",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "jarvis",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            }
        ],
    }


# ───────────────────────────────────────────────────────────
# NEW: Async task submission + tracking
# ───────────────────────────────────────────────────────────

@app.post("/execute/async")
def execute_async(req: ExecuteRequest):
    """Submit a task and return immediately with a task_id."""
    task_state = create_task()
    thread = threading.Thread(
        target=_run_task_in_background,
        args=(task_state, req.text),
        daemon=True,
    )
    thread.start()
    return {
        "task_id": task_state.task_id,
        "status": "pending",
        "message": "Task submitted. Poll GET /task/{task_id} for progress.",
    }


@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    """Get the current state of a task."""
    state = get_task(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return state.to_dict()


@app.get("/tasks")
def get_recent_tasks(limit: int = 20):
    """List recent tasks."""
    return {"tasks": list_tasks(limit=limit)}


@app.get("/debug/llm_calls")
def debug_llm_calls():
    """Debug endpoint: total LLM calls since server start."""
    return {"global_llm_calls": get_global_llm_calls()}
