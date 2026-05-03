import ast
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from ast import literal_eval
from pathlib import Path
from typing import Any, Callable

try:
    import pypdf
except ImportError:
    pypdf = None

CODE_FILE_SUFFIXES = frozenset({".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"})
_FENCE_BLOCK_RE = re.compile(r"```[ \t]*([^\n`]*)\s*\n([\s\S]*?)```", re.MULTILINE)
_CHAFF_PATTERNS = (
    "follow_ups",
    "## instruction",
    "## response",
    "current draft:",
    "snippet of current draft",
    "same diffterminal",
    "terminal_easy",
    "remember that this is just an example",
    "provide me with python code that interfaces",
)

from llm_provider import call_llm
from task_state import (
    TaskState, create_task, log_event, update_step,
    mark_running, mark_completed, mark_failed, increment_llm_counter,
)

PROJECT_ROOT = Path(__file__).resolve().parent
# Canonical workspace root for all tool actions.
# This avoids accidental nested paths like jarvis_brain/jarvis_brain/...
BASE_DIR = PROJECT_ROOT

# Prevent tools from nuking the FastAPI entrypoint / LLM layer when generating random filenames.
_PROTECTED_ROOT_FILENAMES = frozenset({"api.py", "jarvis.py", "llm_provider.py", "task_state.py"})

ALLOWED_ACTIONS = {"create_file", "write_file", "read_file", "delete_file", "list_files", "rename_file", "edit_file", "create_tool", "create_folder"}

LANGUAGE_EXT_MAP = {
    "python": ".py",
    "cpp": ".cpp",
    "c++": ".cpp",
    "javascript": ".js",
    "typescript": ".ts",
    "java": ".java",
    "c": ".c",
    "markdown": ".md",
    "text": ".txt",
    "json": ".json",
    "header": ".h"
}

# ═══════════════════════════════════════════════════════════════════════════
# WORKSPACE ORGANIZATION — strict directory policy
# ═══════════════════════════════════════════════════════════════════════════

CREATED_FILES_DIR = BASE_DIR / "created_files"
TOOLS_DIR_PATH = BASE_DIR / "tools"
MEMORY_DIR = BASE_DIR / "memory"
LOGS_DIR = BASE_DIR / "logs"

# Directories that are part of the system — NOT user content
_SYSTEM_DIRS = frozenset({"__pycache__", ".git", "tools", "memory", "logs", ".env"})

def _ensure_workspace_dirs() -> None:
    """Create mandatory workspace directories if they don't exist."""
    for d in (CREATED_FILES_DIR, TOOLS_DIR_PATH, MEMORY_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

_ensure_workspace_dirs()


def _prefix_created_files(filename: str, intent_type: str | None = None) -> str:
    """Route generated files into created_files/ unless they already live there.
    
    Read/delete/list operations are NOT prefixed — they search for existing files.
    Only creation/edit operations get the prefix.
    """
    if not filename:
        return filename
    
    normalized = filename.replace("\\", "/").strip()
    
    # Already inside created_files/
    if normalized.startswith("created_files/") or normalized.startswith("created_files\\"):
        return normalized
    
    # System paths — don't redirect
    if normalized.startswith("tools/") or normalized.startswith("memory/") or normalized.startswith("logs/"):
        return normalized
    
    return f"created_files/{normalized}"


def _slugify_task(user_text: str) -> str:
    """Create a folder-safe slug from user input for auto-folder naming."""
    # Extract meaningful words, skip very common ones
    stop_words = {"a", "an", "the", "in", "of", "for", "and", "or", "to", "is",
                  "create", "make", "write", "generate", "build", "file", "code",
                  "please", "me", "with", "using", "implement", "program"}
    words = re.findall(r'[a-zA-Z0-9]+', user_text.lower())
    meaningful = [w for w in words if w not in stop_words and len(w) > 1][:3]
    if not meaningful:
        return "project"
    return "_".join(meaningful)


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT SYSTEM — 3 layers, all deterministic
# ═══════════════════════════════════════════════════════════════════════════

_CONVERSATION_BUFFER_SIZE = 5
_TASK_HISTORY_SIZE = 20
_TASK_HISTORY_FILE = MEMORY_DIR / "task_history.json"

# Thread-safe conversation buffer
_conversation_lock = threading.Lock()
_conversation_history: list[dict[str, str]] = []


def _add_to_conversation(role: str, content: str) -> None:
    """Append a message to the conversation buffer (auto-trims to last N)."""
    with _conversation_lock:
        _conversation_history.append({"role": role, "content": content[:500]})
        # Trim to keep only the last N entries
        while len(_conversation_history) > _CONVERSATION_BUFFER_SIZE:
            _conversation_history.pop(0)


def _get_conversation_context() -> str:
    """Return formatted conversation history."""
    with _conversation_lock:
        if not _conversation_history:
            return ""
        lines = []
        for msg in _conversation_history:
            lines.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(lines)


def get_workspace_context() -> str:
    """Scan created_files/ and return a tree-view string. Max 3 levels deep."""
    if not CREATED_FILES_DIR.exists():
        return "created_files/ (empty)"
    
    lines = ["created_files/"]
    _build_tree(CREATED_FILES_DIR, lines, prefix="", depth=0, max_depth=3)
    
    if len(lines) == 1:
        return "created_files/ (empty)"
    return "\n".join(lines)


def _build_tree(directory: Path, lines: list[str], prefix: str, depth: int, max_depth: int) -> None:
    """Recursively build a tree view."""
    if depth >= max_depth:
        return
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return
    
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "+-- " if is_last else "|-- "
        lines.append(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if is_last else "|   "
            _build_tree(entry, lines, prefix + extension, depth + 1, max_depth)


def _save_task_summary(summary: str) -> None:
    """Append a task summary to the persistent task history file."""
    _ensure_workspace_dirs()
    history: list[dict[str, str]] = []
    if _TASK_HISTORY_FILE.exists():
        try:
            history = json.loads(_TASK_HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            history = []
    
    history.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": summary[:200],
    })
    # Keep only last N
    history = history[-_TASK_HISTORY_SIZE:]
    _TASK_HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _get_task_summaries(n: int = 5) -> str:
    """Return the last N task summaries as formatted text."""
    if not _TASK_HISTORY_FILE.exists():
        return ""
    try:
        history = json.loads(_TASK_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return ""
    
    recent = history[-n:]
    if not recent:
        return ""
    lines = []
    for entry in recent:
        lines.append(f"- [{entry.get('timestamp', '?')}] {entry.get('summary', '?')}")
    return "\n".join(lines)


def _generate_task_summary(intent: dict[str, Any], result: dict[str, Any]) -> str:
    """Create a deterministic 1-line task summary from intent + result. No LLM call."""
    task_type = intent.get("task_type", "unknown")
    files = intent.get("files", [])
    ok = result.get("ok", False)
    status = "OK" if ok else "FAILED"
    
    if task_type == "chat":
        return f"[{status}] Chat response"
    
    filenames = [f.get("filename", "?") for f in files]
    names_str = ", ".join(filenames[:5])
    if len(filenames) > 5:
        names_str += f" (+{len(filenames) - 5} more)"
    
    return f"[{status}] {task_type}: {names_str}"


def build_context(user_input: str) -> str:
    """Combine all 3 context layers into a lightweight context string."""
    parts: list[str] = []
    
    # Layer 1: Conversation history
    conv = _get_conversation_context()
    if conv:
        parts.append(f"Recent conversation:\n{conv}")
    
    # Layer 2: Workspace state
    ws = get_workspace_context()
    if ws:
        parts.append(f"Workspace:\n{ws}")
    
    # Layer 3: Task memory
    tasks = _get_task_summaries(5)
    if tasks:
        parts.append(f"Recent tasks:\n{tasks}")
    
    if not parts:
        return ""
    return "\n\n".join(parts)



def _fix_filename(name: str, task: str) -> str:
    """Sanitize filename (remove quotes, spaces). Stop blindly appending extensions."""
    if not name or not isinstance(name, str):
        return name
    return name.strip().replace('"', '').replace("'", "")


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data}


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _fallback_plan() -> dict[str, Any]:
    # Safe deterministic fallback: read-only action to keep pipeline stable.
    return {"action": "list_files", "path": "."}


def _looks_like_tool_request(text: str) -> bool:
    """If False, treat as normal conversation (don't run filesystem planner fallbacks)."""
    if not (text or "").strip():
        return False
    t = text.lower()
    if re.search(r"\b(create|write|make|generate|save)\s+.+\b(file|folder)\b", t):
        return True
    if re.search(r"\b(read|open|show)\s+.+\bfile\b", t):
        return True
    if re.search(r"\b(delete|remove)\s+.+\bfile\b", t):
        return True
    if re.search(r"\blist\b.*\b(files|folders?|directories|directory)\b", t):
        return True
    if re.search(r"\bsave\b.+\b(to|into)\s+[\w./\\\-]+\.(py|txt|cpp|c|js|java|md|json)", t):
        return True
    if re.search(r"\.(py|txt|cpp|cxx|cc|c|js|java|md|json|h|hpp)\b", t):
        return True
    if re.search(r"\b(folder|directory)\s+[\w\-]+\s+(already\s+)?present\b", t):
        return True
    if "inside " in t and " folder" in t:
        return True
    if "jarvis_brain" in t or "workspace" in t.replace("_", ""):
        return True
    if _looks_like_create_request(text):
        return True
    return False


def _chat_reply(user_text: str) -> str:
    prompt = (
        "You are Jarvis: concise, friendly, accurate. Answer in plain language. "
        "No JSON. Do not list workspace files unless the user explicitly asks about files or folders.\n\n"
        f"User message:\n{user_text}"
    )
    return call_llm(prompt).strip()


def _contains_generation_chaff(text: str) -> bool:
    if not text:
        return True
    folded = text.casefold()
    return any(p in folded for p in _CHAFF_PATTERNS)


def _validate_python_source(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _validate_cpp_source(code: str, temp_suffix: str = ".cpp") -> bool:
    exe = shutil.which("g++") or shutil.which("clang++")
    if not exe:
        if re.search(r"#include\s*[<\"]", code) and "{" in code and len(code.strip()) > 20:
            return True
        if temp_suffix == ".c" and "/*" in code and "*/" in code and len(code.strip()) > 5:
            return True
        return False
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=temp_suffix, text=False)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(code)
        completed = subprocess.run(
            [exe, "-std=c++17", "-fsyntax-only", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _validate_code_by_extension(code: str, filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    if ext == ".py":
        return _validate_python_source(code)
    if ext in {".h", ".hpp", ".hxx"}:
        return bool(
            re.search(r"#\s*(ifndef|define|pragma\s+once|include)", code)
            and len(code.strip()) > 5
        )
    if ext in {".cpp", ".cc", ".cxx", ".c"}:
        return _validate_cpp_source(code, ".c" if ext == ".c" else ".cpp")
    return True


def _extract_code_from_llm_output(raw: str, filename: str) -> str | None:
    """Pick best code block from messy LLM output; prefer fenced blocks that pass syntax."""
    if not raw or not str(raw).strip():
        return None
    text = str(raw).replace("\r\n", "\n")
    ext = Path(filename).suffix.lower()

    single = re.match(r"^\s*```[ \t]*[^\n]*\s*\n([\s\S]*?)```\s*\Z", text.strip())
    if single:
        body = single.group(1).strip("\n")
        if not _contains_generation_chaff(body) and _validate_code_by_extension(body, filename):
            return body

    candidates: list[tuple[int, str]] = []
    for match in _FENCE_BLOCK_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower()
        body = match.group(2).strip("\n")
        if not body or _contains_generation_chaff(body):
            continue
        priority = 0
        if ext == ".py" and lang in {"python", "py", ""}:
            priority = 3
        elif ext in {".cpp", ".cc", ".cxx"} and lang in {"cpp", "c++", "cxx", "c", ""}:
            priority = 3
        elif ext == ".c" and lang in {"c", ""}:
            priority = 3
        valid = _validate_code_by_extension(body, filename)
        score = priority * 1_000_000 + (1 if valid else 0) * 100_000 + len(body)
        candidates.append((score, body))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _score, body in candidates:
            if _validate_code_by_extension(body, filename):
                return body
        return candidates[0][1] if ext not in CODE_FILE_SUFFIXES else None

    stripped = text.strip()
    if ext in CODE_FILE_SUFFIXES and not _contains_generation_chaff(stripped):
        if _validate_code_by_extension(stripped, filename):
            return stripped
    return None


def _unwrap_code_fence(text: str) -> str:
    """Backward-compatible name: extract usable text from fenced blobs."""
    value = (text or "").replace("\r\n", "\n")
    match = re.match(r"^\s*```[a-zA-Z0-9_+\-#]*\s*\n([\s\S]*?)\n```\s*$", value.strip())
    if match:
        return match.group(1)
    return value


def _finalize_code_content(content: str, filename: str) -> str:
    """Normalize planner/LLM content before execute."""
    raw = content if isinstance(content, str) else str(content)
    ext = Path(filename).suffix.lower()
    if ext not in CODE_FILE_SUFFIXES:
        return raw
    extracted = _extract_code_from_llm_output(raw, filename)
    if extracted:
        return extracted
    inner = _unwrap_code_fence(raw)
    if inner != raw and not _contains_generation_chaff(inner):
        if _validate_code_by_extension(inner, filename):
            return inner
    return raw.strip()


def _looks_like_code_request(user_text: str) -> bool:
    lowered = (user_text or "").lower()
    return any(
        token in lowered
        for token in (
            "code",
            "implement",
            "implementation",
            "program",
            ".cpp",
            ".py",
            ".js",
            ".java",
            ".c",
            ".ts",
            "class ",
            "function ",
            "algorithm",
            "selection sort",
        )
    )


def _is_likely_bad_code(content: str, filename: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    if _contains_generation_chaff(text):
        return True
    lowered = text.lower()
    if lowered.startswith("{") and "follow_ups" in lowered:
        return True
    if lowered.startswith("{") and '"action"' in lowered:
        return True
    if "can you provide an example" in lowered:
        return True
    ext = Path(filename).suffix.lower()
    if ext in CODE_FILE_SUFFIXES and not _validate_code_by_extension(text, filename):
        return True
    return False


def _self_improve_code(user_text: str, filename: str, draft_content: str) -> str:
    """Single short rewrite pass; output must pass extract + syntax + chaff checks."""
    if Path(filename).suffix.lower() not in CODE_FILE_SUFFIXES:
        return draft_content
    prompt = (
        "Rewrite the program to satisfy the task. Output exactly one markdown fenced code block "
        f"for the file type ({Path(filename).suffix}). No commentary outside the fence."
    )
    try:
        raw = call_llm(f"{prompt}\n\nTask:\n{user_text}\n\nFile:\n{filename}\n\nDraft:\n{draft_content[:12000]}")
    except Exception:
        return draft_content
    improved = _extract_code_from_llm_output(raw, filename)
    if (
        improved
        and not _contains_generation_chaff(improved)
        and _validate_code_by_extension(improved, filename)
    ):
        return improved
    return draft_content


def _extract_filename_from_text(user_text: str) -> str:
    text = user_text or ""
    match = re.search(r"\bcalled\s+([a-zA-Z0-9_\- ]+)", text, flags=re.IGNORECASE)
    raw_name = match.group(1).strip(" ,.") if match else ""
    if not raw_name:
        ext_match = re.search(r"\b([a-zA-Z0-9_\- ]+\.[a-zA-Z0-9]+)\b", text)
        raw_name = ext_match.group(1).strip() if ext_match else "new_file.txt"

    # Trim trailing instruction words from names like "sorted array write the code..."
    raw_name = re.split(r"\b(write|with|inside|in)\b", raw_name, flags=re.IGNORECASE)[0].strip(" ,.")
    safe_name = re.sub(r"\s+", "_", raw_name)
    safe_name = re.sub(r"[^a-zA-Z0-9_.\-]", "", safe_name) or "new_file"

    if "." not in safe_name:
        lowered = text.lower()
        if "cpp" in lowered or "c++" in lowered:
            safe_name = f"{safe_name}.cpp"
        elif "python" in lowered or ".py" in lowered:
            safe_name = f"{safe_name}.py"
        else:
            safe_name = f"{safe_name}.txt"
    return safe_name


def _extract_folder_from_text(user_text: str) -> str | None:
    text = user_text or ""
    match = re.search(r"\b(?:inside|in)\s+([a-zA-Z0-9_\-/ ]+?)\s+folder\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    folder = match.group(1).strip().replace(" ", "_")
    folder = re.sub(r"[^a-zA-Z0-9_\-/]", "", folder)
    return folder or None


def _build_fallback_content(user_text: str, filename: str) -> str:
    lowered = (user_text or "").lower()
    if filename.lower().endswith(".cpp"):
        if "sorted" in lowered and "array" in lowered:
            return (
                "#include <iostream>\n"
                "#include <vector>\n"
                "using namespace std;\n\n"
                "bool isSorted(const vector<int>& arr) {\n"
                "    for (size_t i = 1; i < arr.size(); ++i) {\n"
                "        if (arr[i] < arr[i - 1]) {\n"
                "            return false;\n"
                "        }\n"
                "    }\n"
                "    return true;\n"
                "}\n\n"
                "int main() {\n"
                "    vector<int> arr = {1, 2, 3, 4, 5};\n"
                "    cout << (isSorted(arr) ? \"Array is sorted\" : \"Array is not sorted\") << endl;\n"
                "    return 0;\n"
                "}\n"
            )
    return ""


def _fallback_user_text_to_json(user_text: str) -> dict[str, Any]:
    lowered = (user_text or "").lower()
    folder = _extract_folder_from_text(user_text)

    if "list" in lowered and "file" in lowered:
        return {"action": "list_files", "path": folder or "."}
    if any(token in lowered for token in ("read", "show", "open")) and "file" in lowered:
        filename = _extract_filename_from_text(user_text)
        return {"action": "read_file", "filename": f"{folder}/{filename}" if folder else filename}
    if any(token in lowered for token in ("delete", "remove")) and "file" in lowered:
        filename = _extract_filename_from_text(user_text)
        return {"action": "delete_file", "filename": f"{folder}/{filename}" if folder else filename}

    if any(token in lowered for token in ("create", "write")) and "file" in lowered:
        filename = _extract_filename_from_text(user_text)
        path = f"{folder}/{filename}" if folder else filename
        content = _build_fallback_content(user_text, filename)
        return {"action": "write_file", "filename": path, "content": content}

    return _fallback_plan()


def _looks_like_create_request(user_text: str) -> bool:
    lowered = (user_text or "").lower()
    return any(token in lowered for token in ("create", "write", "make", "generate")) and "file" in lowered


def _build_direct_write_plan(user_text: str) -> dict[str, Any] | None:
    if not _looks_like_create_request(user_text):
        return None
    filename = _extract_filename_from_text(user_text)
    folder = _extract_folder_from_text(user_text)
    content = _generate_file_content(user_text, filename)
    if content is None:
        content = _build_fallback_content(user_text, filename)
    elif _looks_like_code_request(user_text):
        content = _self_improve_code(user_text, filename, content)

    if _is_likely_bad_code(content, filename):
        second_try = _generate_file_content(user_text, filename)
        if second_try:
            if _looks_like_code_request(user_text):
                second_try = _self_improve_code(user_text, filename, second_try)
            if not _is_likely_bad_code(second_try, filename):
                content = second_try

    if not content.strip():
        ext = Path(filename).suffix.lower()
        if ext == ".py":
            content = "# Empty placeholder: no valid model output.\n"
        elif ext in {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}:
            content = "// Empty placeholder: no valid model output.\n"
        elif ext == ".c":
            content = "/* Empty placeholder: no valid model output. */\n"
        else:
            content = "New file created."
    path = f"{folder}/{filename}" if folder else filename
    return {"action": "write_file", "filename": path, "content": content}


def safe_resolve_path(relative_path: str) -> Path:
    if not relative_path or not isinstance(relative_path, str):
        raise ValueError("Path is required")

    normalized = relative_path.replace("\\", "/").strip()
    if not normalized:
        raise ValueError("Path is required")

    # If planner includes workspace prefix, map it back to the canonical root.
    lowered = normalized.lower().lstrip("./")
    if lowered == "jarvis_brain":
        normalized = "."
    elif lowered.startswith("jarvis_brain/"):
        normalized = normalized.split("/", 1)[1]

    if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        raise ValueError("Invalid path")

    candidate = (BASE_DIR / normalized).resolve()
    if candidate != BASE_DIR and BASE_DIR not in candidate.parents:
        raise ValueError("Invalid path")
    return candidate


def _format_file_if_supported(target: Path) -> bool:
    suffix = target.suffix.lower()
    if suffix in {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"}:
        if shutil.which("clang-format"):
            completed = subprocess.run(
                ["clang-format", "-i", str(target)],
                capture_output=True,
                text=True,
            )
            return completed.returncode == 0
    if suffix == ".py":
        if shutil.which("ruff"):
            completed = subprocess.run(
                ["ruff", "format", str(target)],
                capture_output=True,
                text=True,
            )
            return completed.returncode == 0
        if shutil.which("black"):
            completed = subprocess.run(
                ["black", "--quiet", str(target)],
                capture_output=True,
                text=True,
            )
            return completed.returncode == 0
    return False


def create_file(filename: str, content: str) -> dict[str, Any]:
    ext = Path(filename).suffix.lower()
    if ext in CODE_FILE_SUFFIXES:
        cleaned = _extract_code_from_llm_output(content or "", filename) or _finalize_code_content(
            content or "", filename
        )
        if _contains_generation_chaff(cleaned):
            return _error("invalid_generated_content", "Output looks like prose or instructions, not source code.")
        if not _validate_code_by_extension(cleaned, filename):
            return _error("invalid_syntax", "Generated content failed syntax validation.")
        content = cleaned

    target = safe_resolve_path(filename)
    if target.parent.resolve() == BASE_DIR.resolve() and target.name in _PROTECTED_ROOT_FILENAMES:
        return _error(
            "protected_file",
            "Refusing to overwrite core backend files (api.py / jarvis.py / llm_provider.py).",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content or "", encoding="utf-8")
    was_formatted = _format_file_if_supported(target)
    return _ok({"action": "create_file", "filename": filename, "formatted": was_formatted})


def _extract_text_from_pdf(path: Path) -> str:
    if pypdf is None:
        return "[Error: pypdf library not installed. Cannot read PDF.]"
    try:
        reader = pypdf.PdfReader(path)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text).strip()
    except Exception as e:
        return f"[Error extracting PDF text: {e}]"


def read_file(filename: str) -> dict[str, Any]:
    try:
        target = safe_resolve_path(filename)
    except ValueError:
        target = None

    if target is None or not target.exists() or not target.is_file():
        # Fuzzy search: if the specific path fails, look for the filename anywhere in the workspace
        basename = Path(filename).name
        matches = list(BASE_DIR.rglob(basename))
        if matches:
            target = matches[0]
            filename = str(target.relative_to(BASE_DIR)).replace("\\", "/")
        else:
            return _error("missing_file", f"File '{filename}' not found.")

    # Handle PDF
    if target.suffix.lower() == ".pdf":
        content = _extract_text_from_pdf(target)
        if not content:
            content = "[PDF file appears to have no extractable text content.]"
    else:
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = "[Binary file content — cannot be displayed as text.]"

    return _ok({"action": "read_file", "filename": filename, "content": content})


def delete_file(filename: str) -> dict[str, Any]:
    target = safe_resolve_path(filename)
    if target.parent.resolve() == BASE_DIR.resolve() and target.name in _PROTECTED_ROOT_FILENAMES:
        return _error("protected_file", "Cannot delete core backend modules.")
    if not target.exists() or not target.is_file():
        return _error("missing_file", "File not found")
    target.unlink()
    return _ok({"action": "delete_file", "filename": filename})


def rename_file(old_filename: str, new_filename: str) -> dict[str, Any]:
    old_target = safe_resolve_path(old_filename)
    new_target = safe_resolve_path(new_filename)

    if old_target.parent.resolve() == BASE_DIR.resolve() and old_target.name in _PROTECTED_ROOT_FILENAMES:
        return _error("protected_file", "Cannot rename core backend modules.")
    if not old_target.exists():
        return _error("missing_file", f"Source file '{old_filename}' not found.")
    if new_target.exists():
        return _error("file_exists", f"Destination '{new_filename}' already exists.")

    new_target.parent.mkdir(parents=True, exist_ok=True)
    old_target.rename(new_target)
    return _ok({"action": "rename_file", "old_filename": old_filename, "new_filename": new_filename})


def edit_file(filename: str, content: str) -> dict[str, Any]:
    # For now, edit_file is essentially a safe write_file.
    return create_file(filename, content)


def list_files(path: str = ".") -> dict[str, Any]:
    target = safe_resolve_path(path)
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        return _error("invalid_path", "Path is not a directory")
    files = sorted(str(item.relative_to(BASE_DIR)).replace("\\", "/") for item in target.iterdir())
    return _ok({"action": "list_files", "path": path, "files": files})


def create_folder(path: str) -> dict[str, Any]:
    target = safe_resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return _ok({"action": "create_folder", "path": path})


def _enforce_extension(filename: str, language: str | None) -> str:
    if not language or not isinstance(language, str):
        return filename
    
    expected_ext = LANGUAGE_EXT_MAP.get(language.lower())
    if not expected_ext:
        return filename
    
    p = Path(filename)
    stem = p.name
    # Strip all extensions
    while "." in stem:
        stem = Path(stem).stem
        
    new_name = f"{stem}{expected_ext}"
    return str(p.parent / new_name).replace("\\", "/")


def _extract_balanced_json_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        in_string = False
        escaped = False
        for idx, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == opener:
                if depth == 0:
                    start = idx
                depth += 1
            elif char == closer and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunks.append(text[start : idx + 1])
    return chunks


def _candidate_json_strings(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []

    candidates: list[str] = [text]

    # Remove markdown fences, if present.
    fence_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if fence_clean:
        candidates.append(fence_clean)

    # Add balanced JSON-like chunks from all candidate variants.
    expanded: list[str] = []
    for value in candidates:
        expanded.append(value)
        expanded.extend(_extract_balanced_json_chunks(value))

    # Keep order while deduplicating.
    seen: set[str] = set()
    unique: list[str] = []
    for item in expanded:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r"^```(?:json)?\s*|\s*```$", "", repaired, flags=re.IGNORECASE | re.DOTALL).strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # remove trailing commas
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)  # quote keys
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', repaired)
    return repaired


def _parse_planner_output(raw: str) -> list[dict[str, Any]]:
    """Extract one or more JSON objects from LLM output."""
    candidates = _candidate_json_strings(raw)
    parsed_items = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                parsed_items.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        parsed_items.append(item)
        except Exception:
            pass

        if not parsed_items:
            repaired = _repair_json_text(candidate)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    parsed_items.append(parsed)
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            parsed_items.append(item)
            except Exception:
                pass

        if parsed_items:
            break

    if not parsed_items:
        # Final attempt with literal_eval
        for candidate in candidates:
            try:
                repaired = _repair_json_text(candidate)
                literal = literal_eval(repaired)
                if isinstance(literal, dict):
                    parsed_items.append(json.loads(json.dumps(literal)))
                    break
                elif isinstance(literal, list):
                    parsed_items.extend(json.loads(json.dumps(literal)))
                    break
            except Exception:
                pass
    return parsed_items


def _normalize_plan(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _fallback_plan()

    action = payload.get("action")
    if action not in ALLOWED_ACTIONS:
        return _fallback_plan()

    normalized: dict[str, Any] = {"action": action}
    if action in {"create_file", "write_file", "read_file", "delete_file"}:
        filename = payload.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return _fallback_plan()
        normalized["filename"] = filename.strip()

    if action in {"create_file", "write_file"}:
        content = payload.get("content", "")
        raw_content = content if isinstance(content, str) else str(content)
        normalized["content"] = _finalize_code_content(raw_content, normalized["filename"])
    elif action == "list_files":
        path = payload.get("path", ".")
        normalized["path"] = path.strip() if isinstance(path, str) and path.strip() else "."

    return normalized


def _generate_file_content(user_text: str, filename: str) -> str | None:
    prompts = (
        f"""You emit a single source file.

Target filename: {filename}

Rules:
- Put the entire program inside one markdown ``` code fence (language tag optional).
- No JSON, no tutorial text, no "Instruction" sections, no follow-up questions.
- VALID SYNTAX ONLY.
- Generate ONLY raw code, no explanation or commentary.

User request:
{user_text}
""".strip(),
        f"""Emit ONLY one fenced code block for `{filename}`. Absolutely nothing outside the fence. No commentary.

Task: {user_text}
""".strip(),
    )
    for prompt in prompts:
        try:
            raw = call_llm(prompt)
        except Exception:
            return None
        content = _extract_code_from_llm_output(raw, filename)
        if (
            content
            and content.strip()
            and not _contains_generation_chaff(content)
            and _validate_code_by_extension(content, filename)
        ):
            return content
    return None



# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: INTENT EXTRACTION  (1 LLM call)
# ═══════════════════════════════════════════════════════════════════════════

def extract_intent(user_text: str, state: TaskState | None = None) -> dict[str, Any]:
    """Parse user text into a structured intent. Uses exactly 1 LLM call."""
    update_step(state, "Extracting intent", 10) if state else None

    # Build lightweight context (0 LLM calls)
    context = build_context(user_text)
    context_block = f"\nContext (for reference only):\n{context}\n" if context else ""

    prompt = f"""You are an intent parser. Analyze the user request and return ONLY a JSON object.

Output format (no markdown, no explanation):
{{
  "task_type": "file_creation" | "edit" | "delete" | "read" | "list" | "rename" | "chat" | "unknown",
  "files": [
    {{
      "filename": "exact_filename.ext",
      "language": "python" | "cpp" | "javascript" | etc,
      "description": "what this file should contain"
    }}
  ],
  "requirements": ["requirement1", "requirement2"],
  "folder": null
}}
{context_block}
RULES:
1. ONLY extract filenames that the user EXPLICITLY mentions (e.g. "main.py", "utils.cpp").
2. If user says "create a python file" without naming it, use a descriptive default like "main.py".
3. NEVER use random words from the sentence as filenames.
4. If the user asks for MULTIPLE files, list each one separately.
5. For chat/conversation requests (greetings, questions, etc), use task_type "chat" with empty files.
6. If user mentions "inside X folder" or "in X directory", set "folder" to that path.
7. If user references a file from the workspace context above, use its exact path.

User input: {user_text}"""

    try:
        raw = call_llm(prompt)
        increment_llm_counter(state)
    except Exception:
        return {"task_type": "chat", "files": [], "requirements": []}

    payloads = _parse_planner_output(raw)
    if payloads and isinstance(payloads[0], dict):
        intent = payloads[0]
        # Validate task_type
        valid_types = {"file_creation", "edit", "delete", "read", "list", "rename", "chat", "unknown"}
        if intent.get("task_type") not in valid_types:
            intent["task_type"] = "unknown"
        # Ensure files is a list
        if not isinstance(intent.get("files"), list):
            intent["files"] = []
        # Sanitize filenames — reject single-char garbage
        cleaned_files = []
        for f in intent["files"]:
            if isinstance(f, dict) and f.get("filename"):
                name = str(f["filename"]).strip()
                # Reject single-char or obviously garbage names
                basename = Path(name).stem
                if len(basename) <= 1 and basename not in {"_"}:
                    # Only use a default if it's a creation request
                    if intent.get("task_type") == "file_creation":
                        name = _safe_default_filename(user_text)
                
                f["filename"] = _fix_filename(name, user_text)
                cleaned_files.append(f)
        intent["files"] = cleaned_files
        return intent

    return {"task_type": "chat", "files": [], "requirements": []}


def _safe_default_filename(user_text: str) -> str:
    """Generate a sensible default filename from task context."""
    lowered = (user_text or "").lower()
    if "python" in lowered or ".py" in lowered:
        return "main.py"
    if "cpp" in lowered or "c++" in lowered or ".cpp" in lowered:
        return "main.cpp"
    if ".js" in lowered or "javascript" in lowered:
        return "main.js"
    if ".java" in lowered or "java" in lowered:
        return "Main.java"
    return "main.py"


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: CREATIVE PLAN  (deterministic — 0 LLM calls)
# ═══════════════════════════════════════════════════════════════════════════

def generate_plan(intent: dict[str, Any], state: TaskState | None = None) -> list[dict[str, Any]]:
    """Convert intent into a list of planned steps. Purely deterministic."""
    update_step(state, "Generating plan", 20) if state else None

    task_type = intent.get("task_type", "unknown")
    files = intent.get("files", [])
    folder = intent.get("folder")

    steps: list[dict[str, Any]] = []

    if task_type == "chat":
        # If files were identified in a chat request, read them first for context
        for f in files:
            filename = f.get("filename", "")
            if folder:
                filename = f"{folder}/{filename}"
            steps.append({"action": "read_file", "filename": filename, "for_context": True})
        
        steps.append({"action": "chat", "description": "Respond conversationally using file context if provided"})
        return steps

    if folder:
        prefixed_folder = _prefix_created_files(folder)
        steps.append({"action": "create_folder", "path": prefixed_folder})

    if task_type == "list":
        path = folder or "created_files"
        steps.append({"action": "list_files", "path": path})
        return steps

    if task_type in ("file_creation", "edit"):
        for f in files:
            filename = f.get("filename", "")
            description = f.get("description", "")
            if folder:
                filename = f"{folder}/{filename}"
            # STRICT: route into created_files/
            filename = _prefix_created_files(filename)
            action = "create_file" if task_type == "file_creation" else "edit_file"
            steps.append({
                "action": action,
                "filename": filename,
                "language": f.get("language"),
                "description": description,
                "needs_content": True,
            })
        # If no files extracted but it's a creation request, create a default
        if not steps or (len(steps) == 1 and steps[0].get("action") == "create_folder"):
            default_name = _prefix_created_files(
                (folder + "/main.py") if folder else "main.py"
            )
            steps.append({
                "action": "create_file",
                "filename": default_name,
                "description": "User requested file creation but no specific file was identified",
                "needs_content": True,
            })
        return steps

    if task_type == "delete":
        for f in files:
            filename = f.get("filename", "")
            if folder:
                filename = f"{folder}/{filename}"
            steps.append({"action": "delete_file", "filename": filename})
        return steps

    if task_type == "read":
        for f in files:
            filename = f.get("filename", "")
            if folder:
                filename = f"{folder}/{filename}"
            steps.append({"action": "read_file", "filename": filename})
        return steps

    if task_type == "rename":
        if len(files) >= 2:
            steps.append({
                "action": "rename_file",
                "old_filename": files[0].get("filename", ""),
                "new_filename": files[1].get("filename", ""),
            })
        return steps

    # Unknown — treat as chat
    steps.append({"action": "chat", "description": "Could not determine intent"})
    return steps


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: ACTION COMPILER  (1 LLM call per file that needs content)
# ═══════════════════════════════════════════════════════════════════════════

def compile_plan(plan: list[dict[str, Any]], user_text: str,
                 state: TaskState | None = None) -> list[dict[str, Any]]:
    """Compile plan steps into executable actions (metadata only)."""
    update_step(state, "Compiling actions", 40) if state else None

    compiled: list[dict[str, Any]] = []

    for step in plan:
        action = step.get("action")
        if action in ("create_file", "edit_file"):
            filename = step.get("filename", "")
            language = step.get("language")
            # Step 2: Fix filename (extension check)
            if language:
                filename = _enforce_extension(filename, language)
            
            compiled.append({
                "action": action,
                "filename": filename,
                "language": language,
                "description": step.get("description", user_text),
                "needs_content": step.get("needs_content", False)
            })
        else:
            compiled.append(step)

    return compiled


def generate_code(filename: str, language: str | None, description: str, 
                  state: TaskState | None = None) -> str:
    """Stage 3: Dedicated high-quality code generation (1 LLM call per file)."""
    lang_str = language if language else "text"
    prompt = f"""Write complete, working, production-quality code for the following file.

Filename: {filename}
Language: {lang_str}
Description: {description}

RULES:
- Output ONLY the code inside a markdown code block.
- Do NOT output placeholder text.
- Do NOT explain anything.
- Only output code.
- Ensure all necessary imports/includes are present.
- If it's a standalone script, include a main function or entry point.
"""
    try:
        raw = call_llm(prompt)
        increment_llm_counter(state)
        content = _extract_code_from_llm_output(raw, filename)
        
        # Step 4: Quality Validation
        is_garbage = False
        if not content or len(content.strip()) < 50:
            is_garbage = True
        elif any(token in content.lower() for token in ("placeholder", "todo", "insert code here")):
            is_garbage = True
            
        if is_garbage:
            # Retry once with stricter prompt
            log_event(state, f"Code for {filename} rejected (garbage detected). Retrying...") if state else None
            strict_prompt = prompt + "\n\nCRITICAL: Your previous output was rejected for being incomplete or a placeholder. Write the FULL, functional implementation now."
            raw = call_llm(strict_prompt)
            increment_llm_counter(state)
            content = _extract_code_from_llm_output(raw, filename)
            
        return content or ""
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# TOOL CREATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

TOOLS_DIR = BASE_DIR / "tools"
_tool_registry: dict[str, str] = {}  # tool_name -> module path


def _create_tool(tool_name: str, purpose: str, spec: str,
                 state: TaskState | None = None) -> dict[str, Any]:
    """Generate, validate, and register a new tool."""
    log_event(state, f"Creating tool: {tool_name}") if state else None

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tool_file = TOOLS_DIR / f"{tool_name}.py"

    prompt = f"""Create a Python tool module. Output ONLY a fenced code block.

Tool name: {tool_name}
Purpose: {purpose}
Specification: {spec}

Rules:
- Must define a function called `run(**kwargs) -> dict`
- Must operate only on files within the current working directory
- No system-level operations, no network calls
- Include a docstring explaining what the tool does
"""
    try:
        raw = call_llm(prompt)
        increment_llm_counter(state)
    except Exception as exc:
        return _error("tool_generation_failed", str(exc))

    code = _extract_code_from_llm_output(raw, str(tool_file))
    if not code or not _validate_python_source(code):
        return _error("tool_validation_failed", "Generated tool code failed syntax check")

    # Import test
    try:
        compile(code, str(tool_file), "exec")
    except SyntaxError as exc:
        return _error("tool_compile_failed", str(exc))

    tool_file.write_text(code, encoding="utf-8")
    _tool_registry[tool_name] = str(tool_file)
    log_event(state, f"Tool '{tool_name}' created and registered") if state else None
    return _ok({"action": "create_tool", "tool_name": tool_name, "path": str(tool_file)})


# ═══════════════════════════════════════════════════════════════════════════
# STEP-BASED EXECUTOR  (no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════

def execute_action(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action")
    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "create_file": lambda p: create_file(p.get("filename", ""), p.get("content", "")),
        "write_file": lambda p: create_file(p.get("filename", ""), p.get("content", "")),
        "edit_file": lambda p: edit_file(p.get("filename", ""), p.get("content", "")),
        "rename_file": lambda p: rename_file(p.get("old_filename", ""), p.get("new_filename", "")),
        "read_file": lambda p: read_file(p.get("filename", "")),
        "delete_file": lambda p: delete_file(p.get("filename", "")),
        "list_files": lambda p: list_files(p.get("path", ".")),
        "create_tool": lambda p: _create_tool(
            p.get("tool_name", ""), p.get("purpose", ""), p.get("spec", "")
        ),
        "create_folder": lambda p: create_folder(p.get("path", ".")),
    }

    if action not in handlers:
        return _error("unknown_action", f"Action '{action}' is not supported")

    try:
        return handlers[action](payload)
    except ValueError as exc:
        return _error("invalid_path", str(exc))
    except Exception as exc:
        return _error("execution_error", str(exc))


def _format_user_response(action_payload: dict[str, Any], result: dict[str, Any]) -> str:
    action = action_payload.get("action", "")
    if not result.get("ok"):
        error = result.get("error", {})
        message = error.get("message", "Unknown error")
        return f"I could not complete that request: {message}."

    data = result.get("data", {})
    if action in {"create_file", "write_file", "edit_file"}:
        filename = data.get('filename', 'unknown')
        if data.get("formatted"):
            return f"Processed and formatted the file {filename} successfully."
        return f"File {filename} written successfully."
    if action == "rename_file":
        return f"Renamed {data.get('old_filename')} to {data.get('new_filename')} successfully."
    if action == "read_file":
        filename = data.get("filename", "unknown")
        content = data.get("content", "")
        if not content:
            return f"The file {filename} is empty."
        return f"Contents of {filename}:\n{content}"
    if action == "delete_file":
        return f"Deleted the file {data.get('filename', 'unknown')} successfully."
    if action == "list_files":
        files = data.get("files", [])
        path = data.get("path", ".")
        if not files:
            return f"No files found in {path}."
        file_list = "\n".join(f"- {name}" for name in files)
        return f"Files in {path}:\n{file_list}"
    if action == "create_tool":
        return f"Created tool '{data.get('tool_name', 'unknown')}' successfully."
    if action == "create_folder":
        return f"Created folder '{data.get('path', 'unknown')}' successfully."
    return "Done."


def _execute_steps(actions: list[dict[str, Any]], user_text: str,
                   state: TaskState) -> dict[str, Any]:
    """Execute compiled actions with step-by-step progress tracking."""
    total = len(actions)
    results = []
    replies = []
    all_ok = True

    # Identify files needing content for progress reporting
    needing_content = [a for a in actions if a.get("needs_content")]
    content_idx = 0
    context_data = []

    for idx, action in enumerate(actions, 1):
        step_label = f"Step {idx}/{total}: Executing {action.get('action', '?')}"
        if action.get("filename"):
            step_label += f" — {action['filename']}"
        elif action.get("path"):
            step_label += f" — {action['path']}"
            
        pct = 60 + (40 * idx / max(total, 1))
        update_step(state, step_label, pct)

        if action.get("action") == "chat":
            try:
                full_input = user_text
                if context_data:
                    full_input = "Context from files:\n" + "\n\n".join(context_data) + f"\n\nUser request: {user_text}"
                
                reply = _chat_reply(full_input)
                increment_llm_counter(state)
                results.append(_ok({"action": "chat", "reply": reply}))
                replies.append(reply)
            except Exception as exc:
                results.append(_error("llm_error", str(exc)))
                replies.append(f"LLM Error: {exc}")
                all_ok = False
            continue

        # Step 3: Dedicated Code Generation (if needed)
        if action.get("needs_content"):
            content_idx += 1
            log_event(state, f"Generating code for {action['filename']}...")
            content = generate_code(
                action["filename"], 
                action.get("language"), 
                action.get("description", user_text),
                state
            )
            action["content"] = content

        res = execute_action(action)
        
        # If this was a context-read, store it for the chat step
        if action.get("action") == "read_file" and action.get("for_context"):
            if res.get("ok"):
                context_data.append(f"File: {action['filename']}\n{res['data']['content']}")
                replies.append(f"Read {action['filename']} for context.")
            else:
                replies.append(f"Failed to read {action['filename']} for context.")
            results.append(res)
            continue

        results.append(res)
        replies.append(_format_user_response(action, res))
        if not res.get("ok"):
            all_ok = False

    return {
        "ok": all_ok,
        "reply": "\n".join(replies),
        "result": results if len(results) > 1 else (results[0] if results else _ok({})),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — replaces old run_execute
# ═══════════════════════════════════════════════════════════════════════════

def run_execute(user_text: str, state: TaskState | None = None) -> dict[str, Any]:
    """
    Full 4-stage pipeline:
      1. Intent Extraction (1 LLM call)
      2. Plan Generation (0 LLM calls)
      3. Action Compilation (0 LLM calls — metadata only)
      4. Step-based Execution (1 LLM call per file, except chat)
    """
    stripped = (user_text or "").strip()
    if not stripped:
        return {"ok": True, "reply": "No input provided.", "result": _ok({})}

    # Track user input in conversation buffer
    _add_to_conversation("user", stripped)

    # Create task state if not provided (backward compat with old API)
    if state is None:
        state = create_task()
    mark_running(state)

    try:
        # ── Stage 1: Intent ──
        intent = extract_intent(stripped, state)
        log_event(state, f"Intent: type={intent.get('task_type')}, files={len(intent.get('files', []))}")

        # Quick exit for chat
        if intent.get("task_type") == "chat":
            update_step(state, "Generating chat response", 50)
            try:
                reply = _chat_reply(stripped)
                increment_llm_counter(state)
                _add_to_conversation("assistant", reply)
                _save_task_summary(_generate_task_summary(intent, {"ok": True}))
                result = {"ok": True, "reply": reply, "result": _ok({"action": "chat", "reply": reply})}
                mark_completed(state, result)
                return result
            except Exception as exc:
                mark_failed(state, str(exc))
                return {"ok": False, "reply": f"LLM Error: {exc}", "result": _error("llm_error", str(exc))}

        # ── Stage 2: Plan ──
        plan = generate_plan(intent, state)
        log_event(state, f"Plan: {len(plan)} step(s)")

        # ── Stage 3: Compile ──
        actions = compile_plan(plan, stripped, state)
        log_event(state, f"Compiled: {len(actions)} action(s)")

        # ── Stage 4: Execute ──
        result = _execute_steps(actions, stripped, state)
        mark_completed(state, result)
        result["task_id"] = state.task_id

        # Save to conversation buffer + task memory (deterministic, 0 LLM calls)
        _add_to_conversation("assistant", result.get("reply", "Done.")[:500])
        _save_task_summary(_generate_task_summary(intent, result))

        return result

    except Exception as exc:
        mark_failed(state, str(exc))
        return {
            "ok": False,
            "reply": f"Pipeline error: {exc}",
            "result": _error("pipeline_error", str(exc)),
            "task_id": state.task_id,
        }

