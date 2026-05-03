

# JARVIS — Local AI Agent (Version 1)

A structured local AI agent that converts natural language into controlled, validated file and system actions using a layered architecture.

## What This Is

It is a local agent system that takes natural language input and translates it into real, deterministic operations on your workspace.

It can:

- create, read, modify, and organize files
- generate code and structured content
- handle multi-step instructions
- maintain short-term context
- validate actions before reporting success

The focus is not just generation, but reliable execution.

## The Story

This project started as a simple idea: build a JARVIS-like system where:

```
human intent → structured understanding → controlled action
```

I began with a basic prompt wrapper around a local LLM. It worked for a day. Then the cracks appeared. The model couldn't handle multi-file tasks. It invented invalid filenames. It reported success for operations that silently failed.

The problem wasn't the model. It was the lack of structure.

Switching to Gemini helped with understanding, but it revealed a deeper issue: a better model without better architecture just generates better-looking mistakes. I realized I needed to separate concerns—understanding what you want, planning how to do it, and actually doing it—into distinct stages.

That's when the system transformed from a reactive tool into something predictable and reliable.

## Architecture

The system follows a three-stage pipeline:

```
User Input
   ↓
Intent Extraction (LLM)
   ↓
Planning (LLM)
   ↓
Action Compilation (structured JSON)
   ↓
Execution (Python + filesystem)
   ↓
Validation (system-level checks)
   ↓
Response
```

Each stage has a single responsibility. This makes the system predictable, testable, and debuggable.

## Core System Files

```
/project
 ├── jarvis.py          # Core pipeline (intent → plan → execute)
 ├── api.py             # FastAPI interface (OpenAI-compatible)
 ├── llm_provider.py    # LLM abstraction (Gemini / future models)
 ├── task_state.py      # Task tracking and observability
 ├── requirements.txt
 ├── created_files/     # All generated outputs  
 ├── memory/            # Task summaries
 ├── tools/             # Generated tools (optional)
 ├── logs/              # Execution logs   (optional)
```

## How It Works

### Stage 1: Intent Extraction

The LLM converts your natural language input into structured meaning. Instead of guessing, if something is ambiguous, the system asks clarifying questions.

### Stage 2: Planning

The LLM decomposes your task into ordered, atomic steps. This plan is human-readable and can be reviewed before execution begins.

### Stage 3: Action Compilation

The system converts the plan into strict JSON actions. No ambiguity. Each action knows exactly what it needs to do and how to validate success.

### Stage 4: Execution

Python code performs the actual file operations. Not LLM output running directly—structured, validated commands.

### Stage 5: Validation

After each operation, the system verifies the result. If a file was supposed to be created, it checks that the file actually exists. If code was generated, it validates syntax.

```python
if not file_exists:
    fail_task()
```

No hallucinated successes.

## Core Components

### jarvis.py — Core Engine

This is where the system logic lives. It handles:

- intent extraction
- planning
- action compilation
- execution
- validation

All in one coordinated pipeline.

### llm_provider.py — Model Interface

Handles communication with LLMs. Currently supports Gemini as the primary model. Designed to be replaceable—you could swap in local models, NVIDIA endpoints, or other providers without changing the core logic.

### api.py — Interface Layer

Built with FastAPI. It exposes:

- `/execute` endpoint for direct task execution
- `/v1/chat/completions` (OpenAI-compatible) for integration with tools like OpenWebUI

This separation means JARVIS can be used standalone or integrated with your existing tools.

### task_state.py — Observability

Tracks task status, progress, and logs. Gives visibility into what the system is doing and what happened.

## The Context System

Instead of large memory buffers that grow endlessly, JARVIS uses a lightweight context model:

### Conversation Buffer

The last few interactions, kept in memory.

### Workspace Awareness

Scans `created_files/` to understand what exists. No file contents—just the structure and names. Efficient and focused.

### Task Memory

Short summaries stored in `memory/`. Not full transcripts, just what matters for the next request.

This approach keeps context relevant without consuming endless API tokens or memory.

## Problems Solved

Over the course of development, the system evolved to solve specific, real problems:

**Multi-file tasks collapsed into single outputs.** Fixed via a structured action system that handles each file independently.

**Invalid filenames appeared.** Files like `.cpp.py` stopped happening when I enforced language-to-extension mapping.

**Placeholder outputs pretended to be complete.** Per-file generation and validation ensures every claimed output actually exists.

**Context was lost between requests.** A lightweight context system remembers what you've asked for and what was created.

**False success reports.** Filesystem validation means the system only claims success when files actually exist with correct content.

**Root folder clutter.** All outputs go to `created_files/`, keeping your workspace clean.

## Tech Stack

- Python for the core system
- FastAPI for the API layer
- Gemini API for intent extraction and planning
- Local filesystem for execution and state
- OpenWebUI (optional) as a frontend interface

## About OpenWebUI

A clarification: OpenWebUI is not part of the core system. It's optional and serves as a frontend interface. You interact with the API through it, but all logic—all the planning, validation, and execution—happens in `jarvis.py`.

You can use OpenWebUI, a custom UI, a CLI, or integrate JARVIS directly into your application. The core doesn't change.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/yourusername/jarvis.git
cd jarvis
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here
```

Run the API server:

```bash
uvicorn api:app --reload
```

The server starts on `http://localhost:8000`. You can send requests to `/execute`:

```json
POST /execute

{
  "input": "create a python script that implements selection sort"
}
```

## What It Can Do (Version 1)

- multi-file project generation
- multi-language support
- structured execution with validation
- context-aware follow-ups
- filesystem-safe operations
- task tracking and logging

## What It Can't Do (Yet)

- long-term memory (only short-term context)
- execution feedback loops (code is generated, not run and tested)
- autonomous multi-step agent loops (single-pass execution)
- work without an external LLM (depends on Gemini)

## Key Insights

Over the course of building this, certain lessons became clear.

**Weak models hit a ceiling fast.** Local models were useful for prototyping but couldn't handle the complexity of understanding intent and generating reliable code. A cloud model with better training made the difference.

**Structure matters more than prompting.** I spent weeks tuning prompts. Then I spent two days building the three-layer pipeline, and most problems dissolved. The system design solved problems that better prompts never could.

**Balance freedom with constraints.** The planning stage has freedom—it can understand your request in any way and break it into steps however makes sense. The execution stage is strict—only allowed operations, validated paths, deterministic behavior. This balance is where reliability comes from.

**An LLM is just one component.** The model is powerful, but it's not the system. Everything around it—context management, execution validation, state tracking—matters as much.

## What's Next

This is version 1. The next phase includes:

- execution feedback loops (run generated code, fix errors, iterate)
- a plugin system for user-defined tools
- better memory and context management
- multi-step agent loops for complex tasks
- a proper UI beyond API endpoints

## Final Thought

This project started as a simple pipeline: prompt → code → execute. It became something more: intent → plan → execute → validate.

Most of the engineering effort wasn't in the model or prompt engineering. It was in everything around the model—the architecture, the validation, the context management, the state tracking.

If you're building similar systems or experimenting with agent architectures, I'd like to hear your thoughts.

## License

MIT

## Questions?

If you're building similar systems or have feedback on the approach, feel free to open an issue or reach out. This is an active project and I welcome ideas.
