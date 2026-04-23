# Project Guidelines

## Overview

This is a **RAGAS evaluation server** — a FastAPI backend with a browser-based UI that evaluates RAG (Retrieval-Augmented Generation) outputs using [RAGAS](https://docs.ragas.io/) metrics. The LLM and embedding provider is **Ollama** (local), configured via `.env`.

## Architecture

```
server.py         — FastAPI app, REST endpoints, static file mounting
ragas_runner.py   — Ollama LLM/embedding setup, metric registry, async evaluate()
static/index.html — Single-page UI (vanilla JS, no build step)
install_dependency.py — pip install helper; run this to install all deps
.env              — Runtime config (gitignored); copy from .env.example
.env.example      — Template: OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL
```

## Build & Run

```bash
# Install dependencies
python install_dependency.py

# Start the server (requires Ollama running locally)
uvicorn server:app --reload

# Open browser
http://localhost:8000
```

## Key Conventions

### Ollama / LLM integration
- Ollama is accessed via its **OpenAI-compatible endpoint** (`/v1`) using the `openai` Python client.
- Always use `instructor.Mode.MD_JSON` when constructing the LLM via `llm_factory` — Ollama does **not** support `response_format`, and `MD_JSON` avoids that error.
- Embeddings are wired through `litellm` with the `ollama/<model>` prefix, pointing at `OLLAMA_BASE_URL`.
- LLM and embedding instances are **lazy singletons** in `ragas_runner.py` — initialised on first use, not at import time.

### RAGAS metric registry
- All supported metrics live in `METRIC_REGISTRY` in `ragas_runner.py` as a plain dict.
- Each entry declares `required_fields`, `needs_llm`, and `needs_embedding` — the UI reads these from `GET /api/metrics` to drive checkboxes and field hints.
- When adding a new metric, update only `METRIC_REGISTRY`; `server.py` and `index.html` pick it up automatically.

### API
- `GET /api/metrics` — returns the registry (no `cls` key) as a JSON array.
- `POST /api/evaluate/single` — JSON body; `reference` is optional; returns `{"scores": {...}}`.
- `POST /api/evaluate/batch` — multipart form (`file` + `metrics` JSON string); accepts `.json` (array of objects) or `.csv`; `retrieved_contexts` CSV column must be a JSON array string.
- Validation errors → HTTP 422; unexpected evaluation errors → HTTP 500 with detail string.

### Frontend
- No framework, no build step — one self-contained `static/index.html` with inline CSS and vanilla JS.
- Metric checkboxes are rendered dynamically from `/api/metrics` on page load.
- `retrieved_contexts` in the Single tab is entered as **one context chunk per line** (split on `\n`).

### Environment / secrets
- `.env` is gitignored. Never hardcode credentials or Ollama URLs.
- Always validate required env vars at startup (see `lifespan` in `server.py`).

### Python style
- Python 3.11 / 3.12 recommended (see `install_dependency.py` warning for 3.13+).
- Use `async`/`await` for endpoints; blocking RAGAS `evaluate()` calls are offloaded with `asyncio.to_thread`.
- No external test runner is configured yet — add tests under `tests/` if needed.
