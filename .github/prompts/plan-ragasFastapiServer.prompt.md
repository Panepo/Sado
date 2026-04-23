# Plan: RAGAS FastAPI Server with Ollama + Browser UI

A FastAPI server wrapping RAGAS evaluation, using Ollama as LLM/embedding provider (config via `.env`), serving a minimal single-page HTML/JS UI for single-entry and batch evaluation with all RAGAS metrics selectable as checkboxes.

---

## File Structure

```
d:\Github\Sado\
├── install_dependency.py   (extend with new packages)
├── .env.example            (new – Ollama config template)
├── .env                    (new – user fills actual values)
├── server.py               (new – FastAPI app)
├── ragas_runner.py         (new – evaluation logic + metric registry)
└── static/
    └── index.html          (new – browser UI)
```

---

## Steps

### Phase 1 – Config & Deps

1. Create `.env.example` with `OLLAMA_BASE_URL`, `OLLAMA_LLM_MODEL`, `OLLAMA_EMBED_MODEL`
2. Create `.env` (copy of above, user fills values; add to `.gitignore`)
3. Extend `install_dependency.py`'s `pip_install()` to install:
   `fastapi uvicorn[standard] python-dotenv ragas openai instructor litellm python-multipart pandas`

### Phase 2 – `ragas_runner.py`

4. LLM setup: `OpenAI(api_key="ollama", base_url=f"{OLLAMA_BASE_URL}/v1")` → `llm_factory(model, provider="openai", client=client, mode=instructor.Mode.MD_JSON)`
   — `MD_JSON` mode avoids the `response_format` error Ollama throws
5. Embeddings: `embedding_factory("litellm", model=f"ollama/{OLLAMA_EMBED_MODEL}", api_base=OLLAMA_BASE_URL)`
6. Metric registry dict — 9 metrics, each with `required_fields`, `needs_llm`, `needs_embedding`:

   | ID | Class | Required Fields | LLM | Embed |
   |----|-------|-----------------|-----|-------|
   | `faithfulness` | `Faithfulness` | response, retrieved_contexts | yes | no |
   | `llm_context_recall` | `LLMContextRecall` | user_input, retrieved_contexts, reference | yes | no |
   | `llm_context_precision` | `LLMContextPrecision` | user_input, retrieved_contexts, reference | yes | no |
   | `response_relevancy` | `ResponseRelevancy` | user_input, response, retrieved_contexts | yes | yes |
   | `factual_correctness` | `FactualCorrectness` | response, reference | yes | no |
   | `noise_sensitivity` | `NoiseSensitivity` | user_input, retrieved_contexts, response, reference | yes | no |
   | `semantic_similarity` | `SemanticSimilarity` | response, reference | no | yes |
   | `bleu_score` | `BleuScore` | response, reference | no | no |
   | `rouge_score` | `RougeScore` | response, reference | no | no |

7. `async run_evaluation(samples, metric_ids)` → builds `EvaluationDataset`, instantiates metrics with injected llm/embed, calls `evaluate()` in a threadpool (`asyncio.to_thread`), returns score dict

### Phase 3 – `server.py`

8. Load `.env` on startup via `python-dotenv`; mount `static/` dir for the HTML file
9. `GET /api/metrics` → returns the full metric registry as JSON
10. `POST /api/evaluate/single` (JSON body):
    ```json
    {
      "user_input": "...",
      "retrieved_contexts": ["...", "..."],
      "response": "...",
      "reference": "...",
      "metrics": ["faithfulness", "llm_context_recall"]
    }
    ```
    Response: `{"scores": {"faithfulness": 0.85, ...}}`
11. `POST /api/evaluate/batch` (multipart form: `file` + `metrics` JSON string):
    - Parse `.json` (array of objects) or `.csv` (columns = field names) by file extension
    - Response: `{"results": [{"sample": {...}, "scores": {...}}, ...]}`

### Phase 4 – `static/index.html`

12. Single HTML file, inline CSS + vanilla JS:
    - Fetches `/api/metrics` on load → renders checkbox panel grouped by LLM / Embedding / Traditional
    - Shows required fields per metric as a tooltip
    - **Single tab**: form fields — `user_input` (textarea), `retrieved_contexts` (textarea, newline-separated), `response` (textarea), `reference` (textarea, optional) → submit → result score table
    - **Batch tab**: file drop/upload area, format hint (JSON array or CSV), submit → results table with all rows and expandable raw sample
    - Loading spinner + error toast for both tabs

---

## Relevant Files

- [install_dependency.py](install_dependency.py) — extend `pip_install()` call
- `.env.example` — new Ollama config template
- `.env` — new, user fills; gitignored
- `server.py` — new FastAPI entrypoint
- `ragas_runner.py` — new, core evaluation + metric registry
- `static/index.html` — new, entire browser UI

---

## Verification

1. `python install_dependency.py` completes without errors
2. `uvicorn server:app --reload` → `http://localhost:8000` loads the UI
3. `GET /api/metrics` returns all 9 metric entries with metadata
4. Submit single form with Faithfulness selected → score displays in browser
5. Upload a JSON batch file → per-row score table renders
6. Shut down Ollama, run evaluation → structured error returned (no 500 crash)

---

## Decisions & Notes

- `Mode.MD_JSON` for Instructor adapter — Ollama rejects `response_format`, so markdown-wrapped JSON is used instead
- Embeddings via litellm (`ollama/<model>`) — no custom HTTP client needed
- `reference` is optional in the UI; metrics requiring it surface a clear validation error
- Batch accepts both JSON (array) and CSV; format detected by file extension
- No build step — entire frontend in one `index.html` served as a static file
- `.env` added to `.gitignore` to avoid leaking config
