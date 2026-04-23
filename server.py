"""
RAGAS FastAPI server.

Start with:
    uvicorn server:app --reload
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

import ragas_runner as runner

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ragas_server")

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------

load_dotenv()

_REQUIRED_ENV = ("OLLAMA_BASE_URL", "OLLAMA_LLM_MODEL", "OLLAMA_EMBED_MODEL")


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file."
        )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAGAS Evaluation Server", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log 422 validation errors with the raw request body for easier debugging."""
    try:
        body = await request.body()
        body_text = body.decode("utf-8", errors="replace")
    except Exception:
        body_text = "<unreadable>"
    log.warning(
        "422 Unprocessable Entity on %s %s\n  errors: %s\n  body: %s",
        request.method,
        request.url.path,
        exc.errors(),
        body_text,
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Serve static files (the browser UI)
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/metrics")
async def get_metrics():
    """Return the full metric registry."""
    return runner.registry_as_list()


# ---- Single evaluation ----

class SingleEvalRequest(BaseModel):
    user_input: str = ""
    retrieved_contexts: list[str] = []
    response: str = ""
    reference: str = ""
    metrics: list[str]

    @field_validator("metrics")
    @classmethod
    def metrics_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one metric must be selected.")
        return v


@app.post("/api/evaluate/single")
async def evaluate_single(req: SingleEvalRequest):
    log.debug(
        "evaluate_single called | metrics=%s user_input=%r response=%r "
        "retrieved_contexts_count=%d reference=%r",
        req.metrics,
        req.user_input[:120] if req.user_input else "",
        req.response[:120] if req.response else "",
        len(req.retrieved_contexts),
        req.reference[:120] if req.reference else "",
    )

    sample = {
        "user_input": req.user_input,
        "retrieved_contexts": req.retrieved_contexts,
        "response": req.response,
        "reference": req.reference,
    }
    # Remove empty optional fields so RAGAS doesn't complain about blank strings
    sample = {k: v for k, v in sample.items() if v != "" and v != []}
    log.debug("sample fields after stripping empties: %s", list(sample.keys()))

    try:
        scores = await runner.run_evaluation([sample], req.metrics)
    except ValueError as exc:
        log.warning("evaluate_single 422 ValueError: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("evaluate_single 500 unexpected error")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {exc}")

    log.debug("evaluate_single scores: %s", scores)
    return {"scores": scores}


# ---- Batch evaluation ----

@app.post("/api/evaluate/batch")
async def evaluate_batch(
    file: Annotated[UploadFile, File(description="JSON array or CSV file of evaluation samples")],
    metrics: Annotated[str, Form(description="JSON-encoded list of metric IDs")],
):
    # Parse metrics
    try:
        metric_ids: list[str] = json.loads(metrics)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="'metrics' must be a JSON array string.")

    if not metric_ids:
        raise HTTPException(status_code=422, detail="At least one metric must be selected.")

    # Parse uploaded file
    filename = file.filename or ""
    content = await file.read()

    try:
        if filename.endswith(".json"):
            samples: list[dict] = json.loads(content)
            if not isinstance(samples, list):
                raise ValueError("JSON file must be an array of objects.")
        elif filename.endswith(".csv"):
            import io
            df = pd.read_csv(io.BytesIO(content))
            # Convert retrieved_contexts column from JSON string to list if needed
            if "retrieved_contexts" in df.columns:
                df["retrieved_contexts"] = df["retrieved_contexts"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else [x]
                )
            samples = df.where(pd.notna(df), None).to_dict(orient="records")
            # Remove None values from each sample
            samples = [{k: v for k, v in s.items() if v is not None} for s in samples]
        else:
            raise ValueError("Only .json and .csv files are supported.")
    except (ValueError, json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"File parse error: {exc}")

    if not samples:
        raise HTTPException(status_code=422, detail="File contains no samples.")

    # Run evaluation per sample so we can associate scores with the source row
    results = []
    try:
        for sample in samples:
            scores = await runner.run_evaluation([sample], metric_ids)
            results.append({"sample": sample, "scores": scores})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {exc}")

    return {"results": results}
