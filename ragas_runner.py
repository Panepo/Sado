"""
RAGAS evaluation runner.

Initialises Ollama-backed LLM and embeddings (via the OpenAI-compatible
endpoint + litellm), exposes the metric registry, and provides an async
evaluate function for the FastAPI server.
"""

import logging
import math
import os
from typing import Any

log = logging.getLogger("ragas_runner")

from openai import AsyncOpenAI
from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecisionWithReference,
    ContextPrecisionWithoutReference,
    ContextRecall,
    NoiseSensitivity,
    AnswerRelevancy,
    FactualCorrectness,
    SemanticSimilarity,
    BleuScore,
    RougeScore,
)

# ---------------------------------------------------------------------------
# Lazy singletons – initialised on first call to _get_llm() / _get_embeddings()
# ---------------------------------------------------------------------------

_llm = None
_embeddings = None


def _get_llm():
    global _llm
    if _llm is None:
        base_url = os.environ["OLLAMA_BASE_URL"].rstrip("/")
        model = os.environ["OLLAMA_LLM_MODEL"]
        log.debug("Initialising LLM: model=%s base_url=%s", model, base_url)
        # AsyncOpenAI required — InstructorLLM.agenerate() rejects sync clients
        num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", 8192))
        # Reserve ~4096 tokens for the input prompt; the rest is available for
        # generation.  Thinking models (e.g. qwen3) emit a long <think>…</think>
        # block before the JSON output, so max_tokens must be large enough to
        # contain both.  Deriving from num_ctx keeps things consistent regardless
        # of the configured context size.  Override via OLLAMA_MAX_TOKENS in .env.
        default_max_tokens = max(num_ctx - 4096, 4096)
        max_tokens = int(os.environ.get("OLLAMA_MAX_TOKENS", default_max_tokens))
        client = AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        # Mode is handled internally by RAGAS (_get_instructor_client uses Mode.JSON
        # for OpenAI-compatible providers, which works with Ollama)
        # extra_body passes Ollama-specific options (num_ctx) through the
        # OpenAI-compatible endpoint — Ollama defaults to num_ctx=2048 otherwise.
        # max_tokens caps the generated response length; too-small values cause
        # IncompleteOutputException when Instructor parses the structured output.
        _llm = llm_factory(
            model,
            provider="openai",
            client=client,
            max_tokens=max_tokens,
            extra_body={"options": {"num_ctx": num_ctx}},
        )
    return _llm


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        base_url = os.environ["OLLAMA_BASE_URL"].rstrip("/")
        model = os.environ["OLLAMA_EMBED_MODEL"]
        log.debug("Initialising embeddings: model=%s base_url=%s", model, base_url)
        # LiteLLMEmbeddings implements BaseRagasEmbedding with embed_text() —
        # required by collections metrics (SemanticSimilarity etc.)
        _embeddings = LiteLLMEmbeddings(model=f"ollama/{model}", api_base=base_url)
    return _embeddings


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# Each entry:  id -> {display_name, required_fields, needs_llm, needs_embedding, cls}
METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "faithfulness": {
        "display_name": "Faithfulness",
        "required_fields": ["user_input", "response", "retrieved_contexts"],
        "needs_llm": True,
        "needs_embedding": False,
        "cls": Faithfulness,
    },
    "llm_context_recall": {
        "display_name": "LLM Context Recall",
        "required_fields": ["user_input", "retrieved_contexts", "reference"],
        "needs_llm": True,
        "needs_embedding": False,
        "cls": ContextRecall,
    },
    "llm_context_precision": {
        "display_name": "LLM Context Precision",
        "required_fields": ["user_input", "retrieved_contexts", "reference"],
        "needs_llm": True,
        "needs_embedding": False,
        "cls": ContextPrecisionWithReference,
    },
    "context_precision_without_reference": {
        "display_name": "Context Precision (No Reference)",
        "required_fields": ["user_input", "response", "retrieved_contexts"],
        "needs_llm": True,
        "needs_embedding": False,
        "cls": ContextPrecisionWithoutReference,
    },
    "response_relevancy": {
        "display_name": "Response Relevancy",
        "required_fields": ["user_input", "response"],
        "needs_llm": True,
        "needs_embedding": True,
        "cls": AnswerRelevancy,
    },
    "factual_correctness": {
        "display_name": "Factual Correctness",
        "required_fields": ["response", "reference"],
        "needs_llm": True,
        "needs_embedding": False,
        "cls": FactualCorrectness,
    },
    "noise_sensitivity": {
        "display_name": "Noise Sensitivity",
        "required_fields": ["user_input", "retrieved_contexts", "response", "reference"],
        "needs_llm": True,
        "needs_embedding": False,
        "cls": NoiseSensitivity,
    },
    "semantic_similarity": {
        "display_name": "Semantic Similarity",
        "required_fields": ["response", "reference"],
        "needs_llm": False,
        "needs_embedding": True,
        "cls": SemanticSimilarity,
    },
    "bleu_score": {
        "display_name": "BLEU Score",
        "required_fields": ["response", "reference"],
        "needs_llm": False,
        "needs_embedding": False,
        "cls": BleuScore,
    },
    "rouge_score": {
        "display_name": "ROUGE Score",
        "required_fields": ["response", "reference"],
        "needs_llm": False,
        "needs_embedding": False,
        "cls": RougeScore,
    },
}


def registry_as_list() -> list[dict]:
    """Return the registry in a JSON-serialisable format (no 'cls' key)."""
    result = []
    for metric_id, info in METRIC_REGISTRY.items():
        result.append(
            {
                "id": metric_id,
                "display_name": info["display_name"],
                "required_fields": info["required_fields"],
                "needs_llm": info["needs_llm"],
                "needs_embedding": info["needs_embedding"],
            }
        )
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _build_metrics(metric_ids: list[str]) -> list:
    """Instantiate the requested metric objects, injecting llm/embeddings."""
    metrics = []
    llm_needed = any(METRIC_REGISTRY[m]["needs_llm"] for m in metric_ids if m in METRIC_REGISTRY)
    embed_needed = any(METRIC_REGISTRY[m]["needs_embedding"] for m in metric_ids if m in METRIC_REGISTRY)

    llm = _get_llm() if llm_needed else None
    embeddings = _get_embeddings() if embed_needed else None

    for metric_id in metric_ids:
        if metric_id not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {metric_id!r}")
        info = METRIC_REGISTRY[metric_id]
        kwargs: dict[str, Any] = {}
        if info["needs_llm"]:
            kwargs["llm"] = llm
        if info["needs_embedding"]:
            kwargs["embeddings"] = embeddings
        metrics.append(info["cls"](**kwargs))

    return metrics


def _validate_samples(samples: list[dict], metric_ids: list[str]) -> None:
    """Raise ValueError if any required field is missing from all samples."""
    required: set[str] = set()
    for metric_id in metric_ids:
        if metric_id in METRIC_REGISTRY:
            required.update(METRIC_REGISTRY[metric_id]["required_fields"])

    log.debug("_validate_samples: required_fields=%s sample_keys=%s", required, [list(s.keys()) for s in samples])

    missing_globally: list[str] = []
    for field in required:
        if not any(field in sample for sample in samples):
            missing_globally.append(field)

    if missing_globally:
        log.warning("_validate_samples: missing fields %s for metrics %s", missing_globally, metric_ids)
        raise ValueError(
            f"The following fields are required by the selected metrics but not found in any sample: "
            f"{', '.join(missing_globally)}"
        )


async def run_evaluation(
    samples: list[dict],
    metric_ids: list[str],
) -> dict[str, float | None]:
    """
    Run RAGAS evaluation asynchronously.

    Parameters
    ----------
    samples:
        List of dicts with keys: user_input, retrieved_contexts, response, reference.
    metric_ids:
        List of metric IDs from METRIC_REGISTRY.

    Returns
    -------
    Dict mapping metric name -> score (float), or raises on error.
    """
    if not metric_ids:
        raise ValueError("No metrics selected.")
    if not samples:
        raise ValueError("No samples provided.")

    log.debug("run_evaluation: metric_ids=%s samples_count=%d", metric_ids, len(samples))
    _validate_samples(samples, metric_ids)
    metrics = _build_metrics(metric_ids)

    # Map class → registry info for per-metric field filtering
    cls_to_info: dict[type, dict[str, Any]] = {
        info["cls"]: info for info in METRIC_REGISTRY.values()
    }

    scores_by_name: dict[str, list[float | None]] = {m.name: [] for m in metrics}

    for sample in samples:
        for metric in metrics:
            info = cls_to_info[type(metric)]
            required = info["required_fields"]
            filtered = {k: sample[k] for k in required if k in sample}
            log.debug("ascore %s with fields %s", metric.name, list(filtered.keys()))
            result = await metric.ascore(**filtered)
            value = result.value if result is not None else None
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                scores_by_name[metric.name].append(float(value))
            else:
                scores_by_name[metric.name].append(None)

    final: dict[str, float | None] = {}
    for name, values in scores_by_name.items():
        valid = [v for v in values if v is not None]
        final[name] = sum(valid) / len(valid) if valid else None

    log.debug("run_evaluation scores: %s", final)
    return final
