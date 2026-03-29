"""Metrics computation: percentile latencies, cost estimation, quality scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .runner import BenchmarkRun


# ---------------------------------------------------------------------------
# Cost tables (USD per 1K tokens)
# ---------------------------------------------------------------------------
COST_TABLE: dict[str, tuple[float, float]] = {
    # model_prefix: (input_per_1k, output_per_1k)
    # OpenAI - sorted specific before generic to ensure correct prefix matching
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o4-mini": (0.0011, 0.0044),
    "o3-mini": (0.0011, 0.0044),
    "o3": (0.01, 0.04),
    "o1-mini": (0.003, 0.012),
    "o1": (0.015, 0.06),
    # Anthropic
    "claude-opus-4": (0.015, 0.075),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-3.5-sonnet": (0.003, 0.015),
    "claude-3.5-haiku": (0.0008, 0.004),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-sonnet": (0.003, 0.015),
    "claude-opus": (0.015, 0.075),
    # Meta
    "llama-4-maverick": (0.0005, 0.0015),
    "llama-4-scout": (0.00018, 0.0005),
    "llama-3.3": (0.0003, 0.0008),
    "llama-3.1-405b": (0.003, 0.003),
    "llama-3.1-70b": (0.0009, 0.0009),
    "llama-3.1-8b": (0.0001, 0.0001),
    # Mistral
    "mistral-large": (0.002, 0.006),
    "mistral-small": (0.0002, 0.0006),
    # Google
    "gemini-2.5-pro": (0.00125, 0.01),
    "gemini-2.5-flash": (0.00015, 0.0006),
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    # DeepSeek
    "deepseek-r1": (0.00055, 0.0022),
    "deepseek-v3": (0.00027, 0.0011),
}

# Fallback for unknown models
DEFAULT_COST = (0.002, 0.008)


def _lookup_cost(model: str) -> tuple[float, float]:
    """Find cost rates for a model by prefix matching."""
    model_lower = model.lower()
    for prefix, costs in COST_TABLE.items():
        if prefix in model_lower:
            return costs
    return DEFAULT_COST


@dataclass
class LatencyStats:
    """Percentile latency statistics."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass
class RunMetrics:
    """Aggregated metrics for a benchmark run."""

    model: str
    suite: str
    num_prompts: int
    ttft: LatencyStats
    total_latency: LatencyStats
    mean_tps: float
    median_tps: float
    total_prompt_tokens: int
    total_completion_tokens: int
    estimated_cost_usd: float
    mean_quality_score: float  # 0.0 to 1.0


def compute_latency_stats(values: list[float]) -> LatencyStats:
    """Compute percentile statistics from a list of values."""
    arr = np.array(values, dtype=np.float64)
    return LatencyStats(
        p50_ms=round(float(np.percentile(arr, 50)), 1),
        p95_ms=round(float(np.percentile(arr, 95)), 1),
        p99_ms=round(float(np.percentile(arr, 99)), 1),
        mean_ms=round(float(np.mean(arr)), 1),
        min_ms=round(float(np.min(arr)), 1),
        max_ms=round(float(np.max(arr)), 1),
    )


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD."""
    input_rate, output_rate = _lookup_cost(model)
    return round(
        (prompt_tokens / 1000) * input_rate + (completion_tokens / 1000) * output_rate,
        6,
    )


def score_quality(response: str, reference: str) -> float:
    """Score response quality against a reference answer.

    Uses a simple keyword overlap metric (Jaccard-like).
    Returns 0.0 to 1.0. If no reference is provided, returns 1.0.
    """
    if not reference.strip():
        return 1.0

    def tokenize(text: str) -> set[str]:
        return {w.lower().strip(".,!?;:'\"()[]{}") for w in text.split() if len(w) > 1}

    ref_tokens = tokenize(reference)
    resp_tokens = tokenize(response)

    if not ref_tokens:
        return 1.0

    overlap = len(ref_tokens & resp_tokens)
    union = len(ref_tokens | resp_tokens)
    return round(overlap / union, 3) if union > 0 else 0.0


def compute_metrics(run: BenchmarkRun) -> RunMetrics:
    """Compute aggregated metrics from a benchmark run."""
    results = run.results
    if not results:
        raise ValueError("No results to compute metrics from")

    ttft_values = [r.ttft_ms for r in results]
    latency_values = [r.total_latency_ms for r in results]
    tps_values = [r.tokens_per_second for r in results]

    total_prompt = sum(r.prompt_tokens for r in results)
    total_completion = sum(r.completion_tokens for r in results)

    quality_scores = [
        score_quality(r.response_text, r.reference) for r in results
    ]

    return RunMetrics(
        model=run.model,
        suite=run.suite_name,
        num_prompts=len(results),
        ttft=compute_latency_stats(ttft_values),
        total_latency=compute_latency_stats(latency_values),
        mean_tps=round(float(np.mean(tps_values)), 1),
        median_tps=round(float(np.median(tps_values)), 1),
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        estimated_cost_usd=estimate_cost(run.model, total_prompt, total_completion),
        mean_quality_score=round(float(np.mean(quality_scores)), 3),
    )
