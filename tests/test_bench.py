"""Tests for bench-my-llm."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bench_my_llm.prompts import get_suite, SUITES
from bench_my_llm.metrics import (
    compute_latency_stats,
    estimate_cost,
    score_quality,
    compute_metrics,
)
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------
class TestPrompts:
    def test_get_suite_reasoning(self):
        suite = get_suite("reasoning")
        assert suite.name == "reasoning"
        assert len(suite.prompts) == 5

    def test_get_suite_all(self):
        suite = get_suite("all")
        assert len(suite.prompts) == 20

    def test_get_suite_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown suite"):
            get_suite("nonexistent")

    def test_all_suites_have_prompts(self):
        for name, suite in SUITES.items():
            assert len(suite.prompts) >= 5, f"Suite {name} has fewer than 5 prompts"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_compute_latency_stats(self):
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = compute_latency_stats(values)
        assert stats.p50_ms == 300.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 500.0
        assert stats.mean_ms == 300.0

    def test_estimate_cost_known_model(self):
        cost = estimate_cost("gpt-4o", 1000, 500)
        # gpt-4o: $0.0025/1K input, $0.01/1K output
        expected = (1000 / 1000) * 0.0025 + (500 / 1000) * 0.01
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_unknown_model(self):
        cost = estimate_cost("some-random-model", 1000, 1000)
        # Should use default rates
        assert cost > 0

    def test_score_quality_exact_match(self):
        score = score_quality("The answer is 42", "The answer is 42")
        assert score == 1.0

    def test_score_quality_no_reference(self):
        score = score_quality("anything here", "")
        assert score == 1.0

    def test_score_quality_partial_overlap(self):
        score = score_quality(
            "Python lists are mutable containers",
            "Lists are mutable and tuples are immutable",
        )
        assert 0.0 < score < 1.0

    def test_compute_metrics_full(self):
        run = _make_dummy_run()
        metrics = compute_metrics(run)
        assert metrics.model == "test-model"
        assert metrics.num_prompts == 3
        assert metrics.mean_tps > 0
        assert metrics.estimated_cost_usd >= 0


# ---------------------------------------------------------------------------
# Runner / serialization tests
# ---------------------------------------------------------------------------
class TestRunner:
    def test_save_and_load_roundtrip(self):
        run = _make_dummy_run()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        run.save(path)
        loaded = BenchmarkRun.load(path)
        assert loaded.model == run.model
        assert len(loaded.results) == len(run.results)
        assert loaded.results[0].ttft_ms == run.results[0].ttft_ms
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_run() -> BenchmarkRun:
    results = [
        BenchmarkResult(
            model="test-model",
            prompt_text=f"prompt {i}",
            category="reasoning",
            response_text=f"response {i} with some words here",
            ttft_ms=100.0 + i * 50,
            total_latency_ms=500.0 + i * 100,
            tokens_generated=50 + i * 10,
            tokens_per_second=30.0 + i * 5,
            prompt_tokens=20,
            completion_tokens=50 + i * 10,
            reference="response with some reference words",
        )
        for i in range(3)
    ]
    return BenchmarkRun(
        model="test-model",
        suite_name="reasoning",
        base_url="https://api.openai.com/v1",
        results=results,
        timestamp="2025-01-01T00:00:00Z",
    )
