"""Tests for bench-my-llm."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.bench_my_llm.prompts import get_suite, SUITES, Prompt, PromptSuite
from src.bench_my_llm.metrics import (
    compute_latency_stats,
    estimate_cost,
    score_quality,
    compute_metrics,
)
from src.bench_my_llm.runner import (
    BenchmarkResult,
    BenchmarkRun,
    _count_tokens_approx,
    run_benchmark,
    run_single_prompt,
)
from unittest.mock import MagicMock, patch
from openai import AuthenticationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(**kwargs) -> BenchmarkResult:
    """Helper to create a BenchmarkResult with default values."""
    defaults = dict(
        model="gpt-4o",
        prompt_text="Say hello",
        category="test",
        response_text="Hello!",
        ttft_ms=100.0,
        total_latency_ms=200.0,
        tokens_generated=5,
        tokens_per_second=25.0,
        prompt_tokens=3,
        completion_tokens=5,
    )
    defaults.update(kwargs)
    return BenchmarkResult(**defaults)


def make_suite(prompts=None) -> PromptSuite:
    """Helper to create a PromptSuite."""
    return PromptSuite(
        name="test",
        description="test suite",
        prompts=prompts or [Prompt(text="Say hello", category="test", max_tokens=10)],
    )


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
        expected = (1000 / 1000) * 0.0025 + (500 / 1000) * 0.01
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost_unknown_model(self):
        cost = estimate_cost("some-random-model", 1000, 1000)
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
# Count tokens tests
# ---------------------------------------------------------------------------

class TestCountTokensApprox:
    def test_basic(self):
        """Two words * 1.3 = 2.6, int() truncates to 2."""
        assert _count_tokens_approx("hello world") == 2

    def test_empty_string_returns_one(self):
        """Empty string has 0 words, but max(1, ...) ensures minimum of 1 token."""
        assert _count_tokens_approx("") == 1

    def test_longer_sentence(self):
        """Longer sentences scale proportionally."""
        result = _count_tokens_approx("no mocking needed - just call it and assert!")
        assert result == 11


# ---------------------------------------------------------------------------
# run_single_prompt tests
# ---------------------------------------------------------------------------

class TestRunSinglePrompt:
    def test_basic_response(self):
        """Single chunk response is captured correctly."""
        mock_client = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Hello!"
        mock_client.chat.completions.create.return_value = [mock_chunk]

        prompt = Prompt(text="Say hello", category="test", max_tokens=10)
        result = run_single_prompt(mock_client, "gpt-4o", prompt)

        assert result.response_text == "Hello!"
        assert result.model == "gpt-4o"

    def test_empty_response(self):
        """None content chunks result in empty response text."""
        mock_client = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = None
        mock_client.chat.completions.create.return_value = [mock_chunk]

        prompt = Prompt(text="Say hello", category="test", max_tokens=10)
        result = run_single_prompt(mock_client, "gpt-4o", prompt)

        assert result.response_text == ""

    def test_multiple_chunks_concatenated(self):
        """Multiple chunks are joined into a single response text."""
        mock_client = MagicMock()
        chunk1, chunk2, chunk3 = MagicMock(), MagicMock(), MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk2.choices[0].delta.content = " world"
        chunk3.choices[0].delta.content = "!"
        mock_client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]

        prompt = Prompt(text="Say hello", category="test", max_tokens=10)
        result = run_single_prompt(mock_client, "gpt-4o", prompt)

        assert result.response_text == "Hello world!"

    def test_tokens_counted(self):
        """Token counts are calculated from response text."""
        mock_client = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Hello world"
        mock_client.chat.completions.create.return_value = [mock_chunk]

        prompt = Prompt(text="Say hello", category="test", max_tokens=10)
        result = run_single_prompt(mock_client, "gpt-4o", prompt)

        assert result.tokens_generated > 0
        assert result.prompt_tokens > 0

    def test_latency_recorded(self):
        """Latency metrics are recorded."""
        mock_client = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Hi"
        mock_client.chat.completions.create.return_value = [mock_chunk]

        prompt = Prompt(text="Say hello", category="test", max_tokens=10)
        result = run_single_prompt(mock_client, "gpt-4o", prompt)

        assert result.total_latency_ms >= 0
        assert result.ttft_ms >= 0


# ---------------------------------------------------------------------------
# run_benchmark tests
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_basic(self):
        """run_benchmark creates OpenAI client and collects real results."""
        with patch("src.bench_my_llm.runner.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_chunk = MagicMock()
            mock_chunk.choices[0].delta.content = "Hello!"
            mock_client.chat.completions.create.return_value = [mock_chunk]

            run = run_benchmark("gpt-4o", make_suite(), api_key="fake-key")

            assert run.model == "gpt-4o"
            assert len(run.results) == 1
            assert run.results[0].response_text == "Hello!"

    def test_auth_error_raises_system_exit(self):
        """Authentication errors raise SystemExit with helpful message."""
        with patch("src.bench_my_llm.runner.OpenAI"), \
             patch("src.bench_my_llm.runner.run_single_prompt") as mock_run:

            mock_run.side_effect = AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body={},
            )

            with pytest.raises(SystemExit):
                run_benchmark("gpt-4o", make_suite(), api_key="bad-key")

    def test_multiple_prompts(self):
        """All prompts in suite are benchmarked."""
        with patch("src.bench_my_llm.runner.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_chunk = MagicMock()
            mock_chunk.choices[0].delta.content = "response"
            mock_client.chat.completions.create.return_value = [mock_chunk]

            suite = make_suite([
                Prompt(text="Prompt 1", category="test", max_tokens=10),
                Prompt(text="Prompt 2", category="test", max_tokens=10),
                Prompt(text="Prompt 3", category="test", max_tokens=10),
            ])

            run = run_benchmark("gpt-4o", suite, api_key="fake-key")
            assert len(run.results) == 3


# ---------------------------------------------------------------------------
# BenchmarkRun save/load tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunSaveLoad:
    def test_roundtrip(self, tmp_path):
        """Results survive save/load roundtrip."""
        run = BenchmarkRun(
            model="gpt-4o",
            suite_name="test",
            base_url="https://api.openai.com/v1",
            timestamp="2024-01-01T00:00:00+00:00",
        )
        run.results.append(make_result())

        path = tmp_path / "results.json"
        run.save(path)
        loaded = BenchmarkRun.load(path)

        assert loaded.model == "gpt-4o"
        assert len(loaded.results) == 1
        assert loaded.results[0].response_text == "Hello!"
