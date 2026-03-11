"""Extended tests for the metrics module, covering edge cases and cost table."""

from __future__ import annotations

import pytest

from bench_my_llm.metrics import (
    COST_TABLE,
    DEFAULT_COST,
    _lookup_cost,
    compute_latency_stats,
    compute_metrics,
    estimate_cost,
    score_quality,
)
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


class TestCostTable:
    def test_known_models_have_entries(self):
        known = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
                  "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        for model in known:
            cost = _lookup_cost(model)
            assert cost != DEFAULT_COST, f"{model} should have a specific entry"

    def test_unknown_model_uses_default(self):
        cost = _lookup_cost("totally-unknown-model-xyz")
        assert cost == DEFAULT_COST

    def test_lookup_is_case_insensitive(self):
        cost_lower = _lookup_cost("gpt-4o")
        cost_upper = _lookup_cost("GPT-4O")
        assert cost_lower == cost_upper

    def test_cost_tuple_structure(self):
        for model, (inp, out) in COST_TABLE.items():
            assert inp > 0, f"{model} input cost should be positive"
            assert out > 0, f"{model} output cost should be positive"
            assert out >= inp, f"{model} output cost should be >= input cost"


class TestEstimateCost:
    def test_zero_tokens(self):
        cost = estimate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_scales_linearly(self):
        cost_1k = estimate_cost("gpt-4o", 1000, 1000)
        cost_2k = estimate_cost("gpt-4o", 2000, 2000)
        assert abs(cost_2k - cost_1k * 2) < 0.0001

    def test_input_only(self):
        inp_rate, _ = _lookup_cost("gpt-4o")
        cost = estimate_cost("gpt-4o", 1000, 0)
        assert abs(cost - inp_rate) < 0.0001

    def test_output_only(self):
        _, out_rate = _lookup_cost("gpt-4o")
        cost = estimate_cost("gpt-4o", 0, 1000)
        assert abs(cost - out_rate) < 0.0001


class TestScoreQuality:
    def test_identical_texts(self):
        assert score_quality("hello world", "hello world") == 1.0

    def test_completely_different(self):
        score = score_quality("apples oranges bananas", "xyz abc def")
        assert score == 0.0

    def test_empty_response(self):
        score = score_quality("", "some reference text")
        assert score == 0.0

    def test_empty_reference(self):
        score = score_quality("some response text", "")
        assert score == 1.0

    def test_both_empty(self):
        score = score_quality("", "")
        assert score == 1.0

    def test_single_word_match(self):
        score = score_quality("Python", "Python")
        assert score == 1.0

    def test_partial_match(self):
        score = score_quality(
            "Python is a programming language",
            "Python is great for data science",
        )
        assert 0.0 < score < 1.0

    def test_punctuation_stripped(self):
        score = score_quality("Hello, world!", "Hello world")
        assert score == 1.0

    def test_case_insensitive(self):
        score = score_quality("HELLO WORLD", "hello world")
        assert score == 1.0


class TestComputeLatencyStats:
    def test_single_value(self):
        stats = compute_latency_stats([100.0])
        assert stats.p50_ms == 100.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0
        assert stats.mean_ms == 100.0

    def test_two_values(self):
        stats = compute_latency_stats([100.0, 200.0])
        assert stats.min_ms == 100.0
        assert stats.max_ms == 200.0
        assert stats.mean_ms == 150.0

    def test_many_values(self):
        values = list(range(1, 101))
        values_float = [float(v) for v in values]
        stats = compute_latency_stats(values_float)
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        assert stats.p50_ms == 50.5


class TestComputeMetrics:
    def test_empty_run_raises(self):
        run = BenchmarkRun(model="test", suite_name="test", base_url="")
        with pytest.raises(ValueError, match="No results"):
            compute_metrics(run)

    def test_single_result(self):
        results = [
            BenchmarkResult(
                model="test",
                prompt_text="q",
                category="reasoning",
                response_text="a",
                ttft_ms=100.0,
                total_latency_ms=500.0,
                tokens_generated=50,
                tokens_per_second=40.0,
                prompt_tokens=20,
                completion_tokens=50,
                reference="",
            ),
        ]
        run = BenchmarkRun(model="test", suite_name="test", base_url="", results=results)
        metrics = compute_metrics(run)
        assert metrics.num_prompts == 1
        assert metrics.mean_tps == 40.0
        assert metrics.total_prompt_tokens == 20
        assert metrics.total_completion_tokens == 50

    def test_metrics_token_sums(self):
        results = [
            BenchmarkResult(
                model="test",
                prompt_text=f"q{i}",
                category="reasoning",
                response_text=f"a{i}",
                ttft_ms=100.0,
                total_latency_ms=500.0,
                tokens_generated=50,
                tokens_per_second=40.0,
                prompt_tokens=20 + i * 5,
                completion_tokens=50 + i * 10,
                reference="",
            )
            for i in range(5)
        ]
        run = BenchmarkRun(model="test", suite_name="test", base_url="", results=results)
        metrics = compute_metrics(run)
        assert metrics.total_prompt_tokens == sum(20 + i * 5 for i in range(5))
        assert metrics.total_completion_tokens == sum(50 + i * 10 for i in range(5))

    def test_quality_score_range(self):
        results = [
            BenchmarkResult(
                model="test",
                prompt_text="q",
                category="reasoning",
                response_text="some words overlap here",
                ttft_ms=100.0,
                total_latency_ms=500.0,
                tokens_generated=50,
                tokens_per_second=40.0,
                prompt_tokens=20,
                completion_tokens=50,
                reference="different words overlap there",
            ),
        ]
        run = BenchmarkRun(model="test", suite_name="test", base_url="", results=results)
        metrics = compute_metrics(run)
        assert 0.0 <= metrics.mean_quality_score <= 1.0
