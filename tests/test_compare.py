"""Tests for the compare module."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from bench_my_llm.compare import compare_runs, _winner_style
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


class TestWinnerStyle:
    def test_lower_is_better_a_wins(self):
        sa, sb = _winner_style(10.0, 20.0, lower_is_better=True)
        assert "green" in sa
        assert "dim" in sb

    def test_lower_is_better_b_wins(self):
        sa, sb = _winner_style(20.0, 10.0, lower_is_better=True)
        assert "dim" in sa
        assert "green" in sb

    def test_higher_is_better_a_wins(self):
        sa, sb = _winner_style(20.0, 10.0, lower_is_better=False)
        assert "green" in sa
        assert "dim" in sb

    def test_higher_is_better_b_wins(self):
        sa, sb = _winner_style(10.0, 20.0, lower_is_better=False)
        assert "dim" in sa
        assert "green" in sb

    def test_tie_returns_empty(self):
        sa, sb = _winner_style(10.0, 10.0, lower_is_better=True)
        assert sa == ""
        assert sb == ""

    def test_tie_higher_is_better(self):
        sa, sb = _winner_style(10.0, 10.0, lower_is_better=False)
        assert sa == ""
        assert sb == ""


class TestCompareRuns:
    def test_compare_two_models(self):
        runs = [
            _make_dummy_run(model="fast-model", ttft_base=50, tps_base=80),
            _make_dummy_run(model="slow-model", ttft_base=200, tps_base=20),
        ]
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        compare_runs(runs, console)
        text = output.getvalue()
        assert "fast-model" in text
        assert "slow-model" in text
        assert "Winner" in text

    def test_compare_three_models(self):
        runs = [
            _make_dummy_run(model="a", ttft_base=100, tps_base=50),
            _make_dummy_run(model="b", ttft_base=80, tps_base=60),
            _make_dummy_run(model="c", ttft_base=200, tps_base=30),
        ]
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        compare_runs(runs, console)
        text = output.getvalue()
        assert "a" in text
        assert "b" in text
        assert "c" in text

    def test_compare_less_than_two_prints_error(self):
        runs = [_make_dummy_run(model="only-one")]
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        compare_runs(runs, console)
        text = output.getvalue()
        assert "2 runs" in text.lower() or "at least" in text.lower()

    def test_compare_default_console(self):
        runs = [
            _make_dummy_run(model="a"),
            _make_dummy_run(model="b"),
        ]
        # Just ensure no crash
        compare_runs(runs)

    def test_compare_shows_metrics(self):
        runs = [
            _make_dummy_run(model="alpha", ttft_base=100, tps_base=50),
            _make_dummy_run(model="beta", ttft_base=150, tps_base=40),
        ]
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        compare_runs(runs, console)
        text = output.getvalue()
        assert "TTFT" in text
        assert "TPS" in text
        assert "Cost" in text
        assert "Quality" in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_run(
    model: str = "test-model",
    ttft_base: float = 100.0,
    tps_base: float = 30.0,
) -> BenchmarkRun:
    results = [
        BenchmarkResult(
            model=model,
            prompt_text=f"prompt {i}",
            category="reasoning",
            response_text=f"response {i} with some words here",
            ttft_ms=ttft_base + i * 20,
            total_latency_ms=ttft_base * 5 + i * 100,
            tokens_generated=50 + i * 10,
            tokens_per_second=tps_base + i * 5,
            prompt_tokens=20,
            completion_tokens=50 + i * 10,
            reference="response with some reference words",
        )
        for i in range(3)
    ]
    return BenchmarkRun(
        model=model,
        suite_name="reasoning",
        base_url="https://api.openai.com/v1",
        results=results,
        timestamp="2025-01-01T00:00:00Z",
    )
