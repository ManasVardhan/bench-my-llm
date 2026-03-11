"""Tests for the reporter module."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from bench_my_llm.reporter import print_report, _quality_color, _tps_color
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


class TestColorHelpers:
    def test_quality_color_high(self):
        assert _quality_color(0.9) == "green"
        assert _quality_color(0.7) == "green"

    def test_quality_color_medium(self):
        assert _quality_color(0.5) == "yellow"
        assert _quality_color(0.4) == "yellow"

    def test_quality_color_low(self):
        assert _quality_color(0.3) == "red"
        assert _quality_color(0.0) == "red"

    def test_tps_color_high(self):
        assert _tps_color(100) == "green"
        assert _tps_color(80) == "green"

    def test_tps_color_medium(self):
        assert _tps_color(60) == "yellow"
        assert _tps_color(40) == "yellow"

    def test_tps_color_low(self):
        assert _tps_color(20) == "red"
        assert _tps_color(0) == "red"


class TestPrintReport:
    def test_report_renders(self):
        """Report should render without errors."""
        run = _make_dummy_run()
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        print_report(run, console)
        text = output.getvalue()
        assert "test-model" in text
        assert "reasoning" in text

    def test_report_shows_latency_stats(self):
        run = _make_dummy_run()
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        print_report(run, console)
        text = output.getvalue()
        assert "p50" in text
        assert "p95" in text

    def test_report_shows_cost(self):
        run = _make_dummy_run()
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        print_report(run, console)
        text = output.getvalue()
        assert "$" in text

    def test_report_shows_per_prompt(self):
        run = _make_dummy_run()
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        print_report(run, console)
        text = output.getvalue()
        assert "Per-Prompt" in text

    def test_report_with_no_reference(self):
        """Report should handle prompts with no reference answer."""
        results = [
            BenchmarkResult(
                model="test-model",
                prompt_text="test prompt",
                category="creative",
                response_text="a creative response",
                ttft_ms=150.0,
                total_latency_ms=600.0,
                tokens_generated=40,
                tokens_per_second=25.0,
                prompt_tokens=15,
                completion_tokens=40,
                reference="",
            ),
        ]
        run = BenchmarkRun(
            model="test-model",
            suite_name="creative",
            base_url="https://api.openai.com/v1",
            results=results,
            timestamp="2025-01-01T00:00:00Z",
        )
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        print_report(run, console)
        text = output.getvalue()
        assert "100%" in text  # no reference = 100% quality

    def test_report_default_console(self):
        """Should work without passing a console."""
        run = _make_dummy_run()
        # Just ensure no crash
        print_report(run)


class TestReportEdgeCases:
    def test_single_result(self):
        """Report with just one prompt."""
        results = [
            BenchmarkResult(
                model="test-model",
                prompt_text="only prompt",
                category="factual",
                response_text="only response",
                ttft_ms=200.0,
                total_latency_ms=800.0,
                tokens_generated=30,
                tokens_per_second=20.0,
                prompt_tokens=10,
                completion_tokens=30,
                reference="only reference",
            ),
        ]
        run = BenchmarkRun(
            model="test-model",
            suite_name="factual",
            base_url="",
            results=results,
            timestamp="2025-01-01T00:00:00Z",
        )
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        print_report(run, console)
        assert "test-model" in output.getvalue()

    def test_zero_tps(self):
        """Handle edge case of 0 TPS."""
        results = [
            BenchmarkResult(
                model="slow-model",
                prompt_text="test",
                category="reasoning",
                response_text="",
                ttft_ms=5000.0,
                total_latency_ms=10000.0,
                tokens_generated=0,
                tokens_per_second=0.0,
                prompt_tokens=10,
                completion_tokens=0,
                reference="",
            ),
        ]
        run = BenchmarkRun(
            model="slow-model",
            suite_name="reasoning",
            base_url="",
            results=results,
            timestamp="2025-01-01T00:00:00Z",
        )
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        print_report(run, console)
        assert "slow-model" in output.getvalue()


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
