"""Tests for error paths, edge cases, and hardened error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.metrics import compute_latency_stats, score_quality
from bench_my_llm.prompts import Prompt, PromptSuite, get_suite
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


# ---------------------------------------------------------------------------
# Runner error handling
# ---------------------------------------------------------------------------
class TestRunnerErrors:
    """Test that runner surfaces clear errors for API failures."""

    def _make_suite(self) -> PromptSuite:
        return PromptSuite(
            name="tiny",
            description="one prompt",
            prompts=[Prompt(text="hi", category="test", max_tokens=10)],
        )

    @patch("bench_my_llm.runner.OpenAI")
    def test_auth_error_exits(self, mock_openai_cls):
        from openai import AuthenticationError
        from bench_my_llm.runner import run_benchmark

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )

        with pytest.raises(SystemExit, match="Authentication failed"):
            run_benchmark(model="test", suite=self._make_suite(), api_key="bad")

    @patch("bench_my_llm.runner.OpenAI")
    def test_rate_limit_error_exits(self, mock_openai_cls):
        from openai import RateLimitError
        from bench_my_llm.runner import run_benchmark

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        with pytest.raises(SystemExit, match="Rate limit"):
            run_benchmark(model="test", suite=self._make_suite(), api_key="key")

    @patch("bench_my_llm.runner.OpenAI")
    def test_connection_error_exits(self, mock_openai_cls):
        from openai import APIConnectionError
        from bench_my_llm.runner import run_benchmark

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock(),
        )

        with pytest.raises(SystemExit, match="Could not connect"):
            run_benchmark(model="test", suite=self._make_suite(), api_key="key")

    @patch("bench_my_llm.runner.OpenAI")
    def test_timeout_error_exits(self, mock_openai_cls):
        from openai import APITimeoutError
        from bench_my_llm.runner import run_benchmark

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock(),
        )

        with pytest.raises(SystemExit, match="timed out"):
            run_benchmark(model="test", suite=self._make_suite(), api_key="key")


# ---------------------------------------------------------------------------
# CLI report error handling
# ---------------------------------------------------------------------------
class TestCLIReportErrors:
    """Test the report command handles bad input gracefully."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_report_invalid_json(self, runner, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all {{{")
        result = runner.invoke(cli, ["report", str(bad_file)])
        assert result.exit_code != 0

    def test_report_unexpected_list_format(self, runner, tmp_path):
        bad_file = tmp_path / "bad_list.json"
        bad_file.write_text(json.dumps([{"nope": "wrong"}]))
        result = runner.invoke(cli, ["report", str(bad_file)])
        assert result.exit_code != 0

    def test_report_comparison_format(self, runner, tmp_path):
        """The comparison list format should render a comparison report."""
        data = [
            {
                "model": f"model-{i}",
                "suite": "reasoning",
                "timestamp": "2025-01-01T00:00:00Z",
                "results": [
                    BenchmarkResult(
                        model=f"model-{i}",
                        prompt_text="q",
                        category="reasoning",
                        response_text="a a a",
                        ttft_ms=100.0,
                        total_latency_ms=500.0,
                        tokens_generated=50,
                        tokens_per_second=40.0,
                        prompt_tokens=20,
                        completion_tokens=50,
                        reference="a a a",
                    ).to_dict()
                ],
            }
            for i in range(2)
        ]
        path = tmp_path / "cmp.json"
        path.write_text(json.dumps(data, indent=2))
        result = runner.invoke(cli, ["report", str(path)])
        assert result.exit_code == 0
        assert "model-0" in result.output
        assert "model-1" in result.output

    def test_report_single_in_list_format(self, runner, tmp_path):
        """A list with one run should print a single report, not comparison."""
        data = [
            {
                "model": "solo-model",
                "suite": "reasoning",
                "timestamp": "2025-01-01T00:00:00Z",
                "results": [
                    BenchmarkResult(
                        model="solo-model",
                        prompt_text="q",
                        category="reasoning",
                        response_text="a",
                        ttft_ms=100.0,
                        total_latency_ms=500.0,
                        tokens_generated=50,
                        tokens_per_second=40.0,
                        prompt_tokens=20,
                        completion_tokens=50,
                    ).to_dict()
                ],
            }
        ]
        path = tmp_path / "single_list.json"
        path.write_text(json.dumps(data, indent=2))
        result = runner.invoke(cli, ["report", str(path)])
        assert result.exit_code == 0
        assert "solo-model" in result.output


# ---------------------------------------------------------------------------
# Metrics edge cases
# ---------------------------------------------------------------------------
class TestMetricsEdgeCases:
    def test_score_quality_whitespace_only_reference(self):
        """A reference of only whitespace should be treated as empty."""
        assert score_quality("any response", "   \n\t  ") == 1.0

    def test_score_quality_single_char_words_stripped(self):
        """Words with 1 char should be stripped from tokenization."""
        # "I" and "a" have length 1 and should be excluded
        score = score_quality("I wrote a book", "I wrote a book")
        # "wrote" and "book" match, so score should be 1.0
        assert score == 1.0

    def test_latency_stats_identical_values(self):
        """All identical values should produce uniform percentiles."""
        stats = compute_latency_stats([42.0, 42.0, 42.0, 42.0])
        assert stats.p50_ms == 42.0
        assert stats.p95_ms == 42.0
        assert stats.p99_ms == 42.0
        assert stats.mean_ms == 42.0
        assert stats.min_ms == 42.0
        assert stats.max_ms == 42.0


# ---------------------------------------------------------------------------
# BenchmarkRun load edge cases
# ---------------------------------------------------------------------------
class TestBenchmarkRunLoadErrors:
    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            BenchmarkRun.load("/nonexistent/path/xyz.json")

    def test_load_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json!!!")
        with pytest.raises(json.JSONDecodeError):
            BenchmarkRun.load(str(bad))

    def test_load_missing_keys(self, tmp_path):
        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text(json.dumps({"model": "test"}))
        with pytest.raises((KeyError, TypeError)):
            BenchmarkRun.load(str(incomplete))


# ---------------------------------------------------------------------------
# Prompt suite edge cases
# ---------------------------------------------------------------------------
class TestPromptEdgeCases:
    def test_get_suite_error_lists_available(self):
        """Error message should list available suites."""
        with pytest.raises(KeyError) as exc_info:
            get_suite("nope")
        error_msg = str(exc_info.value)
        assert "reasoning" in error_msg
        assert "coding" in error_msg

    def test_prompt_default_max_tokens(self):
        p = Prompt(text="hello", category="test")
        assert p.max_tokens == 512

    def test_prompt_frozen(self):
        """Prompt is frozen and should not allow attribute assignment."""
        p = Prompt(text="hello", category="test")
        with pytest.raises(AttributeError):
            p.text = "changed"  # type: ignore[misc]
