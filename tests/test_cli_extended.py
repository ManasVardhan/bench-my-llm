"""Extended CLI tests covering remaining edge cases.

Covers the report command's unrecognized format path,
compare with custom options, and the __main__ block.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_result(model: str = "m", **overrides) -> BenchmarkResult:
    defaults = dict(
        model=model,
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
    )
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def _make_run(model: str = "test-model", n_results: int = 2) -> BenchmarkRun:
    run = BenchmarkRun(
        model=model,
        suite_name="reasoning",
        base_url="https://api.openai.com/v1",
        timestamp="2025-01-01T00:00:00Z",
    )
    run.results = [_make_result(model=model, ttft_ms=100.0 + i * 20) for i in range(n_results)]
    return run


class TestReportUnrecognized:
    def test_report_with_integer_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """A JSON file containing just an integer should fail gracefully."""
        bad = tmp_path / "int.json"
        bad.write_text("42")
        result = runner.invoke(cli, ["report", str(bad)])
        assert result.exit_code != 0

    def test_report_with_string_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """A JSON file containing just a string should fail gracefully."""
        bad = tmp_path / "str.json"
        bad.write_text('"hello"')
        result = runner.invoke(cli, ["report", str(bad)])
        assert result.exit_code != 0

    def test_report_with_null_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """A JSON file containing null should fail gracefully."""
        bad = tmp_path / "null.json"
        bad.write_text("null")
        result = runner.invoke(cli, ["report", str(bad)])
        assert result.exit_code != 0

    def test_report_with_boolean_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """A JSON file containing a boolean should fail gracefully."""
        bad = tmp_path / "bool.json"
        bad.write_text("true")
        result = runner.invoke(cli, ["report", str(bad)])
        assert result.exit_code != 0


class TestReportDictFormat:
    def test_report_single_run_dict(self, runner: CliRunner, tmp_path: Path) -> None:
        """Report should handle a single run in dict format."""
        run = _make_run()
        path = tmp_path / "single.json"
        run.save(str(path))

        result = runner.invoke(cli, ["report", str(path)])
        assert result.exit_code == 0
        assert "test-model" in result.output


class TestCompareEdgeCases:
    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_with_suite_option(self, mock_run: MagicMock, runner: CliRunner) -> None:
        mock_run.side_effect = [_make_run("a"), _make_run("b")]
        result = runner.invoke(cli, ["compare", "a", "b", "--suite", "coding"])
        assert result.exit_code == 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_with_temperature(self, mock_run: MagicMock, runner: CliRunner) -> None:
        mock_run.side_effect = [_make_run("a"), _make_run("b")]
        result = runner.invoke(cli, ["compare", "a", "b", "--temperature", "0.5"])
        assert result.exit_code == 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_with_api_key(self, mock_run: MagicMock, runner: CliRunner) -> None:
        mock_run.side_effect = [_make_run("a"), _make_run("b")]
        result = runner.invoke(cli, ["compare", "a", "b", "--api-key", "sk-test"])
        assert result.exit_code == 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_with_base_url(self, mock_run: MagicMock, runner: CliRunner) -> None:
        mock_run.side_effect = [_make_run("a"), _make_run("b")]
        result = runner.invoke(cli, ["compare", "a", "b", "--base-url", "http://localhost:8080/v1"])
        assert result.exit_code == 0

    def test_compare_no_models_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compare"])
        assert result.exit_code != 0


class TestRunCommandEdgeCases:
    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_all_suite(self, mock_run: MagicMock, runner: CliRunner) -> None:
        mock_run.return_value = _make_run()
        result = runner.invoke(cli, ["run", "--model", "m", "--suite", "all"])
        assert result.exit_code == 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_creative_suite(self, mock_run: MagicMock, runner: CliRunner) -> None:
        mock_run.return_value = _make_run()
        result = runner.invoke(cli, ["run", "--model", "m", "--suite", "creative"])
        assert result.exit_code == 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_factual_suite(self, mock_run: MagicMock, runner: CliRunner) -> None:
        mock_run.return_value = _make_run()
        result = runner.invoke(cli, ["run", "--model", "m", "--suite", "factual"])
        assert result.exit_code == 0
