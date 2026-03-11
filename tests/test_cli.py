"""Tests for the CLI module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def dummy_results_file(tmp_path):
    """Create a dummy results JSON file."""
    run = _make_dummy_run()
    path = tmp_path / "results.json"
    run.save(str(path))
    return str(path)


@pytest.fixture
def dummy_comparison_file(tmp_path):
    """Create a dummy comparison JSON file with multiple runs."""
    runs = [_make_dummy_run(model=f"model-{i}") for i in range(2)]
    data = [
        {
            "model": r.model,
            "suite": r.suite_name,
            "timestamp": r.timestamp,
            "results": [res.to_dict() for res in r.results],
        }
        for r in runs
    ]
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


class TestCLIVersion:
    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "bench-my-llm" in result.output
        assert "0.1.1" in result.output

    def test_help_flag(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "bench-my-llm" in result.output
        assert "run" in result.output
        assert "compare" in result.output
        assert "report" in result.output


class TestCLIRun:
    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_basic(self, mock_run_benchmark, runner):
        mock_run_benchmark.return_value = _make_dummy_run()
        result = runner.invoke(cli, ["run", "--model", "test-model"])
        assert result.exit_code == 0
        mock_run_benchmark.assert_called_once()

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_with_suite(self, mock_run_benchmark, runner):
        mock_run_benchmark.return_value = _make_dummy_run()
        result = runner.invoke(cli, ["run", "--model", "test-model", "--suite", "reasoning"])
        assert result.exit_code == 0
        call_kwargs = mock_run_benchmark.call_args
        assert call_kwargs[1]["suite"].name == "reasoning" or call_kwargs.kwargs.get("suite", call_kwargs[1].get("suite"))

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_with_output(self, mock_run_benchmark, runner, tmp_path):
        mock_run_benchmark.return_value = _make_dummy_run()
        output_path = str(tmp_path / "out.json")
        result = runner.invoke(cli, ["run", "--model", "test-model", "--output", output_path])
        assert result.exit_code == 0
        assert Path(output_path).exists()

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_with_temperature(self, mock_run_benchmark, runner):
        mock_run_benchmark.return_value = _make_dummy_run()
        result = runner.invoke(cli, ["run", "--model", "test-model", "--temperature", "0.7"])
        assert result.exit_code == 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_with_api_key(self, mock_run_benchmark, runner):
        mock_run_benchmark.return_value = _make_dummy_run()
        result = runner.invoke(cli, ["run", "--model", "test-model", "--api-key", "sk-test123"])
        assert result.exit_code == 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_with_base_url(self, mock_run_benchmark, runner):
        mock_run_benchmark.return_value = _make_dummy_run()
        result = runner.invoke(cli, ["run", "--model", "test-model", "--base-url", "http://localhost:8080/v1"])
        assert result.exit_code == 0

    def test_run_missing_model(self, runner):
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0

    def test_run_invalid_suite(self, runner):
        result = runner.invoke(cli, ["run", "--model", "test", "--suite", "nonexistent"])
        assert result.exit_code != 0


class TestCLICompare:
    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_two_models(self, mock_run_benchmark, runner):
        mock_run_benchmark.side_effect = [
            _make_dummy_run(model="model-a"),
            _make_dummy_run(model="model-b"),
        ]
        result = runner.invoke(cli, ["compare", "model-a", "model-b"])
        assert result.exit_code == 0

    def test_compare_one_model_fails(self, runner):
        result = runner.invoke(cli, ["compare", "only-one"])
        assert result.exit_code != 0

    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_with_output(self, mock_run_benchmark, runner, tmp_path):
        mock_run_benchmark.side_effect = [
            _make_dummy_run(model="model-a"),
            _make_dummy_run(model="model-b"),
        ]
        output_path = str(tmp_path / "cmp.json")
        result = runner.invoke(cli, ["compare", "model-a", "model-b", "--output", output_path])
        assert result.exit_code == 0
        data = json.loads(Path(output_path).read_text())
        assert len(data) == 2
        assert data[0]["model"] == "model-a"
        assert data[1]["model"] == "model-b"

    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_three_models(self, mock_run_benchmark, runner):
        mock_run_benchmark.side_effect = [
            _make_dummy_run(model="a"),
            _make_dummy_run(model="b"),
            _make_dummy_run(model="c"),
        ]
        result = runner.invoke(cli, ["compare", "a", "b", "c"])
        assert result.exit_code == 0


class TestCLIReport:
    def test_report_from_file(self, runner, dummy_results_file):
        result = runner.invoke(cli, ["report", dummy_results_file])
        assert result.exit_code == 0

    def test_report_nonexistent_file(self, runner):
        result = runner.invoke(cli, ["report", "/nonexistent/path.json"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_run(model: str = "test-model") -> BenchmarkRun:
    results = [
        BenchmarkResult(
            model=model,
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
        model=model,
        suite_name="reasoning",
        base_url="https://api.openai.com/v1",
        results=results,
        timestamp="2025-01-01T00:00:00Z",
    )
