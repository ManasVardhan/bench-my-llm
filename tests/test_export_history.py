"""Tests for the export and history CLI commands, and __main__.py module support."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


@pytest.fixture
def runner():
    return CliRunner()


def _make_results_file(tmp_path: Path, model: str = "gpt-4o", suite: str = "reasoning",
                       filename: str = "results.json", timestamp: str = "2026-04-01T10:00:00") -> Path:
    """Create a valid benchmark results JSON file."""
    run = BenchmarkRun(model=model, suite_name=suite, base_url="https://api.openai.com/v1",
                       timestamp=timestamp)
    run.results = [
        BenchmarkResult(
            model=model, prompt_text="Test prompt 1", category="reasoning",
            response_text="The answer is 42", ttft_ms=150.0, total_latency_ms=1200.0,
            tokens_generated=20, tokens_per_second=50.0, prompt_tokens=15,
            completion_tokens=20, reference="42",
        ),
        BenchmarkResult(
            model=model, prompt_text="Test prompt 2", category="coding",
            response_text="def foo(): return 1", ttft_ms=200.0, total_latency_ms=1500.0,
            tokens_generated=10, tokens_per_second=30.0, prompt_tokens=12,
            completion_tokens=10, reference="def foo",
        ),
    ]
    path = tmp_path / filename
    run.save(str(path))
    return path


# -----------------------------------------------------------------------
# export command tests
# -----------------------------------------------------------------------

class TestExport:
    """Tests for bench-my-llm export command."""

    def test_export_help(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "markdown" in result.output

    def test_export_markdown_stdout(self, runner, tmp_path):
        results_file = _make_results_file(tmp_path)
        result = runner.invoke(cli, ["export", str(results_file), "--format", "markdown"])
        assert result.exit_code == 0
        assert "# Benchmark: gpt-4o" in result.output
        assert "| p50 |" in result.output
        assert "Mean TPS:" in result.output
        assert "Per-Prompt Results" in result.output

    def test_export_markdown_to_file(self, runner, tmp_path):
        results_file = _make_results_file(tmp_path)
        output_file = tmp_path / "report.md"
        result = runner.invoke(cli, ["export", str(results_file), "--format", "markdown",
                                     "-o", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "# Benchmark: gpt-4o" in content

    def test_export_csv_stdout(self, runner, tmp_path):
        results_file = _make_results_file(tmp_path)
        result = runner.invoke(cli, ["export", str(results_file), "--format", "csv"])
        assert result.exit_code == 0
        reader = csv.reader(io.StringIO(result.output))
        rows = [r for r in reader if r]  # skip empty trailing row
        assert rows[0][0] == "model"  # header
        assert len(rows) == 3  # header + 2 results
        assert rows[1][0] == "gpt-4o"

    def test_export_csv_to_file(self, runner, tmp_path):
        results_file = _make_results_file(tmp_path)
        output_file = tmp_path / "report.csv"
        result = runner.invoke(cli, ["export", str(results_file), "--format", "csv",
                                     "-o", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()

    def test_export_json_stdout(self, runner, tmp_path):
        results_file = _make_results_file(tmp_path)
        result = runner.invoke(cli, ["export", str(results_file), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["model"] == "gpt-4o"
        assert "ttft_p50" in data[0]
        assert "mean_tps" in data[0]

    def test_export_invalid_json(self, runner, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")
        result = runner.invoke(cli, ["export", str(bad_file)])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_export_comparison_format(self, runner, tmp_path):
        """Export works with comparison-format JSON (list of runs)."""
        run1 = BenchmarkRun(model="gpt-4o", suite_name="reasoning",
                            base_url="", timestamp="2026-04-01")
        run1.results = [
            BenchmarkResult(
                model="gpt-4o", prompt_text="Q", category="reasoning",
                response_text="A", ttft_ms=100, total_latency_ms=500,
                tokens_generated=10, tokens_per_second=40, prompt_tokens=5,
                completion_tokens=10, reference="A",
            ),
        ]
        run2 = BenchmarkRun(model="gpt-4o-mini", suite_name="reasoning",
                            base_url="", timestamp="2026-04-01")
        run2.results = [
            BenchmarkResult(
                model="gpt-4o-mini", prompt_text="Q", category="reasoning",
                response_text="A", ttft_ms=80, total_latency_ms=400,
                tokens_generated=8, tokens_per_second=35, prompt_tokens=5,
                completion_tokens=8, reference="A",
            ),
        ]
        path = tmp_path / "compare.json"
        path.write_text(json.dumps([
            {"model": "gpt-4o", "suite": "reasoning", "results": [r.to_dict() for r in run1.results]},
            {"model": "gpt-4o-mini", "suite": "reasoning", "results": [r.to_dict() for r in run2.results]},
        ], indent=2))

        result = runner.invoke(cli, ["export", str(path), "--format", "markdown"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "gpt-4o-mini" in result.output

    def test_export_markdown_latency_rows(self, runner, tmp_path):
        """Verify all latency rows appear."""
        results_file = _make_results_file(tmp_path)
        result = runner.invoke(cli, ["export", str(results_file), "--format", "markdown"])
        assert result.exit_code == 0
        assert "| p95 |" in result.output
        assert "| p99 |" in result.output
        assert "| Mean |" in result.output


# -----------------------------------------------------------------------
# history command tests
# -----------------------------------------------------------------------

class TestHistory:
    """Tests for bench-my-llm history command."""

    def test_history_help(self, runner):
        result = runner.invoke(cli, ["history", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--limit" in result.output

    def test_history_basic(self, runner, tmp_path):
        _make_results_file(tmp_path, filename="run1.json", model="gpt-4o")
        _make_results_file(tmp_path, filename="run2.json", model="gpt-4o-mini")
        result = runner.invoke(cli, ["history", str(tmp_path)])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "Benchmark History" in result.output

    def test_history_filter_by_model(self, runner, tmp_path):
        _make_results_file(tmp_path, filename="run1.json", model="gpt-4o")
        _make_results_file(tmp_path, filename="run2.json", model="claude-3-opus")
        result = runner.invoke(cli, ["history", str(tmp_path), "--model", "claude"])
        assert result.exit_code == 0
        # Should show exactly 1 run (the claude one)
        assert "Showing 1 run(s)" in result.output
        assert "claud" in result.output  # Rich may truncate "claude-3-opus"

    def test_history_empty_dir(self, runner, tmp_path):
        result = runner.invoke(cli, ["history", str(tmp_path)])
        assert result.exit_code == 0
        assert "No JSON files found" in result.output

    def test_history_limit(self, runner, tmp_path):
        for i in range(5):
            _make_results_file(tmp_path, filename=f"run_{i}.json",
                               model=f"model-{i}", timestamp=f"2026-04-0{i+1}T10:00:00")
        result = runner.invoke(cli, ["history", str(tmp_path), "--limit", "2"])
        assert result.exit_code == 0
        assert "Showing 2 run(s)" in result.output

    def test_history_invalid_json_skipped(self, runner, tmp_path):
        _make_results_file(tmp_path, filename="good.json", model="gpt-4o")
        (tmp_path / "bad.json").write_text("not json{{{")
        result = runner.invoke(cli, ["history", str(tmp_path)])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_history_shows_cost(self, runner, tmp_path):
        _make_results_file(tmp_path, filename="run.json", model="gpt-4o")
        result = runner.invoke(cli, ["history", str(tmp_path)])
        assert result.exit_code == 0
        assert "$" in result.output

    def test_history_no_matching_model(self, runner, tmp_path):
        _make_results_file(tmp_path, filename="run.json", model="gpt-4o")
        result = runner.invoke(cli, ["history", str(tmp_path), "--model", "nonexistent"])
        assert result.exit_code == 0
        assert "No matching benchmark runs" in result.output


# -----------------------------------------------------------------------
# __main__.py tests
# -----------------------------------------------------------------------

class TestMainModule:
    """Tests for python -m bench_my_llm support."""

    def test_main_module_imports(self):
        import bench_my_llm.__main__  # noqa: F401

    def test_main_version_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "bench_my_llm", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "bench-my-llm" in result.stdout

    def test_main_help_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "bench_my_llm", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "bench-my-llm" in result.stdout

    def test_main_export_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "bench_my_llm", "export", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--format" in result.stdout

    def test_main_history_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "bench_my_llm", "history", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
