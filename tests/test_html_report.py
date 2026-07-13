"""Tests for the HTML report renderer and the export --format html path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.html_report import _bar_width, render_html
from bench_my_llm.metrics import compute_metrics
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


def _make_run(model: str = "gpt-4o", prompt_text: str = "What is 6 x 7?") -> BenchmarkRun:
    run = BenchmarkRun(
        model=model,
        suite_name="reasoning",
        base_url="https://api.openai.com/v1",
        timestamp="2026-07-12T22:00:00",
    )
    run.results = [
        BenchmarkResult(
            model=model, prompt_text=prompt_text, category="reasoning",
            response_text="The answer is 42", ttft_ms=150.0, total_latency_ms=1200.0,
            tokens_generated=20, tokens_per_second=50.0, prompt_tokens=15,
            completion_tokens=20, reference="42",
        ),
        BenchmarkResult(
            model=model, prompt_text="Write foo()", category="coding",
            response_text="def foo(): return 1", ttft_ms=200.0, total_latency_ms=1500.0,
            tokens_generated=10, tokens_per_second=30.0, prompt_tokens=12,
            completion_tokens=10, reference="def foo",
        ),
    ]
    return run


def _pairs(*runs: BenchmarkRun) -> list[tuple[BenchmarkRun, object]]:
    return [(r, compute_metrics(r)) for r in runs]


class TestBarWidth:
    def test_proportional(self) -> None:
        assert _bar_width(50, 100) == 50.0

    def test_full(self) -> None:
        assert _bar_width(100, 100) == 100.0

    def test_clamps_above_100(self) -> None:
        assert _bar_width(150, 100) == 100.0

    def test_clamps_negative(self) -> None:
        assert _bar_width(-5, 100) == 0.0

    def test_zero_maximum(self) -> None:
        assert _bar_width(10, 0) == 0.0


class TestRenderHtml:
    def test_is_complete_document(self) -> None:
        doc = render_html(_pairs(_make_run()))
        assert doc.startswith("<!DOCTYPE html>")
        assert "</html>" in doc
        assert "<style>" in doc
        assert "<script" not in doc  # no JS, self-contained

    def test_contains_run_details(self) -> None:
        doc = render_html(_pairs(_make_run()))
        assert "gpt-4o" in doc
        assert "reasoning" in doc
        assert "coding" in doc
        assert "TTFT p50" in doc
        assert "Per-Prompt Results" in doc

    def test_custom_title(self) -> None:
        doc = render_html(_pairs(_make_run()), title="Nightly Bench")
        assert "<title>Nightly Bench</title>" in doc
        assert "Nightly Bench</h1>" in doc

    def test_single_run_has_no_comparison_table(self) -> None:
        doc = render_html(_pairs(_make_run()))
        assert "Model Comparison" not in doc

    def test_multi_run_has_comparison_table(self) -> None:
        doc = render_html(_pairs(_make_run("gpt-4o"), _make_run("claude-sonnet-4")))
        assert "Model Comparison" in doc
        assert "gpt-4o" in doc
        assert "claude-sonnet-4" in doc

    def test_escapes_model_name(self) -> None:
        doc = render_html(_pairs(_make_run(model="<script>alert(1)</script>")))
        assert "<script>alert(1)</script>" not in doc
        assert "&lt;script&gt;" in doc

    def test_escapes_prompt_text(self) -> None:
        doc = render_html(_pairs(_make_run(prompt_text='<img src=x onerror="pwn()">')))
        assert '<img src=x onerror="pwn()">' not in doc
        assert "&lt;img" in doc

    def test_truncates_long_prompts(self) -> None:
        doc = render_html(_pairs(_make_run(prompt_text="z" * 200)))
        assert "z" * 61 not in doc
        assert "z" * 60 + "..." in doc

    def test_empty_runs_rejected(self) -> None:
        with pytest.raises(ValueError, match="No runs"):
            render_html([])

    def test_quality_bars_present(self) -> None:
        doc = render_html(_pairs(_make_run()))
        assert 'class="bar quality"' in doc


class TestExportHtmlCli:
    def _save(self, tmp_path: Path, filename: str = "results.json") -> Path:
        path = tmp_path / filename
        _make_run().save(str(path))
        return path

    def test_html_to_stdout(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        result = CliRunner().invoke(cli, ["export", str(path), "--format", "html"])
        assert result.exit_code == 0
        assert result.output.startswith("<!DOCTYPE html>")
        assert "gpt-4o" in result.output

    def test_html_to_file(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        out = tmp_path / "report.html"
        result = CliRunner().invoke(
            cli, ["export", str(path), "--format", "html", "-o", str(out)]
        )
        assert result.exit_code == 0
        content = out.read_text()
        assert content.startswith("<!DOCTYPE html>")
        assert "Per-Prompt Results" in content

    def test_html_title_option(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        result = CliRunner().invoke(
            cli, ["export", str(path), "--format", "html", "--title", "My Report"]
        )
        assert result.exit_code == 0
        assert "<title>My Report</title>" in result.output

    def test_html_comparison_file(self, tmp_path: Path) -> None:
        runs = [_make_run("gpt-4o"), _make_run("claude-sonnet-4")]
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
        path.write_text(json.dumps(data))

        result = CliRunner().invoke(cli, ["export", str(path), "--format", "html"])

        assert result.exit_code == 0
        assert "Model Comparison" in result.output
        assert "claude-sonnet-4" in result.output

    def test_html_listed_in_help(self) -> None:
        result = CliRunner().invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "html" in result.output

    def test_invalid_json_still_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        result = CliRunner().invoke(cli, ["export", str(path), "--format", "html"])
        assert result.exit_code == 1
