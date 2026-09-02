"""Tests for metadata filtering and diffing (v0.4)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.metadata_query import (
    diff_runs,
    filter_runs,
    find_run,
    load_runs,
)


def _run_dict(
    *,
    model: str = "gpt-4o",
    suite: str = "reasoning",
    endpoint: str = "https://api.openai.com/v1",
    temperature: float = 0.0,
    tool_version: str = "0.4.0",
    ttft: float = 100.0,
    latency: float = 500.0,
    quality_ref: str = "answer",
    response: str = "answer",
    with_metadata: bool = True,
) -> dict:
    result = {
        "model": model,
        "prompt_text": "q",
        "category": "reasoning",
        "response_text": response,
        "ttft_ms": ttft,
        "total_latency_ms": latency,
        "tokens_generated": 10,
        "tokens_per_second": 40.0,
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "reference": quality_ref,
    }
    run = {
        "model": model,
        "suite": suite,
        "base_url": endpoint,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "results": [result],
    }
    if with_metadata:
        run["metadata"] = {
            "notes": "",
            "endpoint": endpoint,
            "temperature": temperature,
            "suite_version": f"{suite} (1 prompts)",
            "tool_version": tool_version,
            "captured_at": "2026-01-01T00:00:00+00:00",
        }
    return run


def _write(directory, name: str, data: dict | list) -> None:
    (directory / name).write_text(json.dumps(data))


def test_load_runs_reads_metadata(tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict(temperature=0.7, tool_version="0.4.0"))
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].metadata.temperature == 0.7
    assert runs[0].metadata.tool_version == "0.4.0"
    assert runs[0].source_file == "a.json"


def test_load_runs_rejects_non_directory(tmp_path) -> None:
    f = tmp_path / "x.json"
    f.write_text("{}")
    with pytest.raises(ValueError):
        load_runs(f)


def test_load_runs_comparison_file_has_empty_metadata(tmp_path) -> None:
    _write(tmp_path, "cmp.json", [_run_dict(with_metadata=False)])
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].metadata.is_empty()


def test_load_runs_skips_invalid_json(tmp_path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    _write(tmp_path, "good.json", _run_dict())
    runs = load_runs(tmp_path)
    assert len(runs) == 1


def test_filter_by_temperature_is_exact(tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict(temperature=0.0))
    _write(tmp_path, "b.json", _run_dict(temperature=0.7))
    runs = load_runs(tmp_path)
    matched = filter_runs(runs, temperature=0.0)
    assert len(matched) == 1
    assert matched[0].source_file == "a.json"


def test_filter_by_endpoint_substring(tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict(endpoint="http://localhost:1234/v1"))
    _write(tmp_path, "b.json", _run_dict(endpoint="https://api.openai.com/v1"))
    runs = load_runs(tmp_path)
    matched = filter_runs(runs, endpoint="localhost")
    assert len(matched) == 1
    assert matched[0].metadata.endpoint == "http://localhost:1234/v1"


def test_filter_by_tool_version_and_model(tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict(model="gpt-4o", tool_version="0.4.0"))
    _write(tmp_path, "b.json", _run_dict(model="claude-3", tool_version="0.3.0"))
    runs = load_runs(tmp_path)
    assert len(filter_runs(runs, tool_version="0.4")) == 1
    assert len(filter_runs(runs, model="claude")) == 1


def test_filter_no_criteria_returns_all(tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict())
    _write(tmp_path, "b.json", _run_dict(model="claude-3"))
    runs = load_runs(tmp_path)
    assert len(filter_runs(runs)) == 2


def test_diff_detects_changed_metadata(tmp_path) -> None:
    _write(tmp_path, "old.json", _run_dict(temperature=0.0, tool_version="0.3.0"))
    _write(tmp_path, "new.json", _run_dict(temperature=0.7, tool_version="0.4.0"))
    runs = load_runs(tmp_path)
    old = find_run(runs, "old.json")
    new = find_run(runs, "new.json")
    diff = diff_runs(old, new)
    changed = {f.field for f in diff.changed_fields}
    assert "temperature" in changed
    assert "tool_version" in changed
    assert "model" not in changed


def test_diff_metric_direction(tmp_path) -> None:
    # latency is lower-is-better: dropping latency should read as improving.
    _write(tmp_path, "old.json", _run_dict(latency=800.0))
    _write(tmp_path, "new.json", _run_dict(latency=400.0))
    runs = load_runs(tmp_path)
    diff = diff_runs(find_run(runs, "old"), find_run(runs, "new"))
    latency = next(m for m in diff.metrics if m.metric == "latency")
    assert latency.direction == "improving"
    assert latency.delta == pytest.approx(-400.0)
    assert latency.delta_pct == pytest.approx(-50.0)


def test_diff_flat_metric(tmp_path) -> None:
    _write(tmp_path, "old.json", _run_dict(latency=500.0))
    _write(tmp_path, "new.json", _run_dict(latency=500.0))
    runs = load_runs(tmp_path)
    diff = diff_runs(find_run(runs, "old"), find_run(runs, "new"))
    latency = next(m for m in diff.metrics if m.metric == "latency")
    assert latency.direction == "flat"


def test_find_run_ambiguous_raises(tmp_path) -> None:
    _write(tmp_path, "run-a.json", _run_dict())
    _write(tmp_path, "run-b.json", _run_dict())
    runs = load_runs(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        find_run(runs, "run")


def test_find_run_no_match_raises(tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict())
    runs = load_runs(tmp_path)
    with pytest.raises(ValueError, match="No run matches"):
        find_run(runs, "zzz")


def test_find_run_exact_beats_substring(tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict())
    _write(tmp_path, "a.json.bak.json", _run_dict(model="claude-3"))
    runs = load_runs(tmp_path)
    found = find_run(runs, "a.json")
    assert found.source_file == "a.json"


# --- CLI ---


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_filter_table(runner, tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict(endpoint="http://localhost:1234/v1"))
    _write(tmp_path, "b.json", _run_dict(endpoint="https://api.openai.com/v1"))
    result = runner.invoke(cli, ["runs", "filter", str(tmp_path), "-e", "localhost"])
    assert result.exit_code == 0
    assert "a.json" in result.output
    assert "b.json" not in result.output


def test_cli_filter_json(runner, tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict(temperature=0.0))
    result = runner.invoke(
        cli, ["runs", "filter", str(tmp_path), "-t", "0", "--json-output"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["source_file"] == "a.json"


def test_cli_filter_no_match(runner, tmp_path) -> None:
    _write(tmp_path, "a.json", _run_dict(temperature=0.0))
    result = runner.invoke(cli, ["runs", "filter", str(tmp_path), "-t", "1.5"])
    assert result.exit_code == 0
    assert "No runs match" in result.output


def test_cli_diff_table(runner, tmp_path) -> None:
    _write(tmp_path, "old.json", _run_dict(temperature=0.0, latency=800.0))
    _write(tmp_path, "new.json", _run_dict(temperature=0.7, latency=400.0))
    result = runner.invoke(cli, ["runs", "diff", str(tmp_path), "old.json", "new.json"])
    assert result.exit_code == 0
    assert "temperature" in result.output
    assert "Metric Changes" in result.output


def test_cli_diff_json(runner, tmp_path) -> None:
    _write(tmp_path, "old.json", _run_dict(tool_version="0.3.0"))
    _write(tmp_path, "new.json", _run_dict(tool_version="0.4.0"))
    result = runner.invoke(
        cli, ["runs", "diff", str(tmp_path), "old", "new", "--json-output"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    fields = {c["field"] for c in data["metadata_changes"]}
    assert "tool_version" in fields
    assert len(data["metrics"]) == 5


def test_cli_diff_ambiguous_errors(runner, tmp_path) -> None:
    _write(tmp_path, "run-a.json", _run_dict())
    _write(tmp_path, "run-b.json", _run_dict())
    result = runner.invoke(cli, ["runs", "diff", str(tmp_path), "run", "run-a.json"])
    assert result.exit_code == 1
    assert "ambiguous" in result.output
