"""Tests for metadata grouping and aggregation (v0.5)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.metadata_query import (
    GROUP_FIELDS,
    group_runs,
    load_runs,
)


def _run_dict(
    *,
    model: str = "gpt-4o",
    suite: str = "reasoning",
    endpoint: str = "https://api.openai.com/v1",
    temperature: float = 0.0,
    tool_version: str = "0.5.0",
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


class TestGroupRuns:
    def test_groups_by_endpoint(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(endpoint="http://localhost:1234/v1"))
        _write(tmp_path, "b.json", _run_dict(endpoint="http://localhost:1234/v1"))
        _write(tmp_path, "c.json", _run_dict(endpoint="https://api.openai.com/v1"))
        groups = group_runs(load_runs(tmp_path), "endpoint")
        assert [g.key for g in groups] == [
            "http://localhost:1234/v1",
            "https://api.openai.com/v1",
        ]
        assert [g.count for g in groups] == [2, 1]
        assert all(g.field == "endpoint" for g in groups)

    def test_groups_by_model_and_suite(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(model="gpt-4o", suite="reasoning"))
        _write(tmp_path, "b.json", _run_dict(model="claude-3", suite="coding"))
        runs = load_runs(tmp_path)
        by_model = group_runs(runs, "model")
        assert [g.key for g in by_model] == ["claude-3", "gpt-4o"]
        by_suite = group_runs(runs, "suite")
        assert [g.key for g in by_suite] == ["coding", "reasoning"]

    def test_aggregates_mean_min_max(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(ttft=100.0))
        _write(tmp_path, "b.json", _run_dict(ttft=300.0))
        groups = group_runs(load_runs(tmp_path), "model")
        assert len(groups) == 1
        agg = groups[0].aggregates["ttft"]
        assert agg.mean == 200.0
        assert agg.min == 100.0
        assert agg.max == 300.0

    def test_quality_aggregation_differs_per_group(self, tmp_path) -> None:
        _write(tmp_path, "good.json", _run_dict(model="a-model", response="answer"))
        _write(tmp_path, "bad.json", _run_dict(model="b-model", response="wrong"))
        groups = group_runs(load_runs(tmp_path), "model")
        quality = {g.key: g.aggregates["quality"].mean for g in groups}
        assert quality["a-model"] > quality["b-model"]

    def test_unset_field_groups_under_unset(self, tmp_path) -> None:
        _write(tmp_path, "cmp.json", [_run_dict(with_metadata=False)])
        _write(tmp_path, "a.json", _run_dict(tool_version="0.5.0"))
        groups = group_runs(load_runs(tmp_path), "tool_version")
        assert [g.key for g in groups] == ["(unset)", "0.5.0"]

    def test_temperature_key_is_display_formatted(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(temperature=0.7))
        _write(tmp_path, "b.json", _run_dict(temperature=0.0))
        groups = group_runs(load_runs(tmp_path), "temperature")
        assert [g.key for g in groups] == ["0", "0.7"]

    def test_unknown_field_raises(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict())
        with pytest.raises(ValueError, match="Unknown group field"):
            group_runs(load_runs(tmp_path), "bogus")

    def test_empty_runs_gives_empty_groups(self) -> None:
        assert group_runs([], "model") == []

    def test_group_fields_cover_expected_names(self) -> None:
        assert set(GROUP_FIELDS) == {
            "model",
            "suite",
            "endpoint",
            "temperature",
            "tool_version",
            "suite_version",
        }

    def test_to_dict_shape(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict())
        group = group_runs(load_runs(tmp_path), "model")[0]
        data = group.to_dict()
        assert data["field"] == "model"
        assert data["key"] == "gpt-4o"
        assert data["count"] == 1
        assert data["source_files"] == ["a.json"]
        assert set(data["aggregates"]) == {"quality", "tps", "ttft", "latency", "cost"}
        assert set(data["aggregates"]["cost"]) == {"metric", "mean", "min", "max"}


class TestRunsGroupCli:
    def test_table_output(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(endpoint="http://localhost:1234/v1"))
        _write(tmp_path, "b.json", _run_dict(endpoint="https://api.openai.com/v1"))
        runner = CliRunner()
        result = runner.invoke(cli, ["runs", "group", str(tmp_path), "--by", "endpoint"])
        assert result.exit_code == 0
        assert "Grouped by endpoint" in result.output
        assert "localhost" in result.output

    def test_default_groups_by_model(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(model="gpt-4o"))
        runner = CliRunner()
        result = runner.invoke(cli, ["runs", "group", str(tmp_path)])
        assert result.exit_code == 0
        assert "Grouped by model" in result.output
        assert "gpt-4o" in result.output

    def test_model_filter_narrows_runs(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(model="gpt-4o"))
        _write(tmp_path, "b.json", _run_dict(model="claude-3"))
        runner = CliRunner()
        result = runner.invoke(
            cli, ["runs", "group", str(tmp_path), "--model", "claude", "--json-output"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["key"] == "claude-3"

    def test_json_output(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict(temperature=0.0))
        _write(tmp_path, "b.json", _run_dict(temperature=0.7))
        runner = CliRunner()
        result = runner.invoke(
            cli, ["runs", "group", str(tmp_path), "--by", "temperature", "--json-output"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert [g["key"] for g in data] == ["0", "0.7"]
        assert all(g["count"] == 1 for g in data)

    def test_unknown_field_exits_one(self, tmp_path) -> None:
        _write(tmp_path, "a.json", _run_dict())
        runner = CliRunner()
        result = runner.invoke(cli, ["runs", "group", str(tmp_path), "--by", "bogus"])
        assert result.exit_code == 1
        assert "Unknown group field" in result.output

    def test_empty_directory_reports_no_runs(self, tmp_path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["runs", "group", str(tmp_path)])
        assert result.exit_code == 0
        assert "No saved runs" in result.output
