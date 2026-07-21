"""Tests for the historical trend tracking module and CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun
from bench_my_llm.trends import (
    SPARK_CHARS,
    TrendPoint,
    _parse_timestamp,
    build_trends,
    format_metric_value,
    load_trend_points,
    sparkline,
)


@pytest.fixture
def runner():
    return CliRunner()


def _make_run(
    model: str = "gpt-4o",
    suite: str = "reasoning",
    response: str = "The answer is 42",
    reference: str = "42",
    ttft: float = 150.0,
    latency: float = 1200.0,
    tps: float = 50.0,
    prompt_tokens: int = 100,
    completion_tokens: int = 200,
    timestamp: str = "2026-07-01T10:00:00",
) -> BenchmarkRun:
    run = BenchmarkRun(
        model=model,
        suite_name=suite,
        base_url="https://api.openai.com/v1",
        timestamp=timestamp,
    )
    run.results = [
        BenchmarkResult(
            model=model,
            prompt_text="Test prompt",
            category="reasoning",
            response_text=response,
            ttft_ms=ttft,
            total_latency_ms=latency,
            tokens_generated=completion_tokens,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reference=reference,
        ),
    ]
    return run


def _save_runs(tmp_path: Path, *runs_by_name: tuple[str, BenchmarkRun]) -> Path:
    for name, run in runs_by_name:
        run.save(str(tmp_path / name))
    return tmp_path


def _point(model: str = "gpt-4o", sort_key: float = 0.0, **overrides) -> TrendPoint:
    values = {
        "source_file": "a.json",
        "timestamp": "2026-07-01T10:00:00",
        "sort_key": sort_key,
        "model": model,
        "suite": "reasoning",
        "num_prompts": 1,
        "quality": 0.5,
        "mean_tps": 50.0,
        "ttft_p50_ms": 150.0,
        "latency_p50_ms": 1200.0,
        "cost_usd": 0.01,
    }
    values.update(overrides)
    return TrendPoint(**values)


# -----------------------------------------------------------------------
# sparkline
# -----------------------------------------------------------------------


class TestSparkline:
    def test_empty(self):
        assert sparkline([]) == ""

    def test_single_value_is_flat(self):
        assert sparkline([5.0]) == SPARK_CHARS[3]

    def test_equal_values_are_flat(self):
        assert sparkline([2.0, 2.0, 2.0]) == SPARK_CHARS[3] * 3

    def test_min_and_max_hit_extremes(self):
        line = sparkline([0.0, 10.0])
        assert line == SPARK_CHARS[0] + SPARK_CHARS[-1]

    def test_monotonic_rise(self):
        line = sparkline([1.0, 2.0, 3.0, 4.0])
        indices = [SPARK_CHARS.index(c) for c in line]
        assert indices == sorted(indices)
        assert indices[0] == 0 and indices[-1] == len(SPARK_CHARS) - 1

    def test_length_matches_input(self):
        assert len(sparkline([1.0, 5.0, 3.0, 2.0, 4.0])) == 5


# -----------------------------------------------------------------------
# _parse_timestamp
# -----------------------------------------------------------------------


class TestParseTimestamp:
    def test_iso(self):
        assert _parse_timestamp("2026-07-01T10:00:00") is not None

    def test_z_suffix(self):
        assert _parse_timestamp("2026-07-01T10:00:00Z") is not None

    def test_empty_returns_none(self):
        assert _parse_timestamp("") is None

    def test_garbage_returns_none(self):
        assert _parse_timestamp("not a date") is None

    def test_ordering(self):
        early = _parse_timestamp("2026-07-01T10:00:00")
        late = _parse_timestamp("2026-07-02T10:00:00")
        assert early is not None and late is not None and early < late


# -----------------------------------------------------------------------
# TrendPoint
# -----------------------------------------------------------------------


class TestTrendPoint:
    def test_metric_lookup(self):
        p = _point(quality=0.7, cost_usd=0.02)
        assert p.metric("quality") == 0.7
        assert p.metric("cost") == 0.02
        assert p.metric("tps") == 50.0
        assert p.metric("ttft") == 150.0
        assert p.metric("latency") == 1200.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            _point().metric("vibes")

    def test_to_dict_keys(self):
        d = _point().to_dict()
        assert d["quality"] == 0.5
        assert d["source_file"] == "a.json"
        assert "sort_key" not in d


# -----------------------------------------------------------------------
# load_trend_points
# -----------------------------------------------------------------------


class TestLoadTrendPoints:
    def test_loads_and_sorts_by_timestamp(self, tmp_path):
        _save_runs(
            tmp_path,
            ("b.json", _make_run(timestamp="2026-07-02T10:00:00", tps=60.0)),
            ("a.json", _make_run(timestamp="2026-07-01T10:00:00", tps=50.0)),
        )
        points = load_trend_points(tmp_path)
        assert [p.source_file for p in points] == ["a.json", "b.json"]
        assert points[0].mean_tps == 50.0

    def test_comparison_file_yields_multiple_points(self, tmp_path):
        runs = [_make_run(model="gpt-4o"), _make_run(model="claude-sonnet-4")]
        data = [
            {
                "model": r.model,
                "suite": r.suite_name,
                "timestamp": r.timestamp,
                "results": [res.to_dict() for res in r.results],
            }
            for r in runs
        ]
        (tmp_path / "cmp.json").write_text(json.dumps(data))
        points = load_trend_points(tmp_path)
        assert {p.model for p in points} == {"gpt-4o", "claude-sonnet-4"}

    def test_invalid_json_skipped(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json")
        _save_runs(tmp_path, ("good.json", _make_run()))
        points = load_trend_points(tmp_path)
        assert len(points) == 1

    def test_missing_results_skipped(self, tmp_path):
        (tmp_path / "meta.json").write_text(json.dumps({"model": "x"}))
        assert load_trend_points(tmp_path) == []

    def test_empty_results_skipped(self, tmp_path):
        (tmp_path / "empty.json").write_text(json.dumps({"model": "x", "results": []}))
        assert load_trend_points(tmp_path) == []

    def test_malformed_result_fields_skipped(self, tmp_path):
        (tmp_path / "weird.json").write_text(
            json.dumps({"model": "x", "results": [{"nope": 1}]})
        )
        assert load_trend_points(tmp_path) == []

    def test_scalar_json_skipped(self, tmp_path):
        (tmp_path / "num.json").write_text("42")
        assert load_trend_points(tmp_path) == []

    def test_missing_timestamp_falls_back_to_mtime(self, tmp_path):
        import os

        _save_runs(tmp_path, ("old.json", _make_run(timestamp="", tps=10.0)))
        _save_runs(tmp_path, ("new.json", _make_run(timestamp="", tps=99.0)))
        os.utime(tmp_path / "old.json", (100.0, 100.0))
        os.utime(tmp_path / "new.json", (200.0, 200.0))
        points = load_trend_points(tmp_path)
        assert [p.source_file for p in points] == ["old.json", "new.json"]

    def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Not a directory"):
            load_trend_points(f)

    def test_empty_directory(self, tmp_path):
        assert load_trend_points(tmp_path) == []

    def test_point_carries_metrics(self, tmp_path):
        _save_runs(tmp_path, ("run.json", _make_run(ttft=250.0, latency=2000.0)))
        (point,) = load_trend_points(tmp_path)
        assert point.ttft_p50_ms == 250.0
        assert point.latency_p50_ms == 2000.0
        assert point.cost_usd > 0
        assert point.num_prompts == 1


# -----------------------------------------------------------------------
# build_trends / ModelTrend
# -----------------------------------------------------------------------


class TestBuildTrends:
    def test_groups_by_model_alphabetically(self):
        points = [
            _point(model="gpt-4o", sort_key=1),
            _point(model="claude-sonnet-4", sort_key=2),
            _point(model="gpt-4o", sort_key=3),
        ]
        trends = build_trends(points)
        assert [t.model for t in trends] == ["claude-sonnet-4", "gpt-4o"]
        assert len(trends[1].points) == 2

    def test_model_filter_substring_case_insensitive(self):
        points = [_point(model="gpt-4o", sort_key=1), _point(model="claude", sort_key=2)]
        trends = build_trends(points, model_filter="GPT")
        assert [t.model for t in trends] == ["gpt-4o"]

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No benchmark runs match"):
            build_trends([_point()], model_filter="missing")

    def test_empty_points_raises(self):
        with pytest.raises(ValueError):
            build_trends([])

    def test_change_and_direction_higher_better(self):
        points = [
            _point(sort_key=1, quality=0.4),
            _point(sort_key=2, quality=0.8),
        ]
        (trend,) = build_trends(points)
        assert trend.change("quality") == pytest.approx(0.4)
        assert trend.direction("quality") == "improving"
        assert trend.change_pct("quality") == pytest.approx(100.0)

    def test_change_and_direction_lower_better(self):
        points = [
            _point(sort_key=1, latency_p50_ms=1000.0),
            _point(sort_key=2, latency_p50_ms=1500.0),
        ]
        (trend,) = build_trends(points)
        assert trend.direction("latency") == "declining"
        assert trend.change("latency") == 500.0

    def test_lower_better_improving(self):
        points = [
            _point(sort_key=1, cost_usd=0.02),
            _point(sort_key=2, cost_usd=0.01),
        ]
        (trend,) = build_trends(points)
        assert trend.direction("cost") == "improving"

    def test_flat_direction(self):
        points = [_point(sort_key=1), _point(sort_key=2)]
        (trend,) = build_trends(points)
        assert trend.direction("tps") == "flat"
        assert trend.change("tps") == 0.0

    def test_change_pct_zero_first_value(self):
        points = [
            _point(sort_key=1, cost_usd=0.0),
            _point(sort_key=2, cost_usd=0.01),
        ]
        (trend,) = build_trends(points)
        assert trend.change_pct("cost") is None

    def test_single_point_trend(self):
        (trend,) = build_trends([_point()])
        assert trend.change("quality") == 0.0
        assert trend.direction("quality") == "flat"

    def test_to_dict(self):
        points = [
            _point(sort_key=1, mean_tps=40.0),
            _point(sort_key=2, mean_tps=60.0),
        ]
        (trend,) = build_trends(points)
        d = trend.to_dict("tps")
        assert d["runs"] == 2
        assert d["values"] == [40.0, 60.0]
        assert d["first"] == 40.0
        assert d["latest"] == 60.0
        assert d["direction"] == "improving"
        assert len(d["points"]) == 2


# -----------------------------------------------------------------------
# format_metric_value
# -----------------------------------------------------------------------


class TestFormatMetricValue:
    def test_quality_percent(self):
        assert format_metric_value("quality", 0.85) == "85%"

    def test_latency_ms(self):
        assert format_metric_value("latency", 1234.5) == "1234ms"
        assert format_metric_value("ttft", 150.0) == "150ms"

    def test_cost_dollars(self):
        assert format_metric_value("cost", 0.0123) == "$0.0123"

    def test_tps_plain(self):
        assert format_metric_value("tps", 55.25) == "55.2"


# -----------------------------------------------------------------------
# CLI: trends command
# -----------------------------------------------------------------------


class TestTrendsCLI:
    def test_basic_table(self, runner, tmp_path):
        _save_runs(
            tmp_path,
            ("a.json", _make_run(timestamp="2026-07-01T10:00:00", tps=40.0)),
            ("b.json", _make_run(timestamp="2026-07-02T10:00:00", tps=60.0)),
        )
        result = runner.invoke(cli, ["trends", str(tmp_path), "--metric", "tps"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "Benchmark Trends" in result.output

    def test_improving_quality_shows_positive_change(self, runner, tmp_path):
        _save_runs(
            tmp_path,
            ("a.json", _make_run(timestamp="2026-07-01T10:00:00", response="42 maybe")),
            ("b.json", _make_run(timestamp="2026-07-02T10:00:00", response="42")),
        )
        result = runner.invoke(cli, ["trends", str(tmp_path)])
        assert result.exit_code == 0
        assert "+" in result.output

    def test_json_output(self, runner, tmp_path):
        _save_runs(
            tmp_path,
            ("a.json", _make_run(timestamp="2026-07-01T10:00:00", tps=40.0)),
            ("b.json", _make_run(timestamp="2026-07-02T10:00:00", tps=60.0)),
        )
        result = runner.invoke(
            cli, ["trends", str(tmp_path), "--metric", "tps", "--json-output"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["model"] == "gpt-4o"
        assert data[0]["values"] == [40.0, 60.0]
        assert data[0]["direction"] == "improving"

    def test_json_output_is_parseable_when_piped(self, runner, tmp_path):
        long_model = "my-organization/very-long-model-name-that-wraps-terminals-9000b-instruct"
        _save_runs(tmp_path, ("a.json", _make_run(model=long_model)))
        result = runner.invoke(cli, ["trends", str(tmp_path), "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["model"] == long_model

    def test_model_filter(self, runner, tmp_path):
        _save_runs(
            tmp_path,
            ("a.json", _make_run(model="gpt-4o")),
            ("b.json", _make_run(model="claude-sonnet-4")),
        )
        result = runner.invoke(cli, ["trends", str(tmp_path), "-m", "claude"])
        assert result.exit_code == 0
        assert "claude-sonnet-4" in result.output
        assert "gpt-4o" not in result.output

    def test_no_matching_model_exits_1(self, runner, tmp_path):
        _save_runs(tmp_path, ("a.json", _make_run()))
        result = runner.invoke(cli, ["trends", str(tmp_path), "-m", "missing"])
        assert result.exit_code == 1
        assert "No benchmark runs match" in result.output

    def test_empty_directory_exits_1(self, runner, tmp_path):
        result = runner.invoke(cli, ["trends", str(tmp_path)])
        assert result.exit_code == 1

    def test_nonexistent_directory_exits_2(self, runner, tmp_path):
        result = runner.invoke(cli, ["trends", str(tmp_path / "missing")])
        assert result.exit_code == 2

    def test_single_run_hint(self, runner, tmp_path):
        _save_runs(tmp_path, ("a.json", _make_run()))
        result = runner.invoke(cli, ["trends", str(tmp_path)])
        assert result.exit_code == 0
        assert "only one run" in result.output

    def test_invalid_metric_rejected(self, runner, tmp_path):
        result = runner.invoke(cli, ["trends", str(tmp_path), "--metric", "vibes"])
        assert result.exit_code == 2

    def test_default_metric_is_quality(self, runner, tmp_path):
        _save_runs(tmp_path, ("a.json", _make_run()))
        result = runner.invoke(cli, ["trends", str(tmp_path)])
        assert result.exit_code == 0
        assert "quality" in result.output
