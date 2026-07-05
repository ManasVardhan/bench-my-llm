"""Tests for the cost-adjusted leaderboard module and CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.leaderboard import (
    LeaderboardEntry,
    _normalize,
    build_leaderboard,
    load_runs,
    sort_entries,
)
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun


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
) -> BenchmarkRun:
    run = BenchmarkRun(
        model=model,
        suite_name=suite,
        base_url="https://api.openai.com/v1",
        timestamp="2026-07-04T10:00:00",
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


def _save_run(run: BenchmarkRun, path: Path) -> Path:
    run.save(str(path))
    return path


# -----------------------------------------------------------------------
# _normalize
# -----------------------------------------------------------------------


class TestNormalize:
    def test_higher_is_better(self):
        assert _normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]

    def test_lower_is_better(self):
        assert _normalize([0.0, 5.0, 10.0], lower_is_better=True) == [1.0, 0.5, 0.0]

    def test_all_equal_normalizes_to_one(self):
        assert _normalize([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]

    def test_all_equal_lower_is_better(self):
        assert _normalize([2.0, 2.0], lower_is_better=True) == [1.0, 1.0]

    def test_single_value(self):
        assert _normalize([7.0]) == [1.0]


# -----------------------------------------------------------------------
# load_runs
# -----------------------------------------------------------------------


class TestLoadRuns:
    def test_single_run_file(self, tmp_path):
        path = _save_run(_make_run(), tmp_path / "r.json")
        runs = load_runs([path])
        assert len(runs) == 1
        assert runs[0].model == "gpt-4o"
        assert len(runs[0].results) == 1

    def test_comparison_file(self, tmp_path):
        run_a = _make_run(model="gpt-4o")
        run_b = _make_run(model="claude-sonnet-4")
        data = [
            {
                "model": r.model,
                "suite": r.suite_name,
                "timestamp": r.timestamp,
                "results": [res.to_dict() for res in r.results],
            }
            for r in (run_a, run_b)
        ]
        path = tmp_path / "cmp.json"
        path.write_text(json.dumps(data))
        runs = load_runs([path])
        assert [r.model for r in runs] == ["gpt-4o", "claude-sonnet-4"]

    def test_multiple_files(self, tmp_path):
        p1 = _save_run(_make_run(model="gpt-4o"), tmp_path / "a.json")
        p2 = _save_run(_make_run(model="gpt-4o-mini"), tmp_path / "b.json")
        runs = load_runs([p1, p2])
        assert len(runs) == 2

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_runs([path])

    def test_non_dict_non_list_raises(self, tmp_path):
        path = tmp_path / "num.json"
        path.write_text("42")
        with pytest.raises(ValueError, match="Unrecognized results format"):
            load_runs([path])

    def test_missing_results_key_raises(self, tmp_path):
        path = tmp_path / "nores.json"
        path.write_text(json.dumps({"model": "gpt-4o"}))
        with pytest.raises(ValueError, match="Missing 'results'"):
            load_runs([path])


# -----------------------------------------------------------------------
# build_leaderboard
# -----------------------------------------------------------------------


class TestBuildLeaderboard:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No benchmark runs"):
            build_leaderboard([])

    def test_single_run_gets_full_score(self):
        entries = build_leaderboard([_make_run()])
        assert len(entries) == 1
        assert entries[0].value_score == 100.0
        assert entries[0].model == "gpt-4o"

    def test_sorted_by_value_score_descending(self):
        # Same quality, cheaper and faster model should win.
        expensive = _make_run(
            model="gpt-4", tps=20.0, prompt_tokens=1000, completion_tokens=2000
        )
        cheap = _make_run(
            model="gpt-4o-mini", tps=80.0, prompt_tokens=1000, completion_tokens=2000
        )
        entries = build_leaderboard([expensive, cheap])
        assert entries[0].model == "gpt-4o-mini"
        assert entries[0].value_score > entries[1].value_score

    def test_quality_dominates(self):
        # A perfect-quality model should beat a zero-quality one even if
        # the latter is cheaper and faster.
        good = _make_run(
            model="gpt-4", response="The answer is 42", reference="42", tps=10.0
        )
        bad = _make_run(
            model="gpt-4o-mini",
            response="completely unrelated words here",
            reference="42",
            tps=90.0,
        )
        entries = build_leaderboard([good, bad])
        assert entries[0].model == "gpt-4"

    def test_quality_per_dollar_computed(self):
        entries = build_leaderboard([_make_run()])
        e = entries[0]
        assert e.cost_usd > 0
        assert e.quality_per_dollar == round(e.quality / e.cost_usd, 1)

    def test_entry_fields_populated(self):
        e = build_leaderboard([_make_run()])[0]
        assert e.suite == "reasoning"
        assert e.num_prompts == 1
        assert e.ttft_p50_ms == 150.0
        assert e.latency_p50_ms == 1200.0
        assert e.mean_tps == 50.0

    def test_to_dict_serializable(self):
        e = build_leaderboard([_make_run()])[0]
        data = e.to_dict()
        json.dumps(data)
        assert data["model"] == "gpt-4o"
        assert data["value_score"] == 100.0

    def test_to_dict_infinite_qpd_is_none(self):
        e = LeaderboardEntry(
            model="free", suite="s", num_prompts=1, quality=1.0, cost_usd=0.0,
            mean_tps=1.0, ttft_p50_ms=1.0, latency_p50_ms=1.0,
            quality_per_dollar=float("inf"), value_score=100.0,
        )
        assert e.to_dict()["quality_per_dollar"] is None


# -----------------------------------------------------------------------
# sort_entries
# -----------------------------------------------------------------------


class TestSortEntries:
    def _entries(self) -> list[LeaderboardEntry]:
        return [
            LeaderboardEntry(
                model="a", suite="s", num_prompts=1, quality=0.9, cost_usd=0.5,
                mean_tps=10.0, ttft_p50_ms=100.0, latency_p50_ms=500.0,
                quality_per_dollar=1.8, value_score=70.0,
            ),
            LeaderboardEntry(
                model="b", suite="s", num_prompts=1, quality=0.5, cost_usd=0.1,
                mean_tps=50.0, ttft_p50_ms=50.0, latency_p50_ms=200.0,
                quality_per_dollar=5.0, value_score=80.0,
            ),
        ]

    def test_sort_by_value(self):
        assert [e.model for e in sort_entries(self._entries(), "value")] == ["b", "a"]

    def test_sort_by_quality(self):
        assert [e.model for e in sort_entries(self._entries(), "quality")] == ["a", "b"]

    def test_sort_by_cost_ascending(self):
        assert [e.model for e in sort_entries(self._entries(), "cost")] == ["b", "a"]

    def test_sort_by_speed(self):
        assert [e.model for e in sort_entries(self._entries(), "speed")] == ["b", "a"]

    def test_sort_by_quality_per_dollar(self):
        result = sort_entries(self._entries(), "quality-per-dollar")
        assert [e.model for e in result] == ["b", "a"]

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown sort key"):
            sort_entries(self._entries(), "vibes")


# -----------------------------------------------------------------------
# leaderboard CLI command
# -----------------------------------------------------------------------


class TestLeaderboardCLI:
    def test_help(self, runner):
        result = runner.invoke(cli, ["leaderboard", "--help"])
        assert result.exit_code == 0
        assert "--sort" in result.output
        assert "quality-per-dollar" in result.output

    def test_requires_file_argument(self, runner):
        result = runner.invoke(cli, ["leaderboard"])
        assert result.exit_code != 0

    def test_missing_file_fails(self, runner):
        result = runner.invoke(cli, ["leaderboard", "does-not-exist.json"])
        assert result.exit_code != 0

    def test_table_output(self, runner, tmp_path):
        p1 = _save_run(_make_run(model="gpt-4o"), tmp_path / "a.json")
        p2 = _save_run(_make_run(model="o3-mini", tps=80.0), tmp_path / "b.json")
        result = runner.invoke(cli, ["leaderboard", str(p1), str(p2)])
        assert result.exit_code == 0
        assert "Cost-Adjusted Leaderboard" in result.output
        assert "gpt-4o" in result.output
        assert "o3-mini" in result.output
        assert "Value score" in result.output

    def test_single_file(self, runner, tmp_path):
        p = _save_run(_make_run(), tmp_path / "a.json")
        result = runner.invoke(cli, ["leaderboard", str(p)])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_json_output(self, runner, tmp_path):
        p1 = _save_run(_make_run(model="gpt-4o"), tmp_path / "a.json")
        p2 = _save_run(_make_run(model="gpt-4o-mini"), tmp_path / "b.json")
        result = runner.invoke(cli, ["leaderboard", str(p1), str(p2), "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert {"model", "value_score", "quality_per_dollar"} <= set(data[0])

    def test_sort_option(self, runner, tmp_path):
        p1 = _save_run(_make_run(model="gpt-4o"), tmp_path / "a.json")
        p2 = _save_run(_make_run(model="gpt-4o-mini"), tmp_path / "b.json")
        result = runner.invoke(
            cli, ["leaderboard", str(p1), str(p2), "--sort", "cost", "--json-output"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["cost_usd"] <= data[1]["cost_usd"]

    def test_invalid_json_exits_1(self, runner, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{oops")
        result = runner.invoke(cli, ["leaderboard", str(bad)])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_comparison_file_input(self, runner, tmp_path):
        run_a = _make_run(model="gpt-4o")
        run_b = _make_run(model="gpt-4o-mini")
        data = [
            {
                "model": r.model,
                "suite": r.suite_name,
                "results": [res.to_dict() for res in r.results],
            }
            for r in (run_a, run_b)
        ]
        path = tmp_path / "cmp.json"
        path.write_text(json.dumps(data))
        result = runner.invoke(cli, ["leaderboard", str(path), "--json-output"])
        assert result.exit_code == 0
        assert len(json.loads(result.output)) == 2

    def test_invalid_sort_choice_rejected(self, runner, tmp_path):
        p = _save_run(_make_run(), tmp_path / "a.json")
        result = runner.invoke(cli, ["leaderboard", str(p), "--sort", "vibes"])
        assert result.exit_code != 0
