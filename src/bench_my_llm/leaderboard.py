"""Cost-adjusted leaderboard: rank models by value, not just raw speed or quality.

Takes one or more saved benchmark result files and ranks the runs by a
composite value score that balances quality, cost, and throughput. Also
exposes a plain quality-per-dollar figure for quick bang-for-buck checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .metrics import compute_metrics
from .runner import BenchmarkResult, BenchmarkRun

# Weights for the composite value score. Quality dominates, cost matters,
# speed breaks ties.
QUALITY_WEIGHT = 0.5
COST_WEIGHT = 0.3
SPEED_WEIGHT = 0.2


@dataclass
class LeaderboardEntry:
    """One ranked row in the cost-adjusted leaderboard."""

    model: str
    suite: str
    num_prompts: int
    quality: float
    cost_usd: float
    mean_tps: float
    ttft_p50_ms: float
    latency_p50_ms: float
    quality_per_dollar: float
    value_score: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "suite": self.suite,
            "num_prompts": self.num_prompts,
            "quality": self.quality,
            "cost_usd": self.cost_usd,
            "mean_tps": self.mean_tps,
            "ttft_p50_ms": self.ttft_p50_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "quality_per_dollar": (
                None if self.quality_per_dollar == float("inf")
                else self.quality_per_dollar
            ),
            "value_score": self.value_score,
        }


def load_runs(paths: list[str | Path]) -> list[BenchmarkRun]:
    """Load benchmark runs from result files.

    Each file may contain a single run (dict) or a comparison (list of runs).

    Raises:
        ValueError: If a file has invalid JSON or an unrecognized structure.
    """
    runs: list[BenchmarkRun] = []
    for path in paths:
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {p}: {exc}") from exc

        items = [data] if isinstance(data, dict) else data if isinstance(data, list) else None
        if items is None:
            raise ValueError(f"Unrecognized results format in {p}")

        for item in items:
            if not isinstance(item, dict) or "results" not in item:
                raise ValueError(f"Missing 'results' in run data in {p}")
            run = BenchmarkRun(
                model=item.get("model", "unknown"),
                suite_name=item.get("suite", "unknown"),
                base_url=item.get("base_url", ""),
                timestamp=item.get("timestamp", ""),
            )
            run.results = [BenchmarkResult(**r) for r in item["results"]]
            runs.append(run)
    return runs


def _normalize(values: list[float], lower_is_better: bool = False) -> list[float]:
    """Min-max normalize values to 0..1. Equal values all normalize to 1.0."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    if lower_is_better:
        return [(hi - v) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


def build_leaderboard(runs: list[BenchmarkRun]) -> list[LeaderboardEntry]:
    """Build a leaderboard from benchmark runs, sorted by value score descending.

    The value score (0 to 100) combines normalized quality (weight 0.5),
    cost (0.3, lower is better), and throughput (0.2) across the field.

    Raises:
        ValueError: If no runs are provided or a run has no results.
    """
    if not runs:
        raise ValueError("No benchmark runs to rank")

    metrics = [compute_metrics(r) for r in runs]

    qualities = [m.mean_quality_score for m in metrics]
    costs = [m.estimated_cost_usd for m in metrics]
    speeds = [m.mean_tps for m in metrics]

    q_norm = _normalize(qualities)
    c_norm = _normalize(costs, lower_is_better=True)
    s_norm = _normalize(speeds)

    entries: list[LeaderboardEntry] = []
    for i, m in enumerate(metrics):
        score = (
            QUALITY_WEIGHT * q_norm[i]
            + COST_WEIGHT * c_norm[i]
            + SPEED_WEIGHT * s_norm[i]
        ) * 100
        qpd = (
            round(m.mean_quality_score / m.estimated_cost_usd, 1)
            if m.estimated_cost_usd > 0
            else float("inf")
        )
        entries.append(
            LeaderboardEntry(
                model=m.model,
                suite=m.suite,
                num_prompts=m.num_prompts,
                quality=m.mean_quality_score,
                cost_usd=m.estimated_cost_usd,
                mean_tps=m.mean_tps,
                ttft_p50_ms=m.ttft.p50_ms,
                latency_p50_ms=m.total_latency.p50_ms,
                quality_per_dollar=qpd,
                value_score=round(score, 1),
            )
        )

    entries.sort(key=lambda e: e.value_score, reverse=True)
    return entries


def sort_entries(entries: list[LeaderboardEntry], sort_by: str) -> list[LeaderboardEntry]:
    """Re-sort leaderboard entries by a named criterion.

    Supported: value, quality, cost, speed, quality-per-dollar.
    """
    keys = {
        "value": (lambda e: e.value_score, True),
        "quality": (lambda e: e.quality, True),
        "cost": (lambda e: e.cost_usd, False),
        "speed": (lambda e: e.mean_tps, True),
        "quality-per-dollar": (lambda e: e.quality_per_dollar, True),
    }
    if sort_by not in keys:
        raise ValueError(f"Unknown sort key: {sort_by}")
    key, reverse = keys[sort_by]
    return sorted(entries, key=key, reverse=reverse)
