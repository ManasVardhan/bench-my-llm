"""Historical trend tracking: how a model's benchmark numbers move over time.

Scans a directory of saved benchmark result files, groups runs by model,
orders them chronologically, and summarizes how quality, throughput,
latency, and cost changed between the first and latest run. Includes a
dependency-free unicode sparkline renderer for terminal visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .metrics import compute_metrics
from .runner import BenchmarkResult, BenchmarkRun

SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

# metric name -> (attribute on TrendPoint, lower_is_better)
TREND_METRICS: dict[str, tuple[str, bool]] = {
    "quality": ("quality", False),
    "tps": ("mean_tps", False),
    "ttft": ("ttft_p50_ms", True),
    "latency": ("latency_p50_ms", True),
    "cost": ("cost_usd", True),
}


@dataclass
class TrendPoint:
    """Metrics for one benchmark run, anchored to a point in time."""

    source_file: str
    timestamp: str
    sort_key: float
    model: str
    suite: str
    num_prompts: int
    quality: float
    mean_tps: float
    ttft_p50_ms: float
    latency_p50_ms: float
    cost_usd: float

    def metric(self, name: str) -> float:
        """Return the value of a named trend metric.

        Raises:
            ValueError: If the metric name is unknown.
        """
        if name not in TREND_METRICS:
            raise ValueError(f"Unknown metric: {name}")
        attr, _ = TREND_METRICS[name]
        return float(getattr(self, attr))

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "timestamp": self.timestamp,
            "suite": self.suite,
            "num_prompts": self.num_prompts,
            "quality": self.quality,
            "mean_tps": self.mean_tps,
            "ttft_p50_ms": self.ttft_p50_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "cost_usd": self.cost_usd,
        }


@dataclass
class ModelTrend:
    """Chronological series of benchmark runs for a single model."""

    model: str
    points: list[TrendPoint] = field(default_factory=list)

    def series(self, metric: str) -> list[float]:
        """Values of a metric across the run series, oldest first."""
        return [p.metric(metric) for p in self.points]

    def change(self, metric: str) -> float:
        """Absolute change in a metric from the first run to the latest."""
        values = self.series(metric)
        return round(values[-1] - values[0], 6)

    def change_pct(self, metric: str) -> float | None:
        """Percent change from first to latest run, or None if first is 0."""
        values = self.series(metric)
        if values[0] == 0:
            return None
        return round((values[-1] - values[0]) / abs(values[0]) * 100, 1)

    def direction(self, metric: str) -> str:
        """One of 'improving', 'declining', or 'flat' for a metric."""
        delta = self.change(metric)
        if delta == 0:
            return "flat"
        _, lower_is_better = TREND_METRICS[metric]
        improved = delta < 0 if lower_is_better else delta > 0
        return "improving" if improved else "declining"

    def to_dict(self, metric: str) -> dict:
        return {
            "model": self.model,
            "runs": len(self.points),
            "metric": metric,
            "values": self.series(metric),
            "first": self.series(metric)[0],
            "latest": self.series(metric)[-1],
            "change": self.change(metric),
            "change_pct": self.change_pct(metric),
            "direction": self.direction(metric),
            "points": [p.to_dict() for p in self.points],
        }


def _parse_timestamp(value: str) -> float | None:
    """Parse an ISO timestamp into an epoch float, or None if unparseable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load_trend_points(directory: str | Path) -> list[TrendPoint]:
    """Load trend points from every result JSON file in a directory.

    Files may contain single runs (dict) or comparisons (list of runs).
    Unreadable files, invalid JSON, and runs without results are skipped
    silently so one bad file never breaks the whole trend view. Points are
    ordered by run timestamp, falling back to file modification time when
    a run has no parseable timestamp.

    Raises:
        ValueError: If the path is not a directory.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    points: list[TrendPoint] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        items = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
        mtime = path.stat().st_mtime
        for item in items:
            if not isinstance(item, dict) or not item.get("results"):
                continue
            run = BenchmarkRun(
                model=item.get("model", "unknown"),
                suite_name=item.get("suite", "unknown"),
                base_url=item.get("base_url", ""),
                timestamp=item.get("timestamp", ""),
            )
            try:
                run.results = [BenchmarkResult(**r) for r in item["results"]]
                metrics = compute_metrics(run)
            except (TypeError, ValueError, ZeroDivisionError):
                continue

            sort_key = _parse_timestamp(run.timestamp)
            points.append(
                TrendPoint(
                    source_file=path.name,
                    timestamp=run.timestamp,
                    sort_key=sort_key if sort_key is not None else mtime,
                    model=metrics.model,
                    suite=metrics.suite,
                    num_prompts=metrics.num_prompts,
                    quality=metrics.mean_quality_score,
                    mean_tps=metrics.mean_tps,
                    ttft_p50_ms=metrics.ttft.p50_ms,
                    latency_p50_ms=metrics.total_latency.p50_ms,
                    cost_usd=metrics.estimated_cost_usd,
                )
            )

    points.sort(key=lambda p: p.sort_key)
    return points


def build_trends(
    points: list[TrendPoint],
    model_filter: str | None = None,
) -> list[ModelTrend]:
    """Group trend points by model into chronological series.

    The optional model filter matches case-insensitively on substrings.
    Models are returned in alphabetical order.

    Raises:
        ValueError: If no points match.
    """
    grouped: dict[str, list[TrendPoint]] = {}
    for point in points:
        if model_filter and model_filter.lower() not in point.model.lower():
            continue
        grouped.setdefault(point.model, []).append(point)

    if not grouped:
        raise ValueError("No benchmark runs match")

    trends = [ModelTrend(model=model, points=pts) for model, pts in grouped.items()]
    trends.sort(key=lambda t: t.model.lower())
    return trends


def sparkline(values: list[float]) -> str:
    """Render values as a unicode sparkline (one block char per value).

    Equal values render as a flat mid-height line. Empty input returns
    an empty string.
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return SPARK_CHARS[3] * len(values)
    span = hi - lo
    top = len(SPARK_CHARS) - 1
    return "".join(SPARK_CHARS[round((v - lo) / span * top)] for v in values)


def format_metric_value(metric: str, value: float) -> str:
    """Format a metric value for display (percent, ms, dollars, or tok/s)."""
    if metric == "quality":
        return f"{value:.0%}"
    if metric in ("ttft", "latency"):
        return f"{value:.0f}ms"
    if metric == "cost":
        return f"${value:.4f}"
    return f"{value:.1f}"
