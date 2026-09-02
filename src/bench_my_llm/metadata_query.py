"""Filter and diff saved benchmark runs by their captured metadata.

Scans a directory of saved result files, pairs each run with its captured
metadata (endpoint, temperature, tool version, ...) and computed metrics,
then lets you filter to like-for-like configurations and diff any two runs
to see which metadata changed and how the numbers moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .metrics import compute_metrics
from .runner import BenchmarkResult, BenchmarkRun, RunMetadata

# metric name -> (attribute on LoadedRun, lower_is_better)
DIFF_METRICS: dict[str, tuple[str, bool]] = {
    "quality": ("quality", False),
    "tps": ("mean_tps", False),
    "ttft": ("ttft_p50_ms", True),
    "latency": ("latency_p50_ms", True),
    "cost": ("cost_usd", True),
}

# metadata field name -> attribute on RunMetadata
META_FIELDS: dict[str, str] = {
    "endpoint": "endpoint",
    "temperature": "temperature",
    "tool_version": "tool_version",
    "suite_version": "suite_version",
    "notes": "notes",
    "captured_at": "captured_at",
}


@dataclass
class LoadedRun:
    """A saved benchmark run paired with its metadata and computed metrics."""

    source_file: str
    model: str
    suite: str
    timestamp: str
    metadata: RunMetadata
    quality: float
    mean_tps: float
    ttft_p50_ms: float
    latency_p50_ms: float
    cost_usd: float

    def metric(self, name: str) -> float:
        """Return the value of a named diff metric.

        Raises:
            ValueError: If the metric name is unknown.
        """
        if name not in DIFF_METRICS:
            raise ValueError(f"Unknown metric: {name}")
        attr, _ = DIFF_METRICS[name]
        return float(getattr(self, attr))

    def meta_value(self, field: str) -> str:
        """Return a metadata field as a display string ('' if unset)."""
        attr = META_FIELDS[field]
        value = getattr(self.metadata, attr)
        if field == "temperature":
            return f"{float(value):g}"
        return str(value)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "model": self.model,
            "suite": self.suite,
            "timestamp": self.timestamp,
            "metadata": self.metadata.to_dict(),
            "quality": self.quality,
            "mean_tps": self.mean_tps,
            "ttft_p50_ms": self.ttft_p50_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "cost_usd": self.cost_usd,
        }


def _metadata_from(item: dict) -> RunMetadata:
    """Build a RunMetadata from a raw run dict, tolerating a missing block."""
    raw = item.get("metadata")
    if not isinstance(raw, dict):
        return RunMetadata()
    known = set(META_FIELDS.values())
    return RunMetadata(**{k: v for k, v in raw.items() if k in known})


def load_runs(directory: str | Path) -> list[LoadedRun]:
    """Load every saved run in a directory paired with metadata and metrics.

    Files may contain single runs (dict) or comparisons (list of runs).
    Runs from comparison files carry an empty metadata block. Unreadable
    files, invalid JSON, and runs without results are skipped silently.

    Raises:
        ValueError: If the path is not a directory.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    loaded: list[LoadedRun] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        items = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
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

            loaded.append(
                LoadedRun(
                    source_file=path.name,
                    model=metrics.model,
                    suite=metrics.suite,
                    timestamp=run.timestamp,
                    metadata=_metadata_from(item),
                    quality=metrics.mean_quality_score,
                    mean_tps=metrics.mean_tps,
                    ttft_p50_ms=metrics.ttft.p50_ms,
                    latency_p50_ms=metrics.total_latency.p50_ms,
                    cost_usd=metrics.estimated_cost_usd,
                )
            )

    return loaded


def filter_runs(
    runs: list[LoadedRun],
    *,
    model: str | None = None,
    endpoint: str | None = None,
    temperature: float | None = None,
    tool_version: str | None = None,
) -> list[LoadedRun]:
    """Filter runs to those matching every given metadata criterion.

    Model, endpoint, and tool_version match case-insensitively on
    substrings; temperature must match exactly. Criteria left as None
    are ignored, so passing none returns every run.
    """
    result = []
    for run in runs:
        if model and model.lower() not in run.model.lower():
            continue
        if endpoint and endpoint.lower() not in run.metadata.endpoint.lower():
            continue
        if tool_version and tool_version.lower() not in run.metadata.tool_version.lower():
            continue
        if temperature is not None and float(run.metadata.temperature) != temperature:
            continue
        result.append(run)
    return result


@dataclass
class FieldDiff:
    """One metadata field compared across two runs."""

    field: str
    old: str
    new: str

    @property
    def changed(self) -> bool:
        return self.old != self.new


@dataclass
class MetricDiff:
    """One metric compared across two runs, with direction of change."""

    metric: str
    old: float
    new: float

    @property
    def delta(self) -> float:
        return round(self.new - self.old, 6)

    @property
    def delta_pct(self) -> float | None:
        if self.old == 0:
            return None
        return round((self.new - self.old) / abs(self.old) * 100, 1)

    @property
    def direction(self) -> str:
        """One of 'improving', 'declining', or 'flat'."""
        if self.delta == 0:
            return "flat"
        _, lower_is_better = DIFF_METRICS[self.metric]
        improved = self.delta < 0 if lower_is_better else self.delta > 0
        return "improving" if improved else "declining"


@dataclass
class RunDiff:
    """The metadata and metric differences between two runs."""

    old: LoadedRun
    new: LoadedRun
    fields: list[FieldDiff]
    metrics: list[MetricDiff]

    @property
    def changed_fields(self) -> list[FieldDiff]:
        return [f for f in self.fields if f.changed]

    def to_dict(self) -> dict:
        return {
            "old": self.old.source_file,
            "new": self.new.source_file,
            "metadata_changes": [
                {"field": f.field, "old": f.old, "new": f.new}
                for f in self.changed_fields
            ],
            "metrics": [
                {
                    "metric": m.metric,
                    "old": m.old,
                    "new": m.new,
                    "delta": m.delta,
                    "delta_pct": m.delta_pct,
                    "direction": m.direction,
                }
                for m in self.metrics
            ],
        }


def diff_runs(old: LoadedRun, new: LoadedRun) -> RunDiff:
    """Diff two runs by metadata fields and computed metrics.

    Compares model, suite, and every captured metadata field, plus the
    five headline metrics. Direction is reported per metric using each
    metric's own better-is-lower convention.
    """
    fields = [
        FieldDiff("model", old.model, new.model),
        FieldDiff("suite", old.suite, new.suite),
    ]
    for field in META_FIELDS:
        fields.append(FieldDiff(field, old.meta_value(field), new.meta_value(field)))

    metrics = [
        MetricDiff(name, old.metric(name), new.metric(name)) for name in DIFF_METRICS
    ]
    return RunDiff(old=old, new=new, fields=fields, metrics=metrics)


def find_run(runs: list[LoadedRun], selector: str) -> LoadedRun:
    """Resolve a run by exact source file name or a unique substring.

    Raises:
        ValueError: If nothing matches or the selector is ambiguous.
    """
    exact = [r for r in runs if r.source_file == selector]
    if len(exact) == 1:
        return exact[0]
    matches = [r for r in runs if selector.lower() in r.source_file.lower()]
    if not matches:
        raise ValueError(f"No run matches '{selector}'")
    if len(matches) > 1:
        names = ", ".join(sorted({m.source_file for m in matches}))
        raise ValueError(f"'{selector}' is ambiguous, matches: {names}")
    return matches[0]
