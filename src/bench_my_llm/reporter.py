"""Rich terminal reporter for benchmark results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .metrics import compute_metrics
from .runner import BenchmarkRun


def _quality_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "yellow"
    return "red"


def _tps_color(tps: float) -> str:
    if tps >= 80:
        return "green"
    if tps >= 40:
        return "yellow"
    return "red"


def print_report(run: BenchmarkRun, console: Console | None = None) -> None:
    """Print a beautiful terminal report for a benchmark run."""
    console = console or Console()
    metrics = compute_metrics(run)

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold cyan]bench-my-llm[/] results for [bold white]{metrics.model}[/]\n"
            f"Suite: [yellow]{metrics.suite}[/] | Prompts: {metrics.num_prompts} | "
            f"Cost: [green]${metrics.estimated_cost_usd:.4f}[/]",
            title="[bold]🏎️  Benchmark Report[/]",
            border_style="cyan",
        )
    )

    # Summary table
    summary = Table(title="Latency Summary", border_style="blue", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("TTFT (ms)", justify="right")
    summary.add_column("Total Latency (ms)", justify="right")

    summary.add_row("p50", f"{metrics.ttft.p50_ms}", f"{metrics.total_latency.p50_ms}")
    summary.add_row("p95", f"{metrics.ttft.p95_ms}", f"{metrics.total_latency.p95_ms}")
    summary.add_row("p99", f"{metrics.ttft.p99_ms}", f"{metrics.total_latency.p99_ms}")
    summary.add_row("Mean", f"{metrics.ttft.mean_ms}", f"{metrics.total_latency.mean_ms}")
    summary.add_row("Min", f"{metrics.ttft.min_ms}", f"{metrics.total_latency.min_ms}")
    summary.add_row("Max", f"{metrics.ttft.max_ms}", f"{metrics.total_latency.max_ms}")

    console.print(summary)

    # Throughput & quality
    tq = Table(title="Throughput & Quality", border_style="green", show_lines=True)
    tq.add_column("Metric", style="bold")
    tq.add_column("Value", justify="right")

    tps_text = Text(f"{metrics.mean_tps} tok/s", style=_tps_color(metrics.mean_tps))
    quality_text = Text(
        f"{metrics.mean_quality_score:.1%}",
        style=_quality_color(metrics.mean_quality_score),
    )

    tq.add_row("Mean TPS", tps_text)
    tq.add_row("Median TPS", f"{metrics.median_tps} tok/s")
    tq.add_row("Quality Score", quality_text)
    tq.add_row("Prompt Tokens", f"{metrics.total_prompt_tokens:,}")
    tq.add_row("Completion Tokens", f"{metrics.total_completion_tokens:,}")
    tq.add_row("Estimated Cost", f"${metrics.estimated_cost_usd:.4f}")

    console.print(tq)

    # Per-prompt detail
    detail = Table(title="Per-Prompt Results", border_style="magenta")
    detail.add_column("#", justify="right", style="dim")
    detail.add_column("Category", style="cyan")
    detail.add_column("TTFT (ms)", justify="right")
    detail.add_column("Total (ms)", justify="right")
    detail.add_column("TPS", justify="right")
    detail.add_column("Tokens", justify="right")
    detail.add_column("Quality", justify="right")

    from .metrics import score_quality

    for i, r in enumerate(run.results, 1):
        q = score_quality(r.response_text, r.reference)
        detail.add_row(
            str(i),
            r.category,
            f"{r.ttft_ms}",
            f"{r.total_latency_ms}",
            f"{r.tokens_per_second}",
            f"{r.tokens_generated}",
            Text(f"{q:.0%}", style=_quality_color(q)),
        )

    console.print(detail)
    console.print()
