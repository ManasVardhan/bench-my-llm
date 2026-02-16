"""Side-by-side model comparison."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .metrics import RunMetrics, compute_metrics
from .runner import BenchmarkRun


def _winner_style(a: float, b: float, lower_is_better: bool = True) -> tuple[str, str]:
    """Return (style_a, style_b) highlighting the winner in green."""
    if lower_is_better:
        if a < b:
            return "bold green", "dim"
        elif b < a:
            return "dim", "bold green"
    else:
        if a > b:
            return "bold green", "dim"
        elif b > a:
            return "dim", "bold green"
    return "", ""


def compare_runs(runs: list[BenchmarkRun], console: Console | None = None) -> None:
    """Print a side-by-side comparison of two or more benchmark runs."""
    console = console or Console()

    if len(runs) < 2:
        console.print("[red]Need at least 2 runs to compare.[/]")
        return

    all_metrics: list[RunMetrics] = [compute_metrics(r) for r in runs]

    console.print()
    console.print(
        Panel(
            "[bold cyan]bench-my-llm[/] model comparison\n"
            + " vs ".join(f"[bold]{m.model}[/]" for m in all_metrics),
            title="[bold]🏁 Model Comparison[/]",
            border_style="cyan",
        )
    )

    table = Table(title="Head-to-Head", border_style="blue", show_lines=True)
    table.add_column("Metric", style="bold")

    for m in all_metrics:
        table.add_column(m.model, justify="right")

    # Rows: TTFT p50, TTFT p95, Total p50, Total p95, Mean TPS, Cost, Quality
    rows: list[tuple[str, list[float], bool]] = [
        ("TTFT p50 (ms)", [m.ttft.p50_ms for m in all_metrics], True),
        ("TTFT p95 (ms)", [m.ttft.p95_ms for m in all_metrics], True),
        ("Total Latency p50 (ms)", [m.total_latency.p50_ms for m in all_metrics], True),
        ("Total Latency p95 (ms)", [m.total_latency.p95_ms for m in all_metrics], True),
        ("Mean TPS", [m.mean_tps for m in all_metrics], False),
        ("Cost (USD)", [m.estimated_cost_usd for m in all_metrics], True),
        ("Quality Score", [m.mean_quality_score for m in all_metrics], False),
    ]

    for label, values, lower_better in rows:
        if len(values) == 2:
            sa, sb = _winner_style(values[0], values[1], lower_better)
            cells: list[Text | str] = [
                Text(f"{values[0]:.4f}" if "Cost" in label else f"{values[0]}", style=sa),
                Text(f"{values[1]:.4f}" if "Cost" in label else f"{values[1]}", style=sb),
            ]
        else:
            best = min(values) if lower_better else max(values)
            cells = []
            for v in values:
                style = "bold green" if v == best else "dim"
                cells.append(Text(f"{v:.4f}" if "Cost" in label else f"{v}", style=style))

        table.add_row(label, *cells)

    console.print(table)

    # Winner summary
    wins: dict[str, int] = {m.model: 0 for m in all_metrics}
    for _, values, lower_better in rows:
        best = min(values) if lower_better else max(values)
        idx = values.index(best)
        wins[all_metrics[idx].model] += 1

    winner = max(wins, key=wins.get)  # type: ignore[arg-type]
    console.print(
        f"\n[bold green]🏆 Winner: {winner}[/] ({wins[winner]}/{len(rows)} metrics)\n"
    )
