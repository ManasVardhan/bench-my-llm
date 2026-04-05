"""CLI entry point for bench-my-llm."""

from __future__ import annotations

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .compare import compare_runs
from .prompts import get_suite, SUITES
from .reporter import print_report
from .runner import BenchmarkRun, run_benchmark


console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="bench-my-llm")
def cli() -> None:
    """🏎️  bench-my-llm: Dead-simple LLM benchmarking CLI."""


@cli.command()
@click.option("--model", "-m", required=True, help="Model name (e.g. gpt-4o)")
@click.option(
    "--suite",
    "-s",
    default="all",
    type=click.Choice(list(SUITES.keys())),
    help="Prompt suite to run",
)
@click.option("--base-url", "-u", default=None, help="API base URL")
@click.option("--api-key", "-k", default=None, help="API key (defaults to OPENAI_API_KEY)")
@click.option("--temperature", "-t", default=0.0, type=float, help="Sampling temperature")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def run(
    model: str,
    suite: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    output: str | None,
) -> None:
    """Run a benchmark suite against a model."""
    prompt_suite = get_suite(suite)
    console.print(
        f"\n[bold cyan]🏎️  Benchmarking[/] [white]{model}[/] "
        f"with [yellow]{prompt_suite.name}[/] suite ({len(prompt_suite.prompts)} prompts)\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=len(prompt_suite.prompts))

        def on_progress(i: int, total: int, result: object) -> None:
            progress.update(task, completed=i, description=f"Prompt {i}/{total}")

        benchmark_run = run_benchmark(
            model=model,
            suite=prompt_suite,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            on_progress=on_progress,
        )

    print_report(benchmark_run, console)

    if output:
        benchmark_run.save(output)
        console.print(f"[green]Results saved to {output}[/]\n")


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option(
    "--suite",
    "-s",
    default="reasoning",
    type=click.Choice(list(SUITES.keys())),
    help="Prompt suite to run",
)
@click.option("--base-url", "-u", default=None, help="API base URL")
@click.option("--api-key", "-k", default=None, help="API key")
@click.option("--temperature", "-t", default=0.0, type=float)
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def compare(
    models: tuple[str, ...],
    suite: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    output: str | None,
) -> None:
    """Compare two or more models side-by-side."""
    if len(models) < 2:
        console.print("[red]Provide at least 2 model names to compare.[/]")
        raise SystemExit(1)

    prompt_suite = get_suite(suite)
    runs: list[BenchmarkRun] = []

    for model in models:
        console.print(f"\n[cyan]Benchmarking {model}...[/]")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
            task = prog.add_task(f"{model}", total=len(prompt_suite.prompts))

            def on_progress(i: int, total: int, result: object, _t=task, _p=prog) -> None:
                _p.update(_t, completed=i)

            r = run_benchmark(
                model=model,
                suite=prompt_suite,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                on_progress=on_progress,
            )
            runs.append(r)

    compare_runs(runs, console)

    if output:
        import json
        from pathlib import Path

        data = [
            {
                "model": r.model,
                "suite": r.suite_name,
                "timestamp": r.timestamp,
                "results": [res.to_dict() for res in r.results],
            }
            for r in runs
        ]
        Path(output).write_text(json.dumps(data, indent=2))
        console.print(f"[green]Results saved to {output}[/]\n")


@cli.command()
def models() -> None:
    """List models with known pricing (used for cost estimation)."""
    from rich.table import Table

    from .metrics import COST_TABLE, DEFAULT_COST

    table = Table(title="Known Model Pricing", border_style="green", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Input ($/1K tok)", justify="right", style="cyan")
    table.add_column("Output ($/1K tok)", justify="right", style="yellow")

    for model, (inp, out) in COST_TABLE.items():
        table.add_row(model, f"${inp:.6f}", f"${out:.6f}")

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Unknown models use default pricing: "
        f"${DEFAULT_COST[0]}/1K input, ${DEFAULT_COST[1]}/1K output[/]\n"
    )


@cli.command()
def suites() -> None:
    """List available prompt suites with descriptions."""
    from rich.table import Table

    table = Table(title="Available Prompt Suites", border_style="cyan", show_lines=True)
    table.add_column("Suite", style="bold cyan")
    table.add_column("Prompts", justify="right")
    table.add_column("Description")

    for name, suite in SUITES.items():
        table.add_row(name, str(len(suite.prompts)), suite.description)

    console.print()
    console.print(table)
    console.print()


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
def report(results_file: str) -> None:
    """Display a report from a saved results JSON file."""
    import json
    from pathlib import Path

    from .runner import BenchmarkResult

    try:
        data = json.loads(Path(results_file).read_text())
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON in {results_file}: {exc}[/]")
        raise SystemExit(1)

    # Comparison format: list of run objects
    if isinstance(data, list):
        runs: list[BenchmarkRun] = []
        for item in data:
            if not isinstance(item, dict) or "model" not in item or "results" not in item:
                console.print("[red]Unexpected format in results file.[/]")
                raise SystemExit(1)
            r = BenchmarkRun(
                model=item["model"],
                suite_name=item.get("suite", "unknown"),
                base_url=item.get("base_url", ""),
                timestamp=item.get("timestamp", ""),
            )
            r.results = [BenchmarkResult(**res) for res in item["results"]]
            runs.append(r)

        if len(runs) == 1:
            print_report(runs[0], console)
        else:
            compare_runs(runs, console)
    # Single run format: plain dict
    elif isinstance(data, dict):
        run = BenchmarkRun.load(results_file)
        print_report(run, console)
    else:
        console.print("[red]Unrecognized results file format.[/]")
        raise SystemExit(1)


@cli.command(name="export")
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["markdown", "csv", "json"]),
              default="markdown", help="Export format.")
@click.option("--output", "-o", default=None, help="Output file (prints to stdout if omitted).")
def export_cmd(results_file: str, fmt: str, output: str | None) -> None:
    """Export benchmark results to markdown, CSV, or JSON.

    Example: bench-my-llm export results.json --format markdown
    """
    import csv
    import io
    import json
    from pathlib import Path

    from .metrics import compute_metrics, score_quality
    from .runner import BenchmarkResult

    try:
        data = json.loads(Path(results_file).read_text())
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON in {results_file}: {exc}[/]")
        raise SystemExit(1)

    # Normalize to list of runs
    if isinstance(data, dict):
        run_data = [data]
    elif isinstance(data, list):
        run_data = data
    else:
        console.print("[red]Unrecognized results format.[/]")
        raise SystemExit(1)

    runs_with_metrics = []
    for item in run_data:
        if not isinstance(item, dict) or "results" not in item:
            console.print("[red]Invalid run data.[/]")
            raise SystemExit(1)
        run = BenchmarkRun(
            model=item.get("model", "unknown"),
            suite_name=item.get("suite", "unknown"),
            base_url=item.get("base_url", ""),
            timestamp=item.get("timestamp", ""),
        )
        run.results = [BenchmarkResult(**r) for r in item["results"]]
        metrics = compute_metrics(run)
        runs_with_metrics.append((run, metrics))

    if fmt == "markdown":
        lines = []
        for run, metrics in runs_with_metrics:
            lines.append(f"# Benchmark: {metrics.model}")
            lines.append("")
            lines.append(f"Suite: {metrics.suite} | Prompts: {metrics.num_prompts} | "
                         f"Cost: ${metrics.estimated_cost_usd:.4f}")
            lines.append("")
            lines.append("## Latency Summary")
            lines.append("")
            lines.append("| Metric | TTFT (ms) | Total Latency (ms) |")
            lines.append("|--------|-----------|---------------------|")
            lines.append(f"| p50 | {metrics.ttft.p50_ms} | {metrics.total_latency.p50_ms} |")
            lines.append(f"| p95 | {metrics.ttft.p95_ms} | {metrics.total_latency.p95_ms} |")
            lines.append(f"| p99 | {metrics.ttft.p99_ms} | {metrics.total_latency.p99_ms} |")
            lines.append(f"| Mean | {metrics.ttft.mean_ms} | {metrics.total_latency.mean_ms} |")
            lines.append("")
            lines.append("## Throughput and Quality")
            lines.append("")
            lines.append(f"- Mean TPS: {metrics.mean_tps} tok/s")
            lines.append(f"- Median TPS: {metrics.median_tps} tok/s")
            lines.append(f"- Quality Score: {metrics.mean_quality_score:.1%}")
            lines.append(f"- Estimated Cost: ${metrics.estimated_cost_usd:.4f}")
            lines.append("")
            lines.append("## Per-Prompt Results")
            lines.append("")
            lines.append("| # | Category | TTFT (ms) | Total (ms) | TPS | Tokens | Quality |")
            lines.append("|---|----------|-----------|------------|-----|--------|---------|")
            for i, r in enumerate(run.results, 1):
                q = score_quality(r.response_text, r.reference)
                lines.append(
                    f"| {i} | {r.category} | {r.ttft_ms} | {r.total_latency_ms} | "
                    f"{r.tokens_per_second} | {r.tokens_generated} | {q:.0%} |"
                )
            lines.append("")

        text = "\n".join(lines)

    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "model", "suite", "category", "prompt", "ttft_ms",
            "total_latency_ms", "tokens_per_second", "tokens_generated",
            "prompt_tokens", "completion_tokens", "quality",
        ])
        for run, _ in runs_with_metrics:
            for r in run.results:
                q = score_quality(r.response_text, r.reference)
                writer.writerow([
                    r.model, run.suite_name, r.category,
                    r.prompt_text[:80], r.ttft_ms, r.total_latency_ms,
                    r.tokens_per_second, r.tokens_generated,
                    r.prompt_tokens, r.completion_tokens, f"{q:.3f}",
                ])
        text = buf.getvalue()

    else:  # json
        text = json.dumps(
            [{"model": m.model, "suite": m.suite, "num_prompts": m.num_prompts,
              "ttft_p50": m.ttft.p50_ms, "ttft_p95": m.ttft.p95_ms,
              "latency_p50": m.total_latency.p50_ms, "latency_p95": m.total_latency.p95_ms,
              "mean_tps": m.mean_tps, "quality": m.mean_quality_score,
              "cost_usd": m.estimated_cost_usd}
             for _, m in runs_with_metrics],
            indent=2,
        )

    if output:
        Path(output).write_text(text)
        console.print(f"[green]Exported to {output}[/]")
    else:
        click.echo(text)


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="Filter by model name.")
@click.option("--limit", "-n", default=20, help="Max number of runs to show.")
def history(results_dir: str, model: str | None, limit: int) -> None:
    """Show history of benchmark runs from a directory of JSON files.

    Point it at a directory containing benchmark result JSON files
    and get a summary table of all past runs.

    Example: bench-my-llm history ./results/
    """
    import json
    from pathlib import Path

    from rich.table import Table

    from .metrics import compute_metrics
    from .runner import BenchmarkResult

    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    if not json_files:
        console.print(f"[yellow]No JSON files found in {results_dir}[/]")
        return

    runs = []
    for f in json_files[:limit * 3]:  # read more than needed in case of filtering
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        items = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
        for item in items:
            if not isinstance(item, dict) or "results" not in item:
                continue
            run = BenchmarkRun(
                model=item.get("model", "unknown"),
                suite_name=item.get("suite", "unknown"),
                base_url=item.get("base_url", ""),
                timestamp=item.get("timestamp", ""),
            )
            run.results = [BenchmarkResult(**r) for r in item["results"]]

            if model and model.lower() not in run.model.lower():
                continue

            try:
                metrics = compute_metrics(run)
                runs.append((f.name, run, metrics))
            except (ValueError, ZeroDivisionError):
                continue

        if len(runs) >= limit:
            break

    runs = runs[:limit]

    if not runs:
        console.print("[yellow]No matching benchmark runs found.[/]")
        return

    console.print()
    table = Table(title="Benchmark History", border_style="cyan", show_lines=True)
    table.add_column("File", style="dim")
    table.add_column("Model", style="bold cyan")
    table.add_column("Suite")
    table.add_column("Prompts", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("Mean TPS", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Cost", justify="right", style="green")
    table.add_column("Timestamp", style="dim")

    for fname, run, m in runs:
        ts = run.timestamp[:19] if run.timestamp else ""
        table.add_row(
            fname, m.model, m.suite,
            str(m.num_prompts),
            f"{m.ttft.p50_ms}ms",
            f"{m.mean_tps}",
            f"{m.mean_quality_score:.0%}",
            f"${m.estimated_cost_usd:.4f}",
            ts,
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(runs)} run(s) from {results_dir}[/]\n")


if __name__ == "__main__":
    cli()
