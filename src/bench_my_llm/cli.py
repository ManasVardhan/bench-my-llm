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


if __name__ == "__main__":
    cli()
