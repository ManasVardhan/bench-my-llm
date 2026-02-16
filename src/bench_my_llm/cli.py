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
@click.argument("results_file", type=click.Path(exists=True))
def report(results_file: str) -> None:
    """Display a report from a saved results JSON file."""
    import json
    from pathlib import Path

    data = json.loads(Path(results_file).read_text())

    # Handle both single run and comparison format
    if isinstance(data, list):
        runs = [BenchmarkRun.load(results_file)]  # try single
        # Actually re-parse
        from .runner import BenchmarkResult

        runs = []
        for item in data:
            r = BenchmarkRun(
                model=item["model"],
                suite_name=item["suite"],
                base_url=item.get("base_url", ""),
                timestamp=item.get("timestamp", ""),
            )
            r.results = [BenchmarkResult(**res) for res in item["results"]]
            runs.append(r)

        if len(runs) == 1:
            print_report(runs[0], console)
        else:
            compare_runs(runs, console)
    else:
        run = BenchmarkRun.load(results_file)
        print_report(run, console)


if __name__ == "__main__":
    cli()
