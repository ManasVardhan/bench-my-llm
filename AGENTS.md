# AGENTS.md - bench-my-llm

## Overview
- Dead-simple LLM benchmarking CLI. Point it at any OpenAI-compatible API and get TTFT (time to first token), throughput (tokens/second), p50/p95/p99 latencies, cost estimation, and quality scoring in seconds. Compare models side by side with winner highlights.
- For developers choosing between LLM models/providers who need real performance numbers, not marketing claims.
- Core value: streaming TTFT measurement, built-in prompt suites (reasoning, coding, creative, factual), side-by-side model comparison with trophy indicators, Rich terminal reports, JSON export for CI pipelines.

## Architecture

```
+----------------+     +------------------+     +-------------------+
|   CLI (click)  | --> |   Runner         | --> | OpenAI-compatible |
|  run, compare, |     | run_benchmark()  |     |   API endpoint    |
|  models, suites|     | run_single_prompt|     | (OpenAI, Ollama,  |
|  report        |     | (streaming)      |     |  vLLM, Together)  |
+----------------+     +------------------+     +-------------------+
       |                       |
       v                       v
+----------------+     +------------------+
|   Reporter     |     |   Metrics        |
| print_report() |     | compute_metrics()|
| (Rich tables)  |     | estimate_cost()  |
+----------------+     | score_quality()  |
       |               | LatencyStats     |
       v               +------------------+
+----------------+
|   Compare      |
| compare_runs() |
| (side-by-side) |
+----------------+
```

**Data flow:**
1. CLI receives model name, suite, and options
2. `run_benchmark()` creates an OpenAI client, iterates through prompts
3. Each prompt is sent with `stream=True` to measure TTFT (time from request start to first chunk)
4. Chunks are collected, total latency and tokens/sec computed
5. `compute_metrics()` aggregates results: percentile latencies (numpy), cost estimation (model-specific pricing), quality scoring (Jaccard overlap with reference answers)
6. `print_report()` renders Rich tables. `compare_runs()` shows head-to-head with trophy icons.

## Directory Structure

```
bench-my-llm/
  .github/workflows/ci.yml        -- CI: lint + test on Python 3.10-3.12
  src/bench_my_llm/
    __init__.py                    -- __version__ = "0.1.1"
    __main__.py                    -- python -m bench_my_llm entry
    cli.py                         -- Click CLI: run, compare, models, suites, report
    runner.py                      -- BenchmarkRun, BenchmarkResult, run_benchmark(), run_single_prompt()
    metrics.py                     -- RunMetrics, LatencyStats, compute_metrics(), estimate_cost(), score_quality(), COST_TABLE
    prompts.py                     -- Prompt, PromptSuite, built-in suites (reasoning, coding, creative, factual, all)
    reporter.py                    -- print_report() with Rich panels and tables
    compare.py                     -- compare_runs() with trophy-annotated head-to-head table
  tests/                           -- 183 tests across 10 test files
    test_bench.py                  -- Core benchmark tests
    test_cli.py                    -- CLI command tests
    test_cli_extended.py           -- Extended CLI tests
    test_compare.py                -- Comparison tests
    test_reporter.py               -- Reporter output tests
    test_runner_mock.py            -- Runner tests with mocked API
    test_runner_extended.py        -- Extended runner tests
    test_metrics_extended.py       -- Metrics computation tests
    test_new_features.py           -- New feature tests
    test_error_paths.py            -- Error handling tests
  pyproject.toml                   -- Hatchling build, metadata
  README.md                        -- Full docs
  ROADMAP.md                       -- v0.2 plans
  CONTRIBUTING.md                  -- Contribution guidelines
  GETTING_STARTED.md               -- Quick start guide
  LICENSE                          -- MIT
```

## Core Concepts

- **BenchmarkResult**: Single prompt result. Fields: model, prompt_text, category, response_text, ttft_ms, total_latency_ms, tokens_generated, tokens_per_second, prompt_tokens, completion_tokens, reference.
- **BenchmarkRun**: Collection of results from a full run. Has model, suite_name, base_url, timestamp. Methods: `save(path)`, `load(path)`.
- **LatencyStats**: Percentile stats: p50_ms, p95_ms, p99_ms, mean_ms, min_ms, max_ms. Computed via `numpy.percentile()`.
- **RunMetrics**: Aggregated metrics: ttft (LatencyStats), total_latency (LatencyStats), mean_tps, median_tps, total_prompt_tokens, total_completion_tokens, estimated_cost_usd, mean_quality_score.
- **COST_TABLE**: Dict mapping model prefixes to (input_per_1k, output_per_1k) USD tuples. Covers OpenAI, Anthropic, Meta (Llama), Mistral, Google (Gemini), DeepSeek. Unknown models fall back to DEFAULT_COST = (0.002, 0.008).
- **score_quality()**: Jaccard word overlap between response and reference. Returns 1.0 if no reference provided.
- **Prompt**: Frozen dataclass with text, category, reference, max_tokens.
- **PromptSuite**: Named collection of Prompts. Built-in suites: reasoning (5), coding (5), creative (5), factual (5), all (20).
- **Streaming TTFT**: Uses `stream=True` on `client.chat.completions.create()`. TTFT = time from request start to first chunk with content.
- **Token estimation**: Approximate via `len(text.split()) * 1.3` (runner) for cost estimation when actual counts unavailable.

## API Reference

### Runner
```python
def run_benchmark(
    model: str,
    suite: PromptSuite,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    on_progress: Callable[[int, int, object], None] | None = None,
) -> BenchmarkRun

def run_single_prompt(
    client: OpenAI,
    model: str,
    prompt: Prompt,
    temperature: float = 0.0,
) -> BenchmarkResult
```

### Metrics
```python
def compute_metrics(run: BenchmarkRun) -> RunMetrics
def compute_latency_stats(values: list[float]) -> LatencyStats
def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float
def score_quality(response: str, reference: str) -> float  # 0.0 to 1.0
```

### Compare
```python
def compare_runs(runs: list[BenchmarkRun], console: Console | None = None) -> None
```

### Prompts
```python
def get_suite(name: str) -> PromptSuite  # raises KeyError if not found
SUITES: dict[str, PromptSuite]  # reasoning, coding, creative, factual, all
```

## CLI Commands

```bash
# Run benchmark on a single model
bench-my-llm run --model gpt-4o --suite reasoning
bench-my-llm run --model gpt-4o --suite all --output results.json
bench-my-llm run --model gpt-4o --suite coding --temperature 0.5

# Local models (Ollama, vLLM, etc.)
bench-my-llm run --model llama3 --base-url http://localhost:11434/v1 --api-key ollama

# Compare two or more models
bench-my-llm compare gpt-4o gpt-4o-mini --suite reasoning
bench-my-llm compare gpt-4o gpt-4o-mini claude-3-haiku --suite all --output comparison.json

# List models with known pricing
bench-my-llm models

# List available prompt suites
bench-my-llm suites

# Display report from saved results
bench-my-llm report results.json

# Version
bench-my-llm --version
```

## Configuration

- **API key**: `--api-key` flag or `OPENAI_API_KEY` env var
- **Base URL**: `--base-url` flag for non-OpenAI endpoints
- **Temperature**: `--temperature` flag (default 0.0)
- **No config files** needed

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

- **183 tests** across 10 test files
- Runner tests use mocked OpenAI client (no real API calls)
- Located in `tests/`

## Dependencies

- **click>=8.1**: CLI framework
- **openai>=1.0**: API client (used for any OpenAI-compatible endpoint)
- **rich>=13.0**: Terminal rendering (panels, tables, progress bars)
- **numpy>=1.24**: Percentile calculations
- **Python >=3.10**

## CI/CD

- **GitHub Actions** (`.github/workflows/ci.yml`)
- Matrix: Python 3.10, 3.11, 3.12
- Steps: install, ruff lint, pytest
- Triggers: push/PR to main

## Current Status

- **Version**: 0.1.1
- **Published on PyPI**: yes (`pip install bench-my-llm`)
- **What works**: Streaming TTFT measurement, full latency percentiles (p50/p95/p99), throughput (TPS), cost estimation for 25+ models, quality scoring (Jaccard), side-by-side comparison with trophy icons, 4 built-in prompt suites (20 prompts), JSON save/load, Rich terminal reports, any OpenAI-compatible API
- **Known limitations**: Requires real API calls for actual benchmarking (tests mock the API). Quality scoring is word-overlap only (no semantic scoring). No HTML report export yet.
- **Roadmap (v0.2)**: HTML report export, Ollama auto-detection, cost-adjusted leaderboard, historical trend tracking

## Development Guide

```bash
git clone https://github.com/manasvardhan/bench-my-llm.git
cd bench-my-llm
pip install -e ".[dev]"
pytest
```

- **Build system**: Hatchling
- **Source layout**: `src/bench_my_llm/`
- **Adding a new prompt suite**: Add `PromptSuite` in `prompts.py`, register in `SUITES` dict
- **Adding model pricing**: Add entry to `COST_TABLE` in `metrics.py` (order matters: specific prefixes before generic)
- **Adding a new metric**: Extend `RunMetrics` in `metrics.py`, update `compute_metrics()`, `print_report()`, and `compare_runs()`
- **Code style**: Ruff, line length 100, target Python 3.10

## Git Conventions

- **Branch**: main
- **Commits**: Imperative style ("Add feature X", "Fix bug Y")
- Never use em dashes in commit messages or docs

## Context

- **Author**: Manas Vardhan (ManasVardhan on GitHub)
- **Part of**: A suite of AI agent tooling
- **Related repos**: llm-cost-guardian (cost tracking), agent-sentry (crash reporting), agent-replay (trace debugging), llm-shelter (safety guardrails), promptdiff (prompt versioning), mcp-forge (MCP server scaffolding)
- **PyPI package**: `bench-my-llm`
