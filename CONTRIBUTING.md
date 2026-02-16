# Contributing to bench-my-llm

Thanks for your interest in contributing! 🏎️

## Getting Started

1. Fork and clone the repo
2. Create a virtual environment and install dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Run the tests:

```bash
pytest
```

## Development Workflow

1. Create a branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run tests and linting: `pytest && ruff check src/ tests/`
4. Commit with a clear message
5. Open a pull request

## What to Contribute

- **New prompt suites** - Add domain-specific prompt collections
- **Better quality scoring** - Improve the reference answer comparison
- **Cost table updates** - Keep model pricing up to date
- **New output formats** - CSV, HTML, Markdown reports
- **Bug fixes** - Always welcome

## Code Style

- Python 3.10+ with type hints
- Format with `ruff format`
- Lint with `ruff check`
- Tests for all new features

## Adding a Prompt Suite

1. Add your suite to `src/bench_my_llm/prompts.py`
2. Register it in the `SUITES` dict
3. Add tests in `tests/`

## Reporting Issues

Open a GitHub issue with:
- What you expected
- What happened
- Steps to reproduce
- Python version and OS
