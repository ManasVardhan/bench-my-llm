# Getting Started with bench-my-llm

A step-by-step guide to get up and running from scratch.

## Prerequisites

You need **Python 3.10 or newer** installed on your machine.

**Check if you have Python:**
```bash
python3 --version
```
If you see `Python 3.10.x` or higher, you're good. If not, download it from [python.org](https://www.python.org/downloads/).

## Step 1: Clone the repository

```bash
git clone https://github.com/ManasVardhan/bench-my-llm.git
cd bench-my-llm
```

## Step 2: Create a virtual environment

```bash
python3 -m venv venv
```

**Activate it:**

- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

## Step 3: Install the package

```bash
pip install -e ".[dev]"
```

## Step 4: Run the tests

```bash
pytest tests/ -v
```

All 12 tests should pass.

## Step 5: Try it out (no API key needed)

### 5a. Explore the CLI

```bash
bench-my-llm --help
```

### 5b. View built-in prompt suites

The package comes with 20 pre-built prompts across 4 categories. You can explore them in Python:

```python
python3 -c "
from bench_my_llm.prompts import SUITES
for name, suite in SUITES.items():
    print(f'{name}: {len(suite.prompts)} prompts - {suite.description}')
"
```

You'll see:
- **reasoning** - Logic and math problems
- **coding** - Programming tasks
- **creative** - Creative writing prompts
- **factual** - Knowledge questions

### 5c. Test the metrics system (no API key needed)

Create a file called `test_it.py`:

```python
from bench_my_llm.metrics import compute_latency_stats, estimate_cost

# Simulate some latency measurements (in seconds)
latencies = [0.5, 0.8, 0.6, 1.2, 0.7, 0.9, 0.55, 0.65, 0.75, 0.85]

stats = compute_latency_stats(latencies)
print("Latency Stats:")
print(f"  Median (p50): {stats['p50']:.3f}s")
print(f"  p95: {stats['p95']:.3f}s")
print(f"  p99: {stats['p99']:.3f}s")
print(f"  Mean: {stats['mean']:.3f}s")

# Estimate cost
cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
print(f"\nEstimated cost for gpt-4o (1K in / 500 out): ${cost:.4f}")
```

Run it:
```bash
python test_it.py
```

## Step 6: Run a real benchmark (API key needed)

To benchmark actual models, you'll need an API key from at least one provider.

### Option A: OpenAI

```bash
export OPENAI_API_KEY="sk-your-key-here"
bench-my-llm run --model gpt-4o --suite reasoning
```

### Option B: Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
bench-my-llm run --model claude-3.5-sonnet --suite reasoning
```

### Option C: Local model (free, no API key)

If you have [Ollama](https://ollama.com) installed:

```bash
ollama pull llama3.2
bench-my-llm run --model ollama/llama3.2 --suite reasoning --base-url http://localhost:11434/v1
```

### Compare two models

```bash
bench-my-llm compare gpt-4o claude-3.5-sonnet --suite coding
```

This runs both models on the same prompts and shows a side-by-side comparison with winner highlights.

### Export results

```bash
bench-my-llm run --model gpt-4o --suite reasoning --output results.json
```

## Step 7: Run the linter (optional)

```bash
ruff check src/ tests/
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python3: command not found` | Install Python from [python.org](https://www.python.org/downloads/) |
| `No module named bench_my_llm` | Make sure you ran `pip install -e ".[dev]"` with the venv activated |
| `bench-my-llm: command not found` | Make sure your venv is activated |
| `AuthenticationError` | Set your API key: `export OPENAI_API_KEY=...` |
| Tests fail | Make sure you're on the latest `main` branch: `git pull origin main` |

## What's next?

- Read the full [README](README.md) for custom prompts, CI integration, and advanced options
- Check `examples/` for benchmark configuration files
- Try benchmarking models on your own custom prompts
