
# 🏎️ bench-my-llm

> **New here?** Start with the [Getting Started Guide](GETTING_STARTED.md).

[![PyPI version](https://img.shields.io/pypi/v/bench-my-llm?color=orange)](https://pypi.org/project/bench-my-llm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/manasvardhan/bench-my-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/manasvardhan/bench-my-llm/actions)

**Stop guessing which model is faster. Measure it.**

Point `bench-my-llm` at any OpenAI-compatible API and get latency, throughput, cost, and quality metrics in seconds. Compare models side by side. Get a beautiful terminal report. Ship with confidence.

## ✨ Features

- 🔥 **TTFT Measurement** - Time to first token via streaming
- ⚡ **Tokens per Second** - Real throughput numbers
- 📊 **p50 / p95 / p99 Latencies** - Production-grade percentiles
- 💰 **Cost Estimation** - Know what you're spending
- 🎯 **Quality Scoring** - Compare responses against reference answers
- 🏁 **Model Comparison** - Side-by-side with winner highlights
- 📦 **Built-in Prompt Suites** - Reasoning, coding, creative, factual
- 🔌 **Any OpenAI-compatible API** - OpenAI, Anthropic, Ollama, vLLM, Together, and more
- 💾 **Export to JSON, CSV, Markdown, HTML** - Pipe into CI, dashboards, or share a report
- 📈 **Historical Trends** - Track quality, speed, and cost per model over time with sparklines
- 🦙 **Ollama Auto-Detection** - Find local models and benchmark them with a single flag
- 🧮 **Custom Prompt Suites** - Load your own domain-specific prompts from JSON or YAML files
- 📦 **Suite Packs** - Install shareable prompt suites by name from a simple JSON registry

## 🚀 Quick Start

```bash
pip install bench-my-llm
```

### Single Model Benchmark

```bash
bench-my-llm run --model gpt-4o --suite reasoning
```

```
┌──────────────────────────────────────────────────────────┐
│  🏎️  Benchmark Report                                    │
│  bench-my-llm results for gpt-4o                         │
│  Suite: reasoning | Prompts: 5 | Cost: $0.0043           │
└──────────────────────────────────────────────────────────┘

          Latency Summary
┌────────┬────────────┬────────────────────┐
│ Metric │ TTFT (ms)  │ Total Latency (ms) │
├────────┼────────────┼────────────────────┤
│ p50    │ 234.1      │ 1,523.4            │
│ p95    │ 312.7      │ 2,187.9            │
│ p99    │ 348.2      │ 2,401.3            │
│ Mean   │ 251.3      │ 1,687.2            │
└────────┴────────────┴────────────────────┘

       Throughput & Quality
┌───────────────────┬─────────────┐
│ Metric            │ Value       │
├───────────────────┼─────────────┤
│ Mean TPS          │ 67.3 tok/s  │
│ Median TPS        │ 64.8 tok/s  │
│ Quality Score     │ 82%         │
│ Estimated Cost    │ $0.0043     │
└───────────────────┴─────────────┘
```

### Model Comparison

```bash
bench-my-llm compare gpt-4o gpt-4o-mini --suite reasoning
```

```
┌──────────────────────────────────────────────────────────┐
│  🏁 Model Comparison                                     │
│  gpt-4o vs gpt-4o-mini                                   │
└──────────────────────────────────────────────────────────┘

              Head-to-Head
┌────────────────────────┬─────────┬─────────────┐
│ Metric                 │ gpt-4o  │ gpt-4o-mini │
├────────────────────────┼─────────┼─────────────┤
│ TTFT p50 (ms)          │ 234.1   │ 142.3  🏆   │
│ TTFT p95 (ms)          │ 312.7   │ 198.4  🏆   │
│ Total Latency p50 (ms) │ 1523.4  │ 876.2  🏆   │
│ Mean TPS               │ 67.3 🏆 │ 54.1        │
│ Cost (USD)             │ $0.0043 │ $0.0008 🏆  │
│ Quality Score          │ 0.82 🏆 │ 0.71        │
└────────────────────────┴─────────┴─────────────┘

🏆 Winner: gpt-4o-mini (4/6 metrics)
```

## 📖 Usage

### Custom Prompt Suites

Benchmark with your own domain-specific prompts by passing a JSON or
YAML file path to `--suite`:

```bash
bench-my-llm run --model gpt-4o --suite my_suite.json
bench-my-llm compare gpt-4o gpt-4o-mini --suite ./suites/support.yaml
```

Suite file format (see `examples/custom_suite.json` for a full example):

```json
{
  "name": "support-bot",
  "description": "Customer support prompts",
  "prompts": [
    {
      "text": "Explain our 30-day return policy in two sentences.",
      "category": "support",
      "reference": "Items can be returned within 30 days for a full refund.",
      "max_tokens": 256
    }
  ]
}
```

Only `text` is required per prompt. `category` defaults to `custom`,
`reference` to empty (skips quality scoring), and `max_tokens` to 512.
Files are validated with friendly error messages, including typo
detection for unknown keys. YAML files need the optional extra:
`pip install 'bench-my-llm[yaml]'`.

### Suite Packs

Share suites with your team and install them by name. A registry is a
JSON document (local file or HTTP URL) mapping pack names to suite
files:

```json
{
  "packs": {
    "support-basics": {
      "version": "1.0.0",
      "description": "Customer support prompts",
      "url": "support_basics.json"
    }
  }
}
```

Pack URLs can be absolute (http, https, or a filesystem path) or
relative to the registry location, so a registry can live next to its
suite files in a repo or on any static file host.

```bash
# Install a pack (validated before it lands on disk)
bench-my-llm suite install support-basics --registry https://example.com/registry.json

# Or set a default registry once
export BENCH_MY_LLM_REGISTRY=https://example.com/registry.json
bench-my-llm suite install support-basics

# See what is installed, then benchmark with a pack by name
bench-my-llm suite list
bench-my-llm run --model gpt-4o --suite support-basics

# Remove a pack
bench-my-llm suite remove support-basics
```

Installed packs live in `~/.bench-my-llm/suites` (override the base
directory with `BENCH_MY_LLM_HOME`). Try it locally with the bundled
example: `bench-my-llm suite install example-basics -r examples/registry.json`.

### Prompt Suites

| Suite | Description | Prompts |
|-------|-------------|---------|
| `reasoning` | Logic, math, step-by-step | 5 |
| `coding` | Code generation and explanation | 5 |
| `creative` | Writing, storytelling, metaphors | 5 |
| `factual` | Knowledge recall, definitions | 5 |
| `all` | Everything combined | 20 |

### Export Results

```bash
bench-my-llm run --model gpt-4o --suite all --output results.json
bench-my-llm report results.json

# Convert saved results to other formats
bench-my-llm export results.json --format markdown
bench-my-llm export results.json --format csv -o results.csv
```

### HTML Reports

Turn any saved results file into a shareable, self-contained HTML report
with summary cards, latency tables, per-prompt quality bars, and a model
comparison section for multi-model files. No JavaScript, no external
assets, safe to attach to a PR or email:

```bash
bench-my-llm export results.json --format html -o report.html
bench-my-llm export comparison.json --format html --title "GPT vs Claude" -o report.html
open report.html
```

### Cost-Adjusted Leaderboard

Rank saved runs by value, not just raw speed. The composite score weighs
quality (50%), cost (30%, lower is better), and throughput (20%):

```bash
bench-my-llm leaderboard results/*.json
bench-my-llm leaderboard a.json b.json --sort quality-per-dollar
bench-my-llm leaderboard results/*.json --json-output
```

Sort options: `value` (default), `quality`, `cost`, `speed`, `quality-per-dollar`.

### Historical Trends

Save each benchmark run to a results directory, then track how every
model's numbers move over time. Runs are grouped per model and ordered
chronologically, with a terminal sparkline and the change from the first
run to the latest:

```bash
bench-my-llm run -m gpt-4o -s reasoning -o results/gpt4o-$(date +%F).json
bench-my-llm trends ./results/
bench-my-llm trends ./results/ --metric latency
bench-my-llm trends ./results/ -m gpt-4o --metric cost --json-output
```

```
Model     Runs  Trend      First   Latest  Change
gpt-4o       5  ▂▃▅▆█        61%      74%  ▲ +21.3%
llama3       4  ▅▄▃▂         55%      48%  ▼ -12.7%
```

Metrics: `quality` (default), `tps`, `ttft`, `latency`, `cost`. For
`ttft`, `latency`, and `cost`, lower is better and the arrows account
for that.

### Local Models (Ollama)

bench-my-llm auto-detects a running Ollama server. List what is
installed locally:

```bash
bench-my-llm ollama
```

```
              Local Ollama Models (2)
Model         Params  Quant    Size    Family
llama3:8b     8B      Q4_0     4.7 GB  llama
phi3:mini     3.8B    Q4_K_M   2.3 GB  phi3
```

Then benchmark with the `--ollama` flag, no URL or key needed. Tagless
names resolve automatically (`llama3` finds `llama3:latest`):

```bash
bench-my-llm run --ollama -m llama3
bench-my-llm compare --ollama llama3 phi3 -s coding
bench-my-llm ollama --json-output          # machine-readable model list
bench-my-llm ollama --url http://gpu:11434 # remote Ollama host
```

If a model is not installed, the error lists everything that is. Manual
configuration still works for any OpenAI-compatible endpoint:

```bash
bench-my-llm run --model llama3 --base-url http://localhost:11434/v1 --api-key ollama
```

### CI Integration

Add to your GitHub Actions workflow:

```yaml
- name: Benchmark LLM
  run: |
    pip install bench-my-llm
    bench-my-llm run --model gpt-4o-mini --suite reasoning --output benchmark.json
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Upload results
  uses: actions/upload-artifact@v4
  with:
    name: benchmark-results
    path: benchmark.json
```

## 🛠️ Development

```bash
git clone https://github.com/manasvardhan/bench-my-llm.git
cd bench-my-llm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT. See [LICENSE](LICENSE).
