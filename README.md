
# 🏎️ bench-my-llm

> **New here?** Start with the [Getting Started Guide](GETTING_STARTED.md).

[![PyPI version](https://img.shields.io/pypi/v/bench-my-llm)](https://pypi.org/project/bench-my-llm/)
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
- 💾 **Export to JSON** - Pipe into CI, dashboards, or your own tools

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

### Custom Prompts

Pass your own prompts file (JSON array):

```json
[
  {"text": "Explain quantum computing", "category": "factual", "reference": "...", "max_tokens": 256}
]
```

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
```

### Local Models (Ollama)

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
