# Roadmap - bench-my-llm

## Shipped

### 💰 Cost-Adjusted Leaderboard
Rank models not just by quality and speed, but by cost-efficiency - showing the best bang-for-buck across providers. Available as the `leaderboard` command: composite value score (50% quality, 30% cost, 20% throughput), quality-per-dollar column, sortable, JSON output.

### 📄 HTML Report Export
Shareable, self-contained HTML benchmark reports with summary cards, latency tables, per-prompt quality bars, and model comparison summaries. Available as `export --format html`, with a `--title` option. Inline CSS only, no JavaScript or external assets.

## v0.2 (Planned)

### 🦙 Ollama Auto-Detection
Automatically detect locally running Ollama models and add them to the benchmark suite without manual configuration.

### 📊 Historical Trend Tracking
Store benchmark results over time and visualize performance trends as models get updated or new versions are released.

---

Have ideas? Open an issue or start a discussion!
