# Roadmap - bench-my-llm

## Shipped

### 💰 Cost-Adjusted Leaderboard
Rank models not just by quality and speed, but by cost-efficiency - showing the best bang-for-buck across providers. Available as the `leaderboard` command: composite value score (50% quality, 30% cost, 20% throughput), quality-per-dollar column, sortable, JSON output.

### 📄 HTML Report Export
Shareable, self-contained HTML benchmark reports with summary cards, latency tables, per-prompt quality bars, and model comparison summaries. Available as `export --format html`, with a `--title` option. Inline CSS only, no JavaScript or external assets.

### 📊 Historical Trend Tracking
Store benchmark results over time and visualize performance trends as models get updated or new versions are released. Available as the `trends` command: per-model chronological series from a results directory, unicode sparklines, first-to-latest change with improvement direction (lower is better for ttft, latency, and cost), model filtering, and JSON output.

### 🦙 Ollama Auto-Detection
Automatically detect locally running Ollama models and benchmark them without manual configuration. The `ollama` command lists installed models (size, params, quantization, JSON output), and `run --ollama` / `compare --ollama` resolve model names against the local server and fill in the OpenAI-compatible endpoint and API key automatically.

### 🧮 Custom Prompt Suites from Files
Load your own prompt suites from a JSON or YAML file so benchmarks can use domain-specific prompts and references. Pass a file path to `--suite` on `run` or `compare`. Files are validated with friendly errors (missing text, unknown keys, bad max_tokens), prompts get sensible defaults, and YAML support comes via the optional `bench-my-llm[yaml]` extra. See `examples/custom_suite.json`.

### 📦 Suite Packs
Share and install community prompt suites as versioned packs. A registry is a JSON document (local file or HTTP URL) mapping pack names to suite files, with relative URLs resolved against the registry location. `suite install` fetches and validates a pack before writing it to `~/.bench-my-llm/suites`, `suite list` and `suite remove` manage installed packs, and `--suite <pack name>` works on `run` and `compare`. Default registry via `BENCH_MY_LLM_REGISTRY`, packs directory override via `BENCH_MY_LLM_HOME`, example registry in `examples/registry.json`.

### 🧪 Suite Pack Publishing
Generate a registry entry and validate a suite for sharing with a `suite publish` helper, so teams can maintain their own pack registries with less hand-editing. Without `--registry` it prints the validated entry as JSON; with `--registry` it merges the entry into a local registry file (created if missing), preserving other entries, storing relative URLs when the suite sits next to the registry, and requiring `--force` to overwrite an existing pack.

### 🔁 Suite Pack Updates
Keep installed packs current with `suite update`: re-fetches each pack's registry from the source recorded at install time, reinstalls packs whose registry version changed, and reports the rest as up to date. Update specific packs by name or all at once, `--force` to reinstall without a version change, `--registry` to point at a moved registry (the new source is recorded), `--json-output` for scripting, and per-pack error reporting with a nonzero exit if anything failed. Python API: `update_pack`, `update_all`, `UpdateResult`.

## v0.3 (Planned)

### 🧾 Run Metadata and Notes
Attach free-form notes and captured environment metadata (endpoint, temperature, suite version) to saved benchmark results so historical comparisons stay interpretable.

---

Have ideas? Open an issue or start a discussion!
