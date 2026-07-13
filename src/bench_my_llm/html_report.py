"""Self-contained HTML report rendering for benchmark results.

Produces a single HTML document with inline CSS only: no JavaScript,
no external assets, safe to email or attach to a PR. All user-derived
strings (model names, categories, prompt text) are HTML-escaped.
"""

from __future__ import annotations

import html

from .metrics import RunMetrics, score_quality
from .runner import BenchmarkRun

_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 960px; margin: 2rem auto; padding: 0 1rem;
  background: #0f1117; color: #e6e6e6;
}
h1 { font-size: 1.6rem; }
h2 { font-size: 1.2rem; margin-top: 2.5rem; border-bottom: 1px solid #2a2d36; padding-bottom: .4rem; }
.meta { color: #9aa0ab; font-size: .9rem; }
.cards { display: flex; flex-wrap: wrap; gap: .8rem; margin: 1rem 0; }
.card {
  background: #1a1d26; border: 1px solid #2a2d36; border-radius: 8px;
  padding: .8rem 1.2rem; min-width: 120px;
}
.card .value { font-size: 1.3rem; font-weight: 600; }
.card .label { color: #9aa0ab; font-size: .8rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #2a2d36; }
th { color: #9aa0ab; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { background: #2a2d36; border-radius: 4px; height: 10px; width: 120px; display: inline-block; vertical-align: middle; }
.bar > span { display: block; height: 100%; border-radius: 4px; background: #4f8ef7; }
.bar.quality > span { background: #3fb950; }
.footer { margin-top: 3rem; color: #9aa0ab; font-size: .8rem; }
""".strip()


def _bar_width(value: float, maximum: float) -> float:
    """Percentage width for a bar, clamped to 0-100."""
    if maximum <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (value / maximum) * 100.0)), 1)


def _bar(value: float, maximum: float, kind: str = "") -> str:
    cls = f"bar {kind}".strip()
    return f'<span class="{cls}"><span style="width: {_bar_width(value, maximum)}%"></span></span>'


def _card(value: str, label: str) -> str:
    return (
        f'<div class="card"><div class="value">{html.escape(value)}</div>'
        f'<div class="label">{html.escape(label)}</div></div>'
    )


def _comparison_table(runs: list[tuple[BenchmarkRun, RunMetrics]]) -> str:
    max_quality = max((m.mean_quality_score for _, m in runs), default=0.0)
    max_tps = max((m.mean_tps for _, m in runs), default=0.0)

    rows = []
    for _, m in runs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(m.model)}</td>"
            f'<td class="num">{m.mean_quality_score:.1%} {_bar(m.mean_quality_score, max_quality, "quality")}</td>'
            f'<td class="num">{m.mean_tps} {_bar(m.mean_tps, max_tps)}</td>'
            f'<td class="num">{m.ttft.p50_ms}</td>'
            f'<td class="num">{m.total_latency.p50_ms}</td>'
            f'<td class="num">${m.estimated_cost_usd:.4f}</td>'
            "</tr>"
        )
    return (
        "<h2>Model Comparison</h2>\n<table>\n"
        '<tr><th>Model</th><th class="num">Quality</th><th class="num">Mean TPS</th>'
        '<th class="num">TTFT p50 (ms)</th><th class="num">Latency p50 (ms)</th>'
        '<th class="num">Cost</th></tr>\n'
        + "\n".join(rows)
        + "\n</table>"
    )


def _run_section(run: BenchmarkRun, metrics: RunMetrics) -> str:
    parts = [f"<h2>{html.escape(metrics.model)}</h2>"]
    ts = f" | {html.escape(run.timestamp[:19])}" if run.timestamp else ""
    parts.append(
        f'<p class="meta">Suite: {html.escape(metrics.suite)} | '
        f"Prompts: {metrics.num_prompts}{ts}</p>"
    )

    parts.append('<div class="cards">')
    parts.append(_card(f"{metrics.mean_quality_score:.1%}", "Quality"))
    parts.append(_card(f"{metrics.mean_tps} tok/s", "Mean TPS"))
    parts.append(_card(f"{metrics.ttft.p50_ms} ms", "TTFT p50"))
    parts.append(_card(f"${metrics.estimated_cost_usd:.4f}", "Est. Cost"))
    parts.append("</div>")

    parts.append("<h3>Latency</h3>")
    parts.append("<table>")
    parts.append(
        '<tr><th>Metric</th><th class="num">TTFT (ms)</th>'
        '<th class="num">Total Latency (ms)</th></tr>'
    )
    for label, ttft_val, total_val in (
        ("p50", metrics.ttft.p50_ms, metrics.total_latency.p50_ms),
        ("p95", metrics.ttft.p95_ms, metrics.total_latency.p95_ms),
        ("p99", metrics.ttft.p99_ms, metrics.total_latency.p99_ms),
        ("Mean", metrics.ttft.mean_ms, metrics.total_latency.mean_ms),
    ):
        parts.append(
            f'<tr><td>{label}</td><td class="num">{ttft_val}</td>'
            f'<td class="num">{total_val}</td></tr>'
        )
    parts.append("</table>")

    parts.append("<h3>Per-Prompt Results</h3>")
    parts.append("<table>")
    parts.append(
        '<tr><th>#</th><th>Category</th><th>Prompt</th><th class="num">TTFT (ms)</th>'
        '<th class="num">Total (ms)</th><th class="num">TPS</th>'
        '<th class="num">Quality</th></tr>'
    )
    for i, r in enumerate(run.results, 1):
        q = score_quality(r.response_text, r.reference)
        prompt_preview = r.prompt_text[:60] + ("..." if len(r.prompt_text) > 60 else "")
        parts.append(
            "<tr>"
            f'<td class="num">{i}</td>'
            f"<td>{html.escape(r.category)}</td>"
            f"<td>{html.escape(prompt_preview)}</td>"
            f'<td class="num">{r.ttft_ms}</td>'
            f'<td class="num">{r.total_latency_ms}</td>'
            f'<td class="num">{r.tokens_per_second}</td>'
            f'<td class="num">{q:.0%} {_bar(q, 1.0, "quality")}</td>'
            "</tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


def render_html(
    runs: list[tuple[BenchmarkRun, RunMetrics]],
    title: str = "Benchmark Report",
) -> str:
    """Render benchmark runs as a self-contained HTML document.

    Args:
        runs: Pairs of (run, computed metrics), one per model.
        title: Document title shown in the header and browser tab.

    Returns:
        A complete HTML document as a string.
    """
    if not runs:
        raise ValueError("No runs to render")

    body: list[str] = [f"<h1>🏎️ {html.escape(title)}</h1>"]

    if len(runs) > 1:
        body.append(_comparison_table(runs))

    for run, metrics in runs:
        body.append(_run_section(run, metrics))

    body.append('<p class="footer">Generated by bench-my-llm</p>')

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>\n{_CSS}\n</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )
