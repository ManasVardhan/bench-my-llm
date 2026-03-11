"""Extended tests for the runner module."""

from __future__ import annotations

import json
from pathlib import Path


from bench_my_llm.runner import (
    BenchmarkResult,
    BenchmarkRun,
    _count_tokens_approx,
)


class TestCountTokensApprox:
    def test_empty_string(self):
        assert _count_tokens_approx("") == 1  # max(1, ...)

    def test_single_word(self):
        count = _count_tokens_approx("hello")
        assert count >= 1

    def test_multiple_words(self):
        count = _count_tokens_approx("hello world this is a test")
        assert count >= 6  # at least word count

    def test_longer_text(self):
        text = " ".join(["word"] * 100)
        count = _count_tokens_approx(text)
        assert count == 130  # 100 * 1.3


class TestBenchmarkResult:
    def test_to_dict(self):
        result = BenchmarkResult(
            model="test",
            prompt_text="q",
            category="reasoning",
            response_text="a",
            ttft_ms=100.0,
            total_latency_ms=500.0,
            tokens_generated=50,
            tokens_per_second=40.0,
            prompt_tokens=20,
            completion_tokens=50,
            reference="ref",
        )
        d = result.to_dict()
        assert d["model"] == "test"
        assert d["ttft_ms"] == 100.0
        assert d["reference"] == "ref"
        assert isinstance(d, dict)

    def test_default_reference(self):
        result = BenchmarkResult(
            model="test",
            prompt_text="q",
            category="coding",
            response_text="a",
            ttft_ms=100.0,
            total_latency_ms=500.0,
            tokens_generated=50,
            tokens_per_second=40.0,
            prompt_tokens=20,
            completion_tokens=50,
        )
        assert result.reference == ""


class TestBenchmarkRun:
    def test_save_and_load(self, tmp_path):
        run = _make_dummy_run()
        path = str(tmp_path / "test.json")
        run.save(path)

        loaded = BenchmarkRun.load(path)
        assert loaded.model == run.model
        assert loaded.suite_name == run.suite_name
        assert len(loaded.results) == len(run.results)

    def test_save_creates_valid_json(self, tmp_path):
        run = _make_dummy_run()
        path = str(tmp_path / "test.json")
        run.save(path)

        data = json.loads(Path(path).read_text())
        assert "model" in data
        assert "suite" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_load_preserves_values(self, tmp_path):
        run = _make_dummy_run()
        run.results[0].ttft_ms = 42.5
        run.results[0].tokens_per_second = 99.9
        path = str(tmp_path / "test.json")
        run.save(path)

        loaded = BenchmarkRun.load(path)
        assert loaded.results[0].ttft_ms == 42.5
        assert loaded.results[0].tokens_per_second == 99.9

    def test_empty_results(self, tmp_path):
        run = BenchmarkRun(
            model="empty",
            suite_name="test",
            base_url="",
            results=[],
            timestamp="2025-01-01T00:00:00Z",
        )
        path = str(tmp_path / "empty.json")
        run.save(path)
        loaded = BenchmarkRun.load(path)
        assert len(loaded.results) == 0

    def test_save_path_types(self, tmp_path):
        run = _make_dummy_run()
        # String path
        run.save(str(tmp_path / "str.json"))
        assert (tmp_path / "str.json").exists()
        # Path object
        run.save(tmp_path / "path.json")
        assert (tmp_path / "path.json").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_run() -> BenchmarkRun:
    results = [
        BenchmarkResult(
            model="test-model",
            prompt_text=f"prompt {i}",
            category="reasoning",
            response_text=f"response {i}",
            ttft_ms=100.0 + i * 50,
            total_latency_ms=500.0 + i * 100,
            tokens_generated=50,
            tokens_per_second=30.0,
            prompt_tokens=20,
            completion_tokens=50,
            reference="",
        )
        for i in range(3)
    ]
    return BenchmarkRun(
        model="test-model",
        suite_name="reasoning",
        base_url="https://api.openai.com/v1",
        results=results,
        timestamp="2025-01-01T00:00:00Z",
    )
