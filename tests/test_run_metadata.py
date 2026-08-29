"""Tests for run metadata and notes (v0.3)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from bench_my_llm import __version__
from bench_my_llm.prompts import Prompt, PromptSuite
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun, RunMetadata, run_benchmark


def _result() -> BenchmarkResult:
    return BenchmarkResult(
        model="gpt-4o",
        prompt_text="hi",
        category="chat",
        response_text="hello",
        ttft_ms=10.0,
        total_latency_ms=20.0,
        tokens_generated=1,
        tokens_per_second=50.0,
        prompt_tokens=1,
        completion_tokens=1,
        reference="",
    )


def test_metadata_is_empty_by_default() -> None:
    assert RunMetadata().is_empty()


def test_metadata_not_empty_with_notes() -> None:
    assert not RunMetadata(notes="ran on prod").is_empty()


def test_metadata_not_empty_with_endpoint() -> None:
    assert not RunMetadata(endpoint="http://localhost:1234/v1").is_empty()


def test_temperature_alone_counts_as_empty() -> None:
    # Temperature always has a numeric default, so it should not by itself
    # make the metadata worth displaying.
    assert RunMetadata(temperature=0.7).is_empty()


def test_save_includes_metadata(tmp_path) -> None:
    run = BenchmarkRun(
        model="gpt-4o",
        suite_name="reasoning",
        base_url="http://localhost:1234/v1",
        results=[_result()],
        timestamp="2026-01-01T00:00:00+00:00",
        metadata=RunMetadata(
            notes="smoke test",
            endpoint="http://localhost:1234/v1",
            temperature=0.3,
            suite_version="reasoning (5 prompts)",
            tool_version="0.3.0",
            captured_at="2026-01-01T00:00:00+00:00",
        ),
    )
    path = tmp_path / "run.json"
    run.save(path)

    raw = json.loads(path.read_text())
    assert raw["metadata"]["notes"] == "smoke test"
    assert raw["metadata"]["temperature"] == 0.3
    assert raw["metadata"]["tool_version"] == "0.3.0"


def test_save_omits_metadata_when_none(tmp_path) -> None:
    run = BenchmarkRun(
        model="gpt-4o",
        suite_name="reasoning",
        base_url="",
        results=[_result()],
    )
    path = tmp_path / "run.json"
    run.save(path)
    assert "metadata" not in json.loads(path.read_text())


def test_save_load_roundtrip(tmp_path) -> None:
    meta = RunMetadata(
        notes="baseline",
        endpoint="https://api.openai.com/v1",
        temperature=0.0,
        suite_version="all (12 prompts)",
        tool_version="0.3.0",
        captured_at="2026-01-01T00:00:00+00:00",
    )
    run = BenchmarkRun(
        model="gpt-4o",
        suite_name="all",
        base_url="https://api.openai.com/v1",
        results=[_result()],
        timestamp="2026-01-01T00:00:00+00:00",
        metadata=meta,
    )
    path = tmp_path / "run.json"
    run.save(path)

    loaded = BenchmarkRun.load(path)
    assert loaded.metadata is not None
    assert loaded.metadata.notes == "baseline"
    assert loaded.metadata.suite_version == "all (12 prompts)"
    assert loaded.metadata.tool_version == "0.3.0"


def test_load_without_metadata_is_none(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-4o",
                "suite": "reasoning",
                "base_url": "",
                "timestamp": "",
                "results": [_result().to_dict()],
            }
        )
    )
    loaded = BenchmarkRun.load(path)
    assert loaded.metadata is None


def test_load_ignores_unknown_metadata_keys(tmp_path) -> None:
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-4o",
                "suite": "reasoning",
                "base_url": "",
                "timestamp": "",
                "results": [_result().to_dict()],
                "metadata": {"notes": "hi", "future_field": "ignored"},
            }
        )
    )
    loaded = BenchmarkRun.load(path)
    assert loaded.metadata is not None
    assert loaded.metadata.notes == "hi"


@patch("bench_my_llm.runner.OpenAI")
def test_run_benchmark_captures_metadata(mock_openai_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    def create_stream(**kwargs):
        choice = MagicMock()
        choice.delta.content = "answer"
        chunk = MagicMock()
        chunk.choices = [choice]
        return iter([chunk])

    mock_client.chat.completions.create.side_effect = create_stream

    suite = PromptSuite(
        name="reasoning",
        description="d",
        prompts=[Prompt(text="q", category="c", reference="", max_tokens=16)],
    )
    run = run_benchmark(
        model="m",
        suite=suite,
        base_url="http://localhost:9/v1",
        api_key="k",
        temperature=0.5,
        notes="captured note",
    )

    assert run.metadata is not None
    assert run.metadata.notes == "captured note"
    assert run.metadata.endpoint == "http://localhost:9/v1"
    assert run.metadata.temperature == 0.5
    assert run.metadata.tool_version == __version__
    assert "reasoning" in run.metadata.suite_version
    assert run.metadata.captured_at
