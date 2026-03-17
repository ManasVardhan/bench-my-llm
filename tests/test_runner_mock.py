"""Tests for runner.py - run_single_prompt and run_benchmark with mocked OpenAI.

Covers the streaming response path, TTFT measurement, token counting,
progress callbacks, and the full run_benchmark success flow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bench_my_llm.prompts import Prompt, PromptSuite
from bench_my_llm.runner import (
    BenchmarkResult,
    BenchmarkRun,
    _count_tokens_approx,
    run_benchmark,
    run_single_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str | None = None) -> MagicMock:
    """Create a mock streaming chunk."""
    chunk = MagicMock()
    if content is not None:
        choice = MagicMock()
        choice.delta.content = content
        chunk.choices = [choice]
    else:
        chunk.choices = []
    return chunk


def _make_stream(texts: list[str]) -> list[MagicMock]:
    """Create a list of mock streaming chunks from text fragments."""
    return [_make_chunk(t) for t in texts]


def _make_prompt(text: str = "What is 2+2?", category: str = "test") -> Prompt:
    return Prompt(text=text, category=category, reference="4", max_tokens=64)


def _make_suite(n: int = 2) -> PromptSuite:
    return PromptSuite(
        name="test-suite",
        description="Test",
        prompts=[_make_prompt(f"Question {i}") for i in range(n)],
    )


# ---------------------------------------------------------------------------
# _count_tokens_approx
# ---------------------------------------------------------------------------


class TestCountTokensApprox:
    def test_minimum_is_one(self) -> None:
        assert _count_tokens_approx("") >= 1

    def test_single_word(self) -> None:
        result = _count_tokens_approx("hello")
        assert result >= 1

    def test_scales_with_words(self) -> None:
        short = _count_tokens_approx("hello world")
        long = _count_tokens_approx("the quick brown fox jumps over the lazy dog")
        assert long > short

    def test_factor_is_1_3(self) -> None:
        # 10 words * 1.3 = 13
        result = _count_tokens_approx("one two three four five six seven eight nine ten")
        assert result == 13


# ---------------------------------------------------------------------------
# run_single_prompt
# ---------------------------------------------------------------------------


class TestRunSinglePrompt:
    def test_basic_streaming_response(self) -> None:
        """Verify TTFT, TPS, and response text from a mocked stream."""
        mock_client = MagicMock()
        chunks = _make_stream(["Hello", ", ", "world", "!"])
        mock_client.chat.completions.create.return_value = iter(chunks)

        prompt = _make_prompt("Say hello")
        result = run_single_prompt(mock_client, "test-model", prompt, temperature=0.5)

        assert isinstance(result, BenchmarkResult)
        assert result.model == "test-model"
        assert result.response_text == "Hello, world!"
        assert result.category == "test"
        assert result.prompt_text == "Say hello"
        assert result.ttft_ms >= 0  # mock is instant, so 0.0 is valid
        assert result.total_latency_ms >= 0
        assert result.tokens_generated >= 1
        assert result.prompt_tokens >= 1

    def test_empty_stream(self) -> None:
        """A stream with no content chunks should still return a result."""
        mock_client = MagicMock()
        # Chunks with no content
        empty_chunks = [_make_chunk(None), _make_chunk(None)]
        mock_client.chat.completions.create.return_value = iter(empty_chunks)

        prompt = _make_prompt()
        result = run_single_prompt(mock_client, "test-model", prompt)

        assert result.response_text == ""
        assert result.total_latency_ms >= 0  # mock is instant
        # TTFT should equal total latency when no first token
        assert result.ttft_ms == result.total_latency_ms

    def test_single_chunk(self) -> None:
        """Stream with a single content chunk."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([_make_chunk("42")])

        prompt = _make_prompt()
        result = run_single_prompt(mock_client, "m", prompt)

        assert result.response_text == "42"
        assert result.tokens_generated >= 1

    def test_temperature_passed_to_api(self) -> None:
        """Verify temperature is forwarded to the API call."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([_make_chunk("ok")])

        prompt = _make_prompt()
        run_single_prompt(mock_client, "m", prompt, temperature=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_max_tokens_passed(self) -> None:
        """Verify max_tokens from prompt is forwarded."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([_make_chunk("ok")])

        prompt = Prompt(text="test", category="t", max_tokens=128)
        run_single_prompt(mock_client, "m", prompt)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 128

    def test_streaming_enabled(self) -> None:
        """Verify stream=True is set."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([_make_chunk("ok")])

        prompt = _make_prompt()
        run_single_prompt(mock_client, "m", prompt)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_reference_preserved(self) -> None:
        """Reference from prompt should be in result."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([_make_chunk("four")])

        prompt = Prompt(text="2+2?", category="math", reference="4")
        result = run_single_prompt(mock_client, "m", prompt)
        assert result.reference == "4"

    def test_multi_chunk_concatenation(self) -> None:
        """Multiple chunks should be concatenated correctly."""
        mock_client = MagicMock()
        chunks = _make_stream(["The ", "answer ", "is ", "42."])
        mock_client.chat.completions.create.return_value = iter(chunks)

        prompt = _make_prompt()
        result = run_single_prompt(mock_client, "m", prompt)
        assert result.response_text == "The answer is 42."

    def test_tps_positive_for_content(self) -> None:
        """TPS should be positive when there is generated content."""
        mock_client = MagicMock()
        chunks = _make_stream(["word " * 20])
        mock_client.chat.completions.create.return_value = iter(chunks)

        prompt = _make_prompt()
        result = run_single_prompt(mock_client, "m", prompt)
        assert result.tokens_per_second > 0


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    @patch("bench_my_llm.runner.OpenAI")
    def test_basic_run(self, mock_openai_cls: MagicMock) -> None:
        """Full benchmark run with mocked client."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def create_stream(**kwargs):
            return iter(_make_stream(["response text here"]))

        mock_client.chat.completions.create.side_effect = create_stream

        suite = _make_suite(3)
        run = run_benchmark(
            model="test-model",
            suite=suite,
            api_key="test-key",
        )

        assert isinstance(run, BenchmarkRun)
        assert run.model == "test-model"
        assert run.suite_name == "test-suite"
        assert len(run.results) == 3
        assert run.timestamp  # should be set

    @patch("bench_my_llm.runner.OpenAI")
    def test_run_with_base_url(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(_make_stream(["ok"]))

        suite = _make_suite(1)
        run = run_benchmark(
            model="m",
            suite=suite,
            base_url="http://localhost:8080/v1",
            api_key="k",
        )

        assert run.base_url == "http://localhost:8080/v1"
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8080/v1",
            api_key="k",
        )

    @patch("bench_my_llm.runner.OpenAI")
    def test_run_default_base_url(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(_make_stream(["ok"]))

        suite = _make_suite(1)
        run = run_benchmark(model="m", suite=suite, api_key="k")

        assert run.base_url == "https://api.openai.com/v1"

    @patch("bench_my_llm.runner.OpenAI")
    def test_progress_callback(self, mock_openai_cls: MagicMock) -> None:
        """Progress callback should be called for each prompt."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(_make_stream(["ok"]))

        suite = _make_suite(3)
        calls: list[tuple[int, int]] = []

        def on_progress(i: int, total: int, result: object) -> None:
            calls.append((i, total))

        # Need to reset side_effect for multiple calls
        mock_client.chat.completions.create.side_effect = lambda **kw: iter(_make_stream(["ok"]))

        run_benchmark(
            model="m",
            suite=suite,
            api_key="k",
            on_progress=on_progress,
        )

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)

    @patch("bench_my_llm.runner.OpenAI")
    def test_no_progress_callback(self, mock_openai_cls: MagicMock) -> None:
        """Run should work without progress callback."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(_make_stream(["ok"]))

        suite = _make_suite(1)
        run = run_benchmark(model="m", suite=suite, api_key="k")
        assert len(run.results) == 1

    @patch("bench_my_llm.runner.OpenAI")
    def test_run_with_temperature(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(_make_stream(["ok"]))

        suite = _make_suite(1)
        run_benchmark(model="m", suite=suite, api_key="k", temperature=0.7)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch("bench_my_llm.runner.OpenAI")
    def test_results_have_correct_model(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = lambda **kw: iter(_make_stream(["ok"]))

        suite = _make_suite(2)
        run = run_benchmark(model="gpt-4o", suite=suite, api_key="k")

        for result in run.results:
            assert result.model == "gpt-4o"


# ---------------------------------------------------------------------------
# BenchmarkResult edge cases
# ---------------------------------------------------------------------------


class TestBenchmarkResultEdgeCases:
    def test_to_dict_round_trip(self) -> None:
        result = BenchmarkResult(
            model="m",
            prompt_text="q",
            category="test",
            response_text="a",
            ttft_ms=10.0,
            total_latency_ms=100.0,
            tokens_generated=5,
            tokens_per_second=50.0,
            prompt_tokens=3,
            completion_tokens=5,
            reference="ref",
        )
        d = result.to_dict()
        restored = BenchmarkResult(**d)
        assert restored.model == result.model
        assert restored.reference == result.reference
        assert restored.ttft_ms == result.ttft_ms

    def test_default_reference_empty(self) -> None:
        result = BenchmarkResult(
            model="m",
            prompt_text="q",
            category="t",
            response_text="a",
            ttft_ms=1.0,
            total_latency_ms=10.0,
            tokens_generated=1,
            tokens_per_second=1.0,
            prompt_tokens=1,
            completion_tokens=1,
        )
        assert result.reference == ""


# ---------------------------------------------------------------------------
# BenchmarkRun save/load edge cases
# ---------------------------------------------------------------------------


class TestBenchmarkRunSaveLoad:
    def test_save_creates_file(self, tmp_path) -> None:
        run = BenchmarkRun(model="m", suite_name="s", base_url="u")
        path = tmp_path / "out.json"
        run.save(str(path))
        assert path.exists()

    def test_save_load_preserves_results(self, tmp_path) -> None:
        run = BenchmarkRun(
            model="gpt-4o",
            suite_name="reasoning",
            base_url="https://api.openai.com/v1",
            timestamp="2025-01-01T00:00:00Z",
        )
        run.results = [
            BenchmarkResult(
                model="gpt-4o",
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
        ]
        path = tmp_path / "results.json"
        run.save(str(path))

        loaded = BenchmarkRun.load(str(path))
        assert loaded.model == "gpt-4o"
        assert len(loaded.results) == 1
        assert loaded.results[0].reference == "ref"

    def test_save_with_path_object(self, tmp_path) -> None:
        run = BenchmarkRun(model="m", suite_name="s", base_url="u")
        path = tmp_path / "pathobj.json"
        run.save(path)  # Path object, not str
        assert path.exists()
