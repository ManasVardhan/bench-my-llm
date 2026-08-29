"""Benchmark runner: sends prompts to an OpenAI-compatible API and measures latency."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

from openai import APIConnectionError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionChunk

from .prompts import Prompt, PromptSuite


@dataclass
class BenchmarkResult:
    """Result from a single prompt benchmark run."""

    model: str
    prompt_text: str
    category: str
    response_text: str
    ttft_ms: float  # time to first token in milliseconds
    total_latency_ms: float  # total request time in milliseconds
    tokens_generated: int
    tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int
    reference: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunMetadata:
    """Free-form notes and captured environment metadata for a run.

    Attached to saved results so historical comparisons stay interpretable:
    you can tell later which endpoint, temperature, and tool version produced
    a given result, and read any note left at run time.
    """

    notes: str = ""
    endpoint: str = ""
    temperature: float = 0.0
    suite_version: str = ""
    tool_version: str = ""
    captured_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def is_empty(self) -> bool:
        """True when no metadata field carries information worth showing."""
        return not any(
            [
                self.notes,
                self.endpoint,
                self.suite_version,
                self.tool_version,
                self.captured_at,
            ]
        )


@dataclass
class BenchmarkRun:
    """Collection of results from a full benchmark run."""

    model: str
    suite_name: str
    base_url: str
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""
    metadata: RunMetadata | None = None

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        out: dict[str, Any] = {
            "model": self.model,
            "suite": self.suite_name,
            "base_url": self.base_url,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }
        if self.metadata is not None:
            out["metadata"] = self.metadata.to_dict()
        Path(path).write_text(json.dumps(out, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkRun:
        """Load results from a JSON file."""
        data = json.loads(Path(path).read_text())
        meta_raw = data.get("metadata")
        metadata = None
        if isinstance(meta_raw, dict):
            known = {f.name for f in fields(RunMetadata)}
            metadata = RunMetadata(**{k: v for k, v in meta_raw.items() if k in known})
        run = cls(
            model=data["model"],
            suite_name=data["suite"],
            base_url=data.get("base_url", ""),
            timestamp=data.get("timestamp", ""),
            metadata=metadata,
        )
        run.results = [BenchmarkResult(**r) for r in data["results"]]
        return run


def _count_tokens_approx(text: str) -> int:
    """Rough token count (words * 1.3). Good enough for cost estimation."""
    return max(1, int(len(text.split()) * 1.3))


def run_single_prompt(
    client: OpenAI,
    model: str,
    prompt: Prompt,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Benchmark a single prompt with streaming to measure TTFT."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.text}]

    start = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=prompt.max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        chunk = cast(ChatCompletionChunk, chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(chunk.choices[0].delta.content)

    end = time.perf_counter()

    response_text = "".join(chunks)
    total_ms = (end - start) * 1000
    ttft_ms = ((first_token_time - start) * 1000) if first_token_time else total_ms

    completion_tokens = _count_tokens_approx(response_text)
    prompt_tokens = _count_tokens_approx(prompt.text)

    generation_time = end - (first_token_time or start)
    tps = completion_tokens / generation_time if generation_time > 0 else 0.0

    return BenchmarkResult(
        model=model,
        prompt_text=prompt.text,
        category=prompt.category,
        response_text=response_text,
        ttft_ms=round(ttft_ms, 1),
        total_latency_ms=round(total_ms, 1),
        tokens_generated=completion_tokens,
        tokens_per_second=round(tps, 1),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reference=prompt.reference,
    )


def run_benchmark(
    model: str,
    suite: PromptSuite,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    on_progress: Callable[[int, int, object], None] | None = None,
    notes: str = "",
) -> BenchmarkRun:
    """Run a full benchmark suite against a model.

    Args:
        model: Model name (e.g. 'gpt-4o').
        suite: Prompt suite to run.
        base_url: Optional API base URL (defaults to OpenAI).
        api_key: Optional API key (defaults to OPENAI_API_KEY env var).
        temperature: Sampling temperature.
        on_progress: Optional callback(index, total, result) for progress updates.
        notes: Optional free-form note stored in the run metadata.

    Returns:
        BenchmarkRun with all results.
    """
    import datetime

    from . import __version__

    client_kwargs: dict = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key

    client = OpenAI(**client_kwargs)

    endpoint = base_url or "https://api.openai.com/v1"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    run = BenchmarkRun(
        model=model,
        suite_name=suite.name,
        base_url=endpoint,
        timestamp=now,
        metadata=RunMetadata(
            notes=notes,
            endpoint=endpoint,
            temperature=temperature,
            suite_version=f"{suite.name} ({len(suite.prompts)} prompts)",
            tool_version=__version__,
            captured_at=now,
        ),
    )

    for i, prompt in enumerate(suite.prompts):
        try:
            result = run_single_prompt(client, model, prompt, temperature)
        except AuthenticationError:
            raise SystemExit(
                "❌ Authentication failed. Please check your API key.\n"
                "Set it via --api-key or the OPENAI_API_KEY environment variable."
            )
        except RateLimitError:
            raise SystemExit(
                "❌ Rate limit exceeded. Wait a moment and try again, or use a "
                "different API key with higher limits."
            )
        except APITimeoutError:
            raise SystemExit(
                "❌ API request timed out. The server may be overloaded. "
                "Try again in a moment."
            )
        except APIConnectionError:
            raise SystemExit(
                f"❌ Could not connect to API at {base_url or 'https://api.openai.com/v1'}.\n"
                "Check your network connection and that the base URL is correct."
            )
        run.results.append(result)
        if on_progress:
            on_progress(i + 1, len(suite.prompts), result)

    return run
