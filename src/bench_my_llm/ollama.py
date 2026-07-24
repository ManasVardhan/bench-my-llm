"""Ollama auto-detection: find locally running Ollama models.

Queries the local Ollama HTTP API (default http://localhost:11434) with
urllib only, no extra dependencies. Detected models can be benchmarked
through Ollama's OpenAI-compatible endpoint at <base_url>/v1.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaNotRunningError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


@dataclass
class OllamaModel:
    """A locally installed Ollama model."""

    name: str
    size_bytes: int = 0
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""
    modified_at: str = ""

    @property
    def size_display(self) -> str:
        """Human readable model size, e.g. '4.7 GB'."""
        if self.size_bytes <= 0:
            return "?"
        gb = self.size_bytes / 1_000_000_000
        if gb >= 1:
            return f"{gb:.1f} GB"
        return f"{self.size_bytes / 1_000_000:.0f} MB"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size": self.size_display,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "family": self.family,
            "modified_at": self.modified_at,
        }


def openai_base_url(base_url: str = DEFAULT_OLLAMA_URL) -> str:
    """Return the OpenAI-compatible endpoint for an Ollama base URL."""
    return base_url.rstrip("/") + "/v1"


def detect_ollama_models(
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 2.0,
) -> list[OllamaModel]:
    """Detect locally installed Ollama models via the /api/tags endpoint.

    Args:
        base_url: Ollama server URL (default http://localhost:11434).
        timeout: Connection timeout in seconds.

    Returns:
        List of OllamaModel, sorted by name.

    Raises:
        OllamaNotRunningError: If the server is unreachable or replies
            with something that is not valid Ollama tag data.
    """
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise OllamaNotRunningError(
            f"Could not reach Ollama at {base_url}: {exc}"
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OllamaNotRunningError(
            f"Ollama at {base_url} returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise OllamaNotRunningError(
            f"Unexpected response from Ollama at {base_url}: missing 'models' list"
        )

    models: list[OllamaModel] = []
    for entry in data["models"]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name:
            continue
        raw_details = entry.get("details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        size = entry.get("size")
        models.append(
            OllamaModel(
                name=name,
                size_bytes=size if isinstance(size, int) else 0,
                parameter_size=str(details.get("parameter_size", "") or ""),
                quantization=str(details.get("quantization_level", "") or ""),
                family=str(details.get("family", "") or ""),
                modified_at=str(entry.get("modified_at", "") or ""),
            )
        )

    models.sort(key=lambda m: m.name)
    return models


def is_ollama_running(
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 2.0,
) -> bool:
    """Return True if a reachable Ollama server responds with model data."""
    try:
        detect_ollama_models(base_url, timeout=timeout)
        return True
    except OllamaNotRunningError:
        return False


def resolve_ollama_model(
    model: str,
    available: list[OllamaModel],
) -> str | None:
    """Match a requested model name against detected Ollama models.

    Accepts exact names ('llama3:8b') or tagless names ('llama3', which
    matches 'llama3:latest' or the first 'llama3:*' entry).

    Returns:
        The resolved full model name, or None if nothing matches.
    """
    names = [m.name for m in available]
    if model in names:
        return model
    if ":" not in model:
        if f"{model}:latest" in names:
            return f"{model}:latest"
        for name in names:
            if name.split(":", 1)[0] == model:
                return name
    return None
