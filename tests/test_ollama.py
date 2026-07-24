"""Tests for Ollama auto-detection.

Covers the detection module (HTTP handling, payload parsing, model name
resolution) and the CLI integration (ollama command, run --ollama,
compare --ollama) with a mocked Ollama server.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.ollama import (
    DEFAULT_OLLAMA_URL,
    OllamaModel,
    OllamaNotRunningError,
    detect_ollama_models,
    is_ollama_running,
    openai_base_url,
    resolve_ollama_model,
)

TAGS_PAYLOAD = {
    "models": [
        {
            "name": "llama3:8b",
            "size": 4_700_000_000,
            "modified_at": "2026-07-01T10:00:00Z",
            "details": {
                "parameter_size": "8B",
                "quantization_level": "Q4_0",
                "family": "llama",
            },
        },
        {
            "name": "phi3:latest",
            "size": 2_300_000_000,
            "modified_at": "2026-07-02T10:00:00Z",
            "details": {
                "parameter_size": "3.8B",
                "quantization_level": "Q4_K_M",
                "family": "phi3",
            },
        },
    ]
}


def _mock_response(payload: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = payload
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _urlopen_returning(payload: bytes):
    return patch(
        "bench_my_llm.ollama.urllib.request.urlopen",
        return_value=_mock_response(payload),
    )


@pytest.fixture
def runner():
    return CliRunner()


# -----------------------------------------------------------------------
# detect_ollama_models
# -----------------------------------------------------------------------

class TestDetectOllamaModels:
    def test_detects_and_sorts_models(self):
        with _urlopen_returning(json.dumps(TAGS_PAYLOAD).encode()) as mock_open:
            models = detect_ollama_models()
        assert [m.name for m in models] == ["llama3:8b", "phi3:latest"]
        assert mock_open.call_args[0][0] == f"{DEFAULT_OLLAMA_URL}/api/tags"

    def test_model_details_parsed(self):
        with _urlopen_returning(json.dumps(TAGS_PAYLOAD).encode()):
            models = detect_ollama_models()
        llama = models[0]
        assert llama.parameter_size == "8B"
        assert llama.quantization == "Q4_0"
        assert llama.family == "llama"
        assert llama.size_bytes == 4_700_000_000
        assert llama.modified_at == "2026-07-01T10:00:00Z"

    def test_custom_url_with_trailing_slash(self):
        with _urlopen_returning(json.dumps({"models": []}).encode()) as mock_open:
            detect_ollama_models("http://gpu-box:11434/")
        assert mock_open.call_args[0][0] == "http://gpu-box:11434/api/tags"

    def test_connection_error_raises(self):
        with patch(
            "bench_my_llm.ollama.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(OllamaNotRunningError, match="Could not reach Ollama"):
                detect_ollama_models()

    def test_timeout_raises(self):
        with patch(
            "bench_my_llm.ollama.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            with pytest.raises(OllamaNotRunningError):
                detect_ollama_models()

    def test_invalid_json_raises(self):
        with _urlopen_returning(b"<html>not json</html>"):
            with pytest.raises(OllamaNotRunningError, match="invalid JSON"):
                detect_ollama_models()

    def test_missing_models_key_raises(self):
        with _urlopen_returning(json.dumps({"status": "ok"}).encode()):
            with pytest.raises(OllamaNotRunningError, match="missing 'models'"):
                detect_ollama_models()

    def test_malformed_entries_skipped(self):
        payload = {
            "models": [
                "not-a-dict",
                {"size": 123},
                {"name": ""},
                {"name": "good:latest", "size": "not-an-int", "details": "nope"},
            ]
        }
        with _urlopen_returning(json.dumps(payload).encode()):
            models = detect_ollama_models()
        assert [m.name for m in models] == ["good:latest"]
        assert models[0].size_bytes == 0
        assert models[0].parameter_size == ""

    def test_model_key_fallback(self):
        payload = {"models": [{"model": "alt-key:latest"}]}
        with _urlopen_returning(json.dumps(payload).encode()):
            models = detect_ollama_models()
        assert models[0].name == "alt-key:latest"

    def test_timeout_passed_to_urlopen(self):
        with _urlopen_returning(json.dumps({"models": []}).encode()) as mock_open:
            detect_ollama_models(timeout=0.5)
        assert mock_open.call_args[1]["timeout"] == 0.5


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

class TestHelpers:
    def test_openai_base_url_default(self):
        assert openai_base_url() == "http://localhost:11434/v1"

    def test_openai_base_url_strips_slash(self):
        assert openai_base_url("http://gpu:11434/") == "http://gpu:11434/v1"

    def test_is_running_true(self):
        with _urlopen_returning(json.dumps({"models": []}).encode()):
            assert is_ollama_running() is True

    def test_is_running_false(self):
        with patch(
            "bench_my_llm.ollama.urllib.request.urlopen",
            side_effect=urllib.error.URLError("nope"),
        ):
            assert is_ollama_running() is False

    def test_size_display_gb(self):
        assert OllamaModel("m", size_bytes=4_700_000_000).size_display == "4.7 GB"

    def test_size_display_mb(self):
        assert OllamaModel("m", size_bytes=500_000_000).size_display == "500 MB"

    def test_size_display_unknown(self):
        assert OllamaModel("m").size_display == "?"

    def test_to_dict_round_trip(self):
        model = OllamaModel("llama3:8b", size_bytes=4_700_000_000,
                            parameter_size="8B", quantization="Q4_0",
                            family="llama", modified_at="2026-07-01")
        data = model.to_dict()
        assert data["name"] == "llama3:8b"
        assert data["size"] == "4.7 GB"
        assert data["parameter_size"] == "8B"


# -----------------------------------------------------------------------
# resolve_ollama_model
# -----------------------------------------------------------------------

class TestResolveModel:
    AVAILABLE = [
        OllamaModel("llama3:8b"),
        OllamaModel("llama3:latest"),
        OllamaModel("phi3:mini"),
    ]

    def test_exact_match(self):
        assert resolve_ollama_model("llama3:8b", self.AVAILABLE) == "llama3:8b"

    def test_tagless_prefers_latest(self):
        assert resolve_ollama_model("llama3", self.AVAILABLE) == "llama3:latest"

    def test_tagless_falls_back_to_first_tag(self):
        assert resolve_ollama_model("phi3", self.AVAILABLE) == "phi3:mini"

    def test_no_match_returns_none(self):
        assert resolve_ollama_model("mistral", self.AVAILABLE) is None

    def test_wrong_tag_returns_none(self):
        assert resolve_ollama_model("phi3:huge", self.AVAILABLE) is None


# -----------------------------------------------------------------------
# ollama CLI command
# -----------------------------------------------------------------------

class TestOllamaCommand:
    def test_table_output(self, runner):
        models = [
            OllamaModel("llama3:8b", size_bytes=4_700_000_000,
                        parameter_size="8B", quantization="Q4_0", family="llama"),
        ]
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=models):
            result = runner.invoke(cli, ["ollama"])
        assert result.exit_code == 0
        assert "llama3:8b" in result.output
        assert "8B" in result.output
        assert "4.7 GB" in result.output
        assert "run --ollama -m llama3:8b" in result.output

    def test_json_output_pipes_cleanly(self, runner):
        models = [OllamaModel("phi3:mini", size_bytes=2_300_000_000)]
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=models):
            result = runner.invoke(cli, ["ollama", "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["name"] == "phi3:mini"

    def test_not_running_exits_1(self, runner):
        with patch(
            "bench_my_llm.cli.detect_ollama_models",
            side_effect=OllamaNotRunningError("Could not reach Ollama at X"),
        ):
            result = runner.invoke(cli, ["ollama"])
        assert result.exit_code == 1
        assert "Could not reach Ollama" in result.output
        assert "ollama serve" in result.output

    def test_no_models_message(self, runner):
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=[]):
            result = runner.invoke(cli, ["ollama"])
        assert result.exit_code == 0
        assert "no models installed" in result.output
        assert "ollama pull" in result.output

    def test_custom_url_forwarded(self, runner):
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=[]) as mock_detect:
            runner.invoke(cli, ["ollama", "--url", "http://gpu:11434"])
        assert mock_detect.call_args[0][0] == "http://gpu:11434"


# -----------------------------------------------------------------------
# run --ollama and compare --ollama
# -----------------------------------------------------------------------

def _fake_run(model, suite, base_url=None, api_key=None, temperature=0.0,
              on_progress=None):
    from bench_my_llm.runner import BenchmarkResult, BenchmarkRun

    run = BenchmarkRun(model=model, suite_name=suite.name,
                       base_url=base_url or "", timestamp="2026-07-23T00:00:00Z")
    run.results.append(BenchmarkResult(
        model=model, prompt_text="p", category="reasoning", response_text="r",
        ttft_ms=10.0, total_latency_ms=100.0, tokens_generated=5,
        tokens_per_second=50.0, prompt_tokens=3, completion_tokens=5,
    ))
    return run


class TestRunWithOllama:
    def test_run_ollama_sets_connection_defaults(self, runner):
        models = [OllamaModel("llama3:latest")]
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=models), \
             patch("bench_my_llm.cli.run_benchmark", side_effect=_fake_run) as mock_run:
            result = runner.invoke(cli, ["run", "--ollama", "-m", "llama3"])
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args[1]
        assert kwargs["model"] == "llama3:latest"
        assert kwargs["base_url"] == "http://localhost:11434/v1"
        assert kwargs["api_key"] == "ollama"

    def test_run_ollama_explicit_url_and_key_kept(self, runner):
        models = [OllamaModel("llama3:latest")]
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=models), \
             patch("bench_my_llm.cli.run_benchmark", side_effect=_fake_run) as mock_run:
            result = runner.invoke(cli, [
                "run", "--ollama", "-m", "llama3",
                "-u", "http://gpu:11434/v1", "-k", "secret",
            ])
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args[1]
        assert kwargs["base_url"] == "http://gpu:11434/v1"
        assert kwargs["api_key"] == "secret"

    def test_run_ollama_unknown_model_lists_available(self, runner):
        models = [OllamaModel("llama3:latest"), OllamaModel("phi3:mini")]
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=models):
            result = runner.invoke(cli, ["run", "--ollama", "-m", "mistral"])
        assert result.exit_code == 1
        assert "not found" in result.output
        assert "llama3:latest" in result.output
        assert "phi3:mini" in result.output

    def test_run_ollama_server_down_exits_1(self, runner):
        with patch(
            "bench_my_llm.cli.detect_ollama_models",
            side_effect=OllamaNotRunningError("Could not reach Ollama at X"),
        ):
            result = runner.invoke(cli, ["run", "--ollama", "-m", "llama3"])
        assert result.exit_code == 1
        assert "Could not reach Ollama" in result.output

    def test_run_ollama_no_models_exits_1(self, runner):
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=[]):
            result = runner.invoke(cli, ["run", "--ollama", "-m", "llama3"])
        assert result.exit_code == 1
        assert "no models installed" in result.output

    def test_run_without_ollama_flag_skips_detection(self, runner):
        with patch("bench_my_llm.cli.detect_ollama_models") as mock_detect, \
             patch("bench_my_llm.cli.run_benchmark", side_effect=_fake_run):
            result = runner.invoke(cli, ["run", "-m", "gpt-4o"])
        assert result.exit_code == 0, result.output
        mock_detect.assert_not_called()

    def test_compare_ollama_resolves_all_models(self, runner):
        models = [OllamaModel("llama3:latest"), OllamaModel("phi3:mini")]
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=models), \
             patch("bench_my_llm.cli.run_benchmark", side_effect=_fake_run) as mock_run:
            result = runner.invoke(cli, ["compare", "--ollama", "llama3", "phi3"])
        assert result.exit_code == 0, result.output
        called_models = [c[1]["model"] for c in mock_run.call_args_list]
        assert called_models == ["llama3:latest", "phi3:mini"]
        assert all(c[1]["base_url"] == "http://localhost:11434/v1"
                   for c in mock_run.call_args_list)

    def test_compare_ollama_unknown_model_exits_1(self, runner):
        models = [OllamaModel("llama3:latest")]
        with patch("bench_my_llm.cli.detect_ollama_models", return_value=models):
            result = runner.invoke(cli, ["compare", "--ollama", "llama3", "mistral"])
        assert result.exit_code == 1
        assert "not found" in result.output
