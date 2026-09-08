"""Tests for custom prompt suite file loading."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.runner import BenchmarkResult, BenchmarkRun
from bench_my_llm.suite_file import (
    SuiteFileError,
    load_suite_file,
    looks_like_suite_file,
)

VALID_SUITE = {
    "name": "my-suite",
    "description": "Test suite",
    "prompts": [
        {"text": "What is 2+2?", "category": "math", "reference": "4", "max_tokens": 64},
        {"text": "Name a color."},
    ],
}


def write_json(tmp_path: Path, data, name: str = "suite.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


class TestLooksLikeSuiteFile:
    def test_json_extension(self):
        assert looks_like_suite_file("my_suite.json")

    def test_yaml_extensions(self):
        assert looks_like_suite_file("suite.yaml")
        assert looks_like_suite_file("suite.yml")

    def test_path_separator(self):
        assert looks_like_suite_file("./suites/custom")
        assert looks_like_suite_file("suites\\custom")

    def test_builtin_names_are_not_files(self):
        assert not looks_like_suite_file("reasoning")
        assert not looks_like_suite_file("all")

    def test_case_insensitive_extension(self):
        assert looks_like_suite_file("SUITE.JSON")


class TestLoadSuiteFileJSON:
    def test_valid_suite(self, tmp_path):
        path = write_json(tmp_path, VALID_SUITE)
        suite = load_suite_file(path)
        assert suite.name == "my-suite"
        assert suite.description == "Test suite"
        assert len(suite.prompts) == 2
        assert suite.prompts[0].text == "What is 2+2?"
        assert suite.prompts[0].category == "math"
        assert suite.prompts[0].reference == "4"
        assert suite.prompts[0].max_tokens == 64

    def test_prompt_defaults(self, tmp_path):
        path = write_json(tmp_path, VALID_SUITE)
        suite = load_suite_file(path)
        p = suite.prompts[1]
        assert p.category == "custom"
        assert p.reference == ""
        assert p.max_tokens == 512

    def test_name_defaults_to_file_stem(self, tmp_path):
        data = {"prompts": [{"text": "hi"}]}
        path = write_json(tmp_path, data, name="domain_qa.json")
        suite = load_suite_file(path)
        assert suite.name == "domain_qa"

    def test_missing_file(self, tmp_path):
        with pytest.raises(SuiteFileError, match="not found"):
            load_suite_file(tmp_path / "nope.json")

    def test_bad_extension(self, tmp_path):
        path = tmp_path / "suite.txt"
        path.write_text("{}")
        with pytest.raises(SuiteFileError, match="extension"):
            load_suite_file(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(SuiteFileError, match="Invalid JSON"):
            load_suite_file(path)

    def test_top_level_not_mapping(self, tmp_path):
        path = write_json(tmp_path, ["just", "a", "list"])
        with pytest.raises(SuiteFileError, match="mapping"):
            load_suite_file(path)

    def test_unknown_top_level_key(self, tmp_path):
        data = {"prompts": [{"text": "hi"}], "promts": []}
        path = write_json(tmp_path, data)
        with pytest.raises(SuiteFileError, match="promts"):
            load_suite_file(path)

    def test_empty_prompts(self, tmp_path):
        path = write_json(tmp_path, {"prompts": []})
        with pytest.raises(SuiteFileError, match="non-empty 'prompts'"):
            load_suite_file(path)

    def test_missing_prompts(self, tmp_path):
        path = write_json(tmp_path, {"name": "x"})
        with pytest.raises(SuiteFileError, match="non-empty 'prompts'"):
            load_suite_file(path)

    def test_prompt_not_mapping(self, tmp_path):
        path = write_json(tmp_path, {"prompts": ["just text"]})
        with pytest.raises(SuiteFileError, match="prompt #1"):
            load_suite_file(path)

    def test_prompt_missing_text(self, tmp_path):
        path = write_json(tmp_path, {"prompts": [{"category": "x"}]})
        with pytest.raises(SuiteFileError, match="'text'"):
            load_suite_file(path)

    def test_prompt_empty_text(self, tmp_path):
        path = write_json(tmp_path, {"prompts": [{"text": "   "}]})
        with pytest.raises(SuiteFileError, match="'text'"):
            load_suite_file(path)

    def test_prompt_unknown_key(self, tmp_path):
        path = write_json(tmp_path, {"prompts": [{"text": "hi", "maxtokens": 5}]})
        with pytest.raises(SuiteFileError, match="maxtokens"):
            load_suite_file(path)

    def test_bad_max_tokens(self, tmp_path):
        for bad in (0, -5, "many", True):
            path = write_json(tmp_path, {"prompts": [{"text": "hi", "max_tokens": bad}]})
            with pytest.raises(SuiteFileError, match="max_tokens"):
                load_suite_file(path)

    def test_bad_reference_type(self, tmp_path):
        path = write_json(tmp_path, {"prompts": [{"text": "hi", "reference": 42}]})
        with pytest.raises(SuiteFileError, match="reference"):
            load_suite_file(path)

    def test_bad_name_type(self, tmp_path):
        path = write_json(tmp_path, {"name": 7, "prompts": [{"text": "hi"}]})
        with pytest.raises(SuiteFileError, match="'name'"):
            load_suite_file(path)

    def test_example_file_loads(self):
        example = Path(__file__).parent.parent / "examples" / "custom_suite.json"
        suite = load_suite_file(example)
        assert suite.name == "support-bot"
        assert len(suite.prompts) == 3


class TestLoadSuiteFileYAML:
    def test_valid_yaml(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        path = tmp_path / "suite.yaml"
        path.write_text(yaml.safe_dump(VALID_SUITE))
        suite = load_suite_file(path)
        assert suite.name == "my-suite"
        assert len(suite.prompts) == 2

    def test_yml_extension(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        path = tmp_path / "suite.yml"
        path.write_text(yaml.safe_dump({"prompts": [{"text": "hi"}]}))
        suite = load_suite_file(path)
        assert suite.name == "suite"

    def test_invalid_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "bad.yaml"
        path.write_text("prompts:\n  - text: [unclosed")
        with pytest.raises(SuiteFileError, match="Invalid YAML"):
            load_suite_file(path)

    def test_missing_pyyaml_message(self, tmp_path):
        path = tmp_path / "suite.yaml"
        path.write_text("prompts:\n  - text: hi\n")
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import), \
                pytest.raises(SuiteFileError, match="pyyaml"):
            load_suite_file(path)


class TestCLICustomSuite:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_with_custom_suite_file(self, mock_run, runner, tmp_path):
        mock_run.return_value = _make_dummy_run()
        path = write_json(tmp_path, VALID_SUITE)
        result = runner.invoke(cli, ["run", "-m", "test-model", "--suite", path])
        assert result.exit_code == 0
        suite = mock_run.call_args.kwargs["suite"]
        assert suite.name == "my-suite"
        assert len(suite.prompts) == 2

    @patch("bench_my_llm.cli.run_benchmark")
    def test_run_with_builtin_suite_still_works(self, mock_run, runner):
        mock_run.return_value = _make_dummy_run()
        result = runner.invoke(cli, ["run", "-m", "test-model", "--suite", "reasoning"])
        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["suite"].name == "reasoning"

    def test_run_with_missing_suite_file(self, runner):
        result = runner.invoke(cli, ["run", "-m", "test-model", "--suite", "missing.json"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_run_with_invalid_suite_file(self, runner, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{broken")
        result = runner.invoke(cli, ["run", "-m", "test-model", "--suite", str(path)])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_run_with_unknown_suite_name(self, runner):
        result = runner.invoke(cli, ["run", "-m", "test-model", "--suite", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown suite" in result.output

    @patch("bench_my_llm.cli.run_benchmark")
    def test_compare_with_custom_suite_file(self, mock_run, runner, tmp_path):
        mock_run.side_effect = [
            _make_dummy_run(model="a"),
            _make_dummy_run(model="b"),
        ]
        path = write_json(tmp_path, VALID_SUITE)
        result = runner.invoke(cli, ["compare", "a", "b", "--suite", path])
        assert result.exit_code == 0
        assert mock_run.call_count == 2

    def test_suites_command_mentions_custom_files(self, runner):
        result = runner.invoke(cli, ["suites"])
        assert result.exit_code == 0
        assert "Custom suites" in result.output


def _make_dummy_run(model: str = "test-model") -> BenchmarkRun:
    results = [
        BenchmarkResult(
            model=model,
            prompt_text=f"prompt {i}",
            category="custom",
            response_text=f"response {i} with some words here",
            ttft_ms=100.0 + i * 50,
            total_latency_ms=500.0 + i * 100,
            tokens_generated=50 + i * 10,
            tokens_per_second=30.0 + i * 5,
            prompt_tokens=20,
            completion_tokens=50 + i * 10,
            reference="response with some reference words",
        )
        for i in range(2)
    ]
    return BenchmarkRun(
        model=model,
        suite_name="my-suite",
        base_url="https://api.openai.com/v1",
        results=results,
        timestamp="2025-01-01T00:00:00Z",
    )
