"""Tests for suite pack publishing: entry building, registry writes, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.packs import (
    PackError,
    build_registry_entry,
    install_pack,
    load_registry,
    publish_pack,
)

SUITE = {
    "name": "support-basics",
    "description": "Support prompts",
    "prompts": [
        {"text": "How do I reset my password?", "category": "support"},
        {"text": "Summarize this ticket.", "category": "support", "max_tokens": 128},
    ],
}


@pytest.fixture()
def suite_file(tmp_path: Path) -> Path:
    path = tmp_path / "support_basics.json"
    path.write_text(json.dumps(SUITE))
    return path


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BENCH_MY_LLM_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


class TestBuildRegistryEntry:
    def test_defaults_from_suite(self, suite_file: Path) -> None:
        name, entry = build_registry_entry(suite_file)
        assert name == "support_basics"
        assert entry["version"] == "1.0.0"
        assert entry["description"] == "Support prompts"
        assert entry["url"] == str(suite_file.resolve())

    def test_explicit_name_version_description(self, suite_file: Path) -> None:
        name, entry = build_registry_entry(
            suite_file, name="support", version="2.1.0", description="Custom"
        )
        assert name == "support"
        assert entry["version"] == "2.1.0"
        assert entry["description"] == "Custom"

    def test_relative_url_when_next_to_registry(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        _, entry = build_registry_entry(suite_file, registry_path=registry)
        assert entry["url"] == "support_basics.json"

    def test_absolute_url_when_outside_registry_dir(
        self, suite_file: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "elsewhere" / "registry.json"
        other.parent.mkdir()
        _, entry = build_registry_entry(suite_file, registry_path=other)
        assert Path(entry["url"]).is_absolute()

    def test_invalid_suite_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"prompts": []}))
        with pytest.raises(PackError, match="not a valid prompt suite"):
            build_registry_entry(bad)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "suite.txt"
        bad.write_text("hello")
        with pytest.raises(PackError, match="Unsupported suite file type"):
            build_registry_entry(bad)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PackError, match="not found"):
            build_registry_entry(tmp_path / "nope.json")

    def test_bad_names_rejected(self, suite_file: Path) -> None:
        for bad in ("", "  ", "a/b", "a\\b", "..") :
            with pytest.raises(PackError):
                build_registry_entry(suite_file, name=bad)

    def test_empty_version_rejected(self, suite_file: Path) -> None:
        with pytest.raises(PackError, match="Version"):
            build_registry_entry(suite_file, version="  ")


class TestPublishPack:
    def test_creates_registry(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        pack = publish_pack(suite_file, registry, name="support")
        assert registry.is_file()
        packs = load_registry(str(registry))
        assert packs["support"].version == "1.0.0"
        assert packs["support"].url == "support_basics.json"
        assert pack.name == "support"

    def test_preserves_existing_entries(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        registry.write_text(json.dumps({
            "packs": {"other": {"version": "0.5.0", "url": "other.json"}}
        }))
        publish_pack(suite_file, registry, name="support")
        data = json.loads(registry.read_text())
        assert set(data["packs"]) == {"other", "support"}
        assert data["packs"]["other"]["version"] == "0.5.0"

    def test_duplicate_requires_force(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        publish_pack(suite_file, registry, name="support")
        with pytest.raises(PackError, match="already exists"):
            publish_pack(suite_file, registry, name="support")
        pack = publish_pack(
            suite_file, registry, name="support", version="2.0.0", force=True
        )
        assert pack.version == "2.0.0"
        assert load_registry(str(registry))["support"].version == "2.0.0"

    def test_http_registry_rejected(self, suite_file: Path) -> None:
        with pytest.raises(PackError, match="HTTP registry"):
            publish_pack(suite_file, "https://example.com/registry.json")

    def test_malformed_registry_rejected(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        registry.write_text("{not json")
        with pytest.raises(PackError, match="Invalid JSON"):
            publish_pack(suite_file, registry)

    def test_registry_without_packs_key_rejected(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        registry.write_text(json.dumps({"suites": {}}))
        with pytest.raises(PackError, match="'packs' mapping"):
            publish_pack(suite_file, registry)

    def test_published_pack_installs(self, suite_file: Path, home: Path) -> None:
        registry = suite_file.parent / "registry.json"
        publish_pack(suite_file, registry, name="support", description="Desc")
        pack = install_pack("support", str(registry))
        assert pack.file.is_file()
        assert pack.description == "Desc"


class TestPublishCli:
    def test_publish_to_registry(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["suite", "publish", str(suite_file), "-r", str(registry),
             "-n", "support"],
        )
        assert result.exit_code == 0, result.output
        assert "Published" in result.output
        assert registry.is_file()

    def test_publish_prints_entry_without_registry(self, suite_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["suite", "publish", str(suite_file)])
        assert result.exit_code == 0, result.output
        assert "support_basics" in result.output
        assert "1.0.0" in result.output

    def test_publish_duplicate_fails_without_force(self, suite_file: Path) -> None:
        registry = suite_file.parent / "registry.json"
        runner = CliRunner()
        args = ["suite", "publish", str(suite_file), "-r", str(registry)]
        assert runner.invoke(cli, args).exit_code == 0
        result = runner.invoke(cli, args)
        assert result.exit_code == 1
        assert "already exists" in result.output
        forced = runner.invoke(cli, args + ["--force", "-v", "1.1.0"])
        assert forced.exit_code == 0, forced.output

    def test_publish_invalid_suite_fails(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"prompts": []}))
        runner = CliRunner()
        result = runner.invoke(cli, ["suite", "publish", str(bad)])
        assert result.exit_code == 1
        assert "not a valid prompt suite" in result.output
