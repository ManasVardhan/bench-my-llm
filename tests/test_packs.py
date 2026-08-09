"""Tests for suite packs: registry loading, install, list, remove, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.packs import (
    PackError,
    default_registry,
    find_installed,
    install_pack,
    list_installed,
    load_registry,
    packs_dir,
    remove_pack,
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
def registry_dir(tmp_path: Path) -> Path:
    (tmp_path / "support_basics.json").write_text(json.dumps(SUITE))
    (tmp_path / "registry.json").write_text(json.dumps({
        "packs": {
            "support-basics": {
                "version": "1.2.0",
                "description": "Customer support prompts",
                "url": "support_basics.json",
            },
            "empty-desc": {
                "url": str(tmp_path / "support_basics.json"),
            },
        }
    }))
    return tmp_path


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BENCH_MY_LLM_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


class TestRegistry:
    def test_load_registry(self, registry_dir: Path) -> None:
        packs = load_registry(str(registry_dir / "registry.json"))
        assert set(packs) == {"support-basics", "empty-desc"}
        info = packs["support-basics"]
        assert info.version == "1.2.0"
        assert info.url == "support_basics.json"

    def test_registry_defaults(self, registry_dir: Path) -> None:
        packs = load_registry(str(registry_dir / "registry.json"))
        assert packs["empty-desc"].version == "0.0.0"
        assert packs["empty-desc"].description == ""

    def test_missing_registry(self, tmp_path: Path) -> None:
        with pytest.raises(PackError, match="not found"):
            load_registry(str(tmp_path / "nope.json"))

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(PackError, match="Invalid JSON"):
            load_registry(str(bad))

    def test_missing_packs_key(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"suites": {}}))
        with pytest.raises(PackError, match="'packs' mapping"):
            load_registry(str(bad))

    def test_pack_without_url(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"packs": {"x": {"version": "1.0"}}}))
        with pytest.raises(PackError, match="requires a non-empty 'url'"):
            load_registry(str(bad))

    def test_default_registry_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BENCH_MY_LLM_REGISTRY", raising=False)
        assert default_registry() is None
        monkeypatch.setenv("BENCH_MY_LLM_REGISTRY", "https://example.com/reg.json")
        assert default_registry() == "https://example.com/reg.json"


class TestInstall:
    def test_install_and_find(self, registry_dir: Path, home: Path) -> None:
        pack = install_pack("support-basics", str(registry_dir / "registry.json"))
        assert pack.version == "1.2.0"
        assert pack.file.is_file()
        assert pack.file.parent == packs_dir()
        assert home in pack.file.parents

        found = find_installed("support-basics")
        assert found == pack.file
        data = json.loads(found.read_text())
        assert data["name"] == "support-basics"

    def test_install_unknown_pack(self, registry_dir: Path, home: Path) -> None:
        with pytest.raises(PackError, match="not found in registry"):
            install_pack("nope", str(registry_dir / "registry.json"))

    def test_install_invalid_suite(self, tmp_path: Path, home: Path) -> None:
        (tmp_path / "bad_suite.json").write_text(json.dumps({"prompts": []}))
        (tmp_path / "reg.json").write_text(json.dumps({
            "packs": {"bad": {"url": "bad_suite.json"}}
        }))
        with pytest.raises(PackError, match="invalid suite"):
            install_pack("bad", str(tmp_path / "reg.json"))
        assert find_installed("bad") is None

    def test_install_unsupported_extension(self, tmp_path: Path, home: Path) -> None:
        (tmp_path / "reg.json").write_text(json.dumps({
            "packs": {"bad": {"url": "suite.txt"}}
        }))
        with pytest.raises(PackError, match="unsupported file type"):
            install_pack("bad", str(tmp_path / "reg.json"))

    def test_reinstall_overwrites(self, registry_dir: Path, home: Path) -> None:
        source = str(registry_dir / "registry.json")
        install_pack("support-basics", source)
        pack = install_pack("support-basics", source)
        assert pack.file.is_file()
        assert len(list_installed()) == 1


class TestListRemove:
    def test_list_empty(self, home: Path) -> None:
        assert list_installed() == []

    def test_list_installed(self, registry_dir: Path, home: Path) -> None:
        install_pack("support-basics", str(registry_dir / "registry.json"))
        installed = list_installed()
        assert len(installed) == 1
        assert installed[0].name == "support-basics"
        assert installed[0].version == "1.2.0"

    def test_remove(self, registry_dir: Path, home: Path) -> None:
        pack = install_pack("support-basics", str(registry_dir / "registry.json"))
        remove_pack("support-basics")
        assert not pack.file.exists()
        assert list_installed() == []

    def test_remove_not_installed(self, home: Path) -> None:
        with pytest.raises(PackError, match="not installed"):
            remove_pack("ghost")


class TestHttpRegistry:
    def test_install_over_http(self, registry_dir: Path, home: Path) -> None:
        import http.server
        import threading

        handler = type(
            "Handler",
            (http.server.SimpleHTTPRequestHandler,),
            {
                "directory": str(registry_dir),
                "log_message": lambda self, *a: None,
            },
        )

        def make_handler(*args, **kwargs):
            return handler(*args, directory=str(registry_dir), **kwargs)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            url = f"http://127.0.0.1:{port}/registry.json"
            pack = install_pack("support-basics", url)
            assert pack.file.is_file()
            assert pack.source == url
        finally:
            server.shutdown()
            thread.join(timeout=5)


class TestSuiteCLI:
    def test_cli_install_list_remove(self, registry_dir: Path, home: Path) -> None:
        runner = CliRunner()
        reg = str(registry_dir / "registry.json")

        result = runner.invoke(cli, ["suite", "install", "support-basics", "-r", reg])
        assert result.exit_code == 0
        assert "Installed" in result.output

        result = runner.invoke(cli, ["suite", "list"])
        assert result.exit_code == 0
        assert "support-basics" in result.output

        result = runner.invoke(cli, ["suite", "list", "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["name"] == "support-basics"

        result = runner.invoke(cli, ["suite", "remove", "support-basics"])
        assert result.exit_code == 0
        assert "Removed" in result.output

        result = runner.invoke(cli, ["suite", "list"])
        assert "No suite packs installed" in result.output

    def test_cli_install_no_registry(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BENCH_MY_LLM_REGISTRY", raising=False)
        runner = CliRunner()
        result = runner.invoke(cli, ["suite", "install", "whatever"])
        assert result.exit_code == 1
        assert "No registry" in result.output

    def test_cli_install_uses_env_registry(
        self, registry_dir: Path, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "BENCH_MY_LLM_REGISTRY", str(registry_dir / "registry.json")
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["suite", "install", "support-basics"])
        assert result.exit_code == 0
        assert "Installed" in result.output

    def test_cli_install_unknown(self, registry_dir: Path, home: Path) -> None:
        runner = CliRunner()
        reg = str(registry_dir / "registry.json")
        result = runner.invoke(cli, ["suite", "install", "nope", "-r", reg])
        assert result.exit_code == 1
        assert "not found in registry" in result.output

    def test_cli_remove_missing(self, home: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["suite", "remove", "ghost"])
        assert result.exit_code == 1
        assert "not installed" in result.output


class TestResolveSuiteWithPacks:
    def test_run_resolves_installed_pack(
        self, registry_dir: Path, home: Path
    ) -> None:
        from bench_my_llm.cli import _resolve_suite

        install_pack("support-basics", str(registry_dir / "registry.json"))
        suite = _resolve_suite("support-basics")
        assert suite.name == "support-basics"
        assert len(suite.prompts) == 2

    def test_unknown_name_mentions_packs(self, home: Path) -> None:
        from bench_my_llm.cli import _resolve_suite

        with pytest.raises(SystemExit):
            _resolve_suite("no-such-suite")

    def test_builtin_still_wins(self, home: Path) -> None:
        from bench_my_llm.cli import _resolve_suite
        from bench_my_llm.prompts import SUITES

        name = next(iter(SUITES))
        suite = _resolve_suite(name)
        assert suite is not None


class TestExampleRegistry:
    def test_example_registry_installs(self, home: Path) -> None:
        examples = Path(__file__).resolve().parent.parent / "examples"
        pack = install_pack("example-basics", str(examples / "registry.json"))
        assert pack.file.is_file()
        assert pack.version == "1.0.0"
