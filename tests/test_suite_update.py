"""Tests for suite pack updates: update_pack, update_all, and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.packs import (
    MANIFEST_NAME,
    PackError,
    install_pack,
    list_installed,
    packs_dir,
    update_all,
    update_pack,
)

SUITE_V1 = {
    "name": "support-basics",
    "description": "Support prompts",
    "prompts": [
        {"text": "How do I reset my password?", "category": "support"},
    ],
}

SUITE_V2 = {
    "name": "support-basics",
    "description": "Support prompts, expanded",
    "prompts": [
        {"text": "How do I reset my password?", "category": "support"},
        {"text": "Summarize this ticket.", "category": "support"},
    ],
}


def write_registry(directory: Path, version: str, suite: dict) -> Path:
    (directory / "support_basics.json").write_text(json.dumps(suite))
    registry = directory / "registry.json"
    registry.write_text(json.dumps({
        "packs": {
            "support-basics": {
                "version": version,
                "description": "Customer support prompts",
                "url": "support_basics.json",
            }
        }
    }))
    return registry


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BENCH_MY_LLM_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    reg_dir = tmp_path / "reg"
    reg_dir.mkdir()
    return write_registry(reg_dir, "1.0.0", SUITE_V1)


class TestUpdatePack:
    def test_up_to_date(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        result = update_pack("support-basics")
        assert result.status == "up-to-date"
        assert result.old_version == "1.0.0"
        assert result.new_version == "1.0.0"

    def test_updated_on_version_change(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        write_registry(registry.parent, "1.1.0", SUITE_V2)

        result = update_pack("support-basics")
        assert result.status == "updated"
        assert result.old_version == "1.0.0"
        assert result.new_version == "1.1.0"

        installed = list_installed()
        assert installed[0].version == "1.1.0"
        data = json.loads(installed[0].file.read_text())
        assert len(data["prompts"]) == 2

    def test_force_reinstalls_same_version(self, registry: Path, home: Path) -> None:
        pack = install_pack("support-basics", str(registry))
        pack.file.write_text(json.dumps({"name": "tampered", "prompts": []}))

        result = update_pack("support-basics", force=True)
        assert result.status == "reinstalled"
        assert result.new_version == "1.0.0"
        data = json.loads(pack.file.read_text())
        assert data["name"] == "support-basics"

    def test_not_installed(self, home: Path) -> None:
        with pytest.raises(PackError, match="not installed"):
            update_pack("nope")

    def test_registry_override(self, registry: Path, tmp_path: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        other = write_registry(other_dir, "2.0.0", SUITE_V2)

        result = update_pack("support-basics", registry_source=str(other))
        assert result.status == "updated"
        assert result.new_version == "2.0.0"
        # New source is recorded so future updates use the override.
        assert list_installed()[0].source == str(other)

    def test_registry_override_recorded_when_up_to_date(
        self, registry: Path, tmp_path: Path, home: Path
    ) -> None:
        install_pack("support-basics", str(registry))
        moved_dir = tmp_path / "moved"
        moved_dir.mkdir()
        moved = write_registry(moved_dir, "1.0.0", SUITE_V1)

        result = update_pack("support-basics", registry_source=str(moved))
        assert result.status == "up-to-date"
        # The moved registry becomes the recorded source for next time.
        assert list_installed()[0].source == str(moved)
        assert update_pack("support-basics").status == "up-to-date"

    def test_no_recorded_source(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        manifest_path = packs_dir() / MANIFEST_NAME
        data = json.loads(manifest_path.read_text())
        data["packs"]["support-basics"]["source"] = ""
        manifest_path.write_text(json.dumps(data))

        with pytest.raises(PackError, match="no recorded registry source"):
            update_pack("support-basics")

    def test_pack_removed_from_registry(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        registry.write_text(json.dumps({"packs": {}}))
        with pytest.raises(PackError, match="no longer present"):
            update_pack("support-basics")

    def test_registry_gone(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        registry.unlink()
        with pytest.raises(PackError, match="not found"):
            update_pack("support-basics")


class TestUpdateAll:
    def test_mixed_results(self, tmp_path: Path, home: Path) -> None:
        reg_a_dir = tmp_path / "a"
        reg_a_dir.mkdir()
        reg_a = write_registry(reg_a_dir, "1.0.0", SUITE_V1)

        reg_b_dir = tmp_path / "b"
        reg_b_dir.mkdir()
        (reg_b_dir / "other.json").write_text(json.dumps({
            "name": "other",
            "prompts": [{"text": "Hi there."}],
        }))
        reg_b = reg_b_dir / "registry.json"
        reg_b.write_text(json.dumps({
            "packs": {"other": {"version": "0.1.0", "url": "other.json"}}
        }))

        install_pack("support-basics", str(reg_a))
        install_pack("other", str(reg_b))

        # Bump support-basics, break other's registry.
        write_registry(reg_a_dir, "1.1.0", SUITE_V2)
        reg_b.unlink()

        results = {r.name: r for r in update_all()}
        assert results["support-basics"].status == "updated"
        assert results["other"].status == "error"
        assert "not found" in results["other"].message

    def test_empty(self, home: Path) -> None:
        assert update_all() == []


class TestUpdateCli:
    def test_update_named_pack(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        write_registry(registry.parent, "1.1.0", SUITE_V2)

        result = CliRunner().invoke(cli, ["suite", "update", "support-basics"])
        assert result.exit_code == 0
        assert "Updated" in result.output
        assert "1.0.0" in result.output and "1.1.0" in result.output

    def test_update_all_default(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        result = CliRunner().invoke(cli, ["suite", "update"])
        assert result.exit_code == 0
        assert "up to date" in result.output
        assert "1 pack(s) checked, 0 updated, 0 failed" in result.output

    def test_no_packs_installed(self, home: Path) -> None:
        result = CliRunner().invoke(cli, ["suite", "update"])
        assert result.exit_code == 0
        assert "No suite packs installed" in result.output

    def test_unknown_pack_fails(self, home: Path) -> None:
        result = CliRunner().invoke(cli, ["suite", "update", "nope"])
        assert result.exit_code == 1
        assert "Failed" in result.output
        assert "not installed" in result.output

    def test_force_flag(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        result = CliRunner().invoke(cli, ["suite", "update", "--force"])
        assert result.exit_code == 0
        assert "Reinstalled" in result.output

    def test_registry_override(self, registry: Path, tmp_path: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        other = write_registry(other_dir, "3.0.0", SUITE_V2)

        result = CliRunner().invoke(
            cli, ["suite", "update", "support-basics", "-r", str(other)]
        )
        assert result.exit_code == 0
        assert "3.0.0" in result.output

    def test_json_output(self, registry: Path, home: Path) -> None:
        install_pack("support-basics", str(registry))
        write_registry(registry.parent, "1.1.0", SUITE_V2)

        result = CliRunner().invoke(
            cli, ["suite", "update", "--json-output"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == [{
            "name": "support-basics",
            "status": "updated",
            "old_version": "1.0.0",
            "new_version": "1.1.0",
            "message": "",
        }]

    def test_json_output_error_exit_code(self, home: Path) -> None:
        result = CliRunner().invoke(
            cli, ["suite", "update", "nope", "--json-output"]
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data[0]["status"] == "error"
