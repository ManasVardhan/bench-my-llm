"""Suite packs: install shareable prompt suites from a registry.

A registry is a JSON document (local file or HTTP URL) that maps pack
names to suite files:

    {
      "packs": {
        "support-basics": {
          "version": "1.0.0",
          "description": "Customer support prompts",
          "url": "support_basics.json"
        }
      }
    }

Pack URLs may be absolute (http, https, or a filesystem path) or
relative, in which case they are resolved against the registry
location. Installed packs live in the packs directory (by default
~/.bench-my-llm/suites, override with BENCH_MY_LLM_HOME) and can be
used by name with --suite on run and compare.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .suite_file import SUITE_FILE_EXTENSIONS, SuiteFileError, parse_suite_text

DEFAULT_REGISTRY_ENV = "BENCH_MY_LLM_REGISTRY"
HOME_ENV = "BENCH_MY_LLM_HOME"
MANIFEST_NAME = "manifest.json"
_FETCH_TIMEOUT = 15.0


class PackError(ValueError):
    """Raised when a registry or pack is missing, malformed, or invalid."""


@dataclass
class PackInfo:
    """One pack entry from a registry."""

    name: str
    version: str
    description: str
    url: str


@dataclass
class InstalledPack:
    """One installed pack from the local manifest."""

    name: str
    version: str
    description: str
    source: str
    file: Path


def packs_dir() -> Path:
    """Return the directory where installed suite packs are stored."""
    home = os.environ.get(HOME_ENV)
    base = Path(home).expanduser() if home else Path.home() / ".bench-my-llm"
    return base / "suites"


def default_registry() -> str | None:
    """Return the default registry source from the environment, if set."""
    value = os.environ.get(DEFAULT_REGISTRY_ENV, "").strip()
    return value or None


def _is_http(source: str) -> bool:
    return urllib.parse.urlparse(source).scheme in ("http", "https")


def _fetch(source: str, what: str) -> str:
    """Fetch text content from an HTTP(S) URL or a local file path."""
    if _is_http(source):
        try:
            with urllib.request.urlopen(source, timeout=_FETCH_TIMEOUT) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError, UnicodeDecodeError) as exc:
            raise PackError(f"Could not fetch {what} from {source}: {exc}") from exc
    path = Path(source).expanduser()
    if not path.is_file():
        raise PackError(f"{what.capitalize()} not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PackError(f"Could not read {what} from {path}: {exc}") from exc


def _resolve_url(url: str, registry_source: str) -> str:
    """Resolve a possibly relative pack URL against the registry location."""
    if _is_http(url):
        return url
    if _is_http(registry_source):
        return urllib.parse.urljoin(registry_source, url)
    path = Path(url).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(registry_source).expanduser().parent / path))


def load_registry(source: str) -> dict[str, PackInfo]:
    """Load and validate a pack registry from a URL or file path.

    Returns a mapping of pack name to PackInfo, sorted by name.
    Raises PackError on any problem.
    """
    text = _fetch(source, "registry")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PackError(f"Invalid JSON in registry {source}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("packs"), dict):
        raise PackError(
            f"Registry {source} must be a JSON object with a 'packs' mapping."
        )

    packs: dict[str, PackInfo] = {}
    for name in sorted(data["packs"]):
        entry = data["packs"][name]
        if not isinstance(name, str) or not name.strip():
            raise PackError(f"Registry {source} contains a pack with an empty name.")
        if not isinstance(entry, dict):
            raise PackError(
                f"Pack '{name}' in registry {source} must be a mapping."
            )
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            raise PackError(
                f"Pack '{name}' in registry {source} requires a non-empty 'url'."
            )
        version = entry.get("version", "0.0.0")
        if not isinstance(version, str) or not version.strip():
            raise PackError(
                f"'version' for pack '{name}' in registry {source} must be a string."
            )
        description = entry.get("description", "")
        if not isinstance(description, str):
            raise PackError(
                f"'description' for pack '{name}' in registry {source} must be a string."
            )
        packs[name] = PackInfo(
            name=name, version=version, description=description, url=url.strip()
        )
    return packs


def _load_manifest(directory: Path) -> dict[str, dict[str, str]]:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    packs = data.get("packs") if isinstance(data, dict) else None
    return packs if isinstance(packs, dict) else {}


def _save_manifest(directory: Path, packs: dict[str, dict[str, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({"packs": packs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def install_pack(
    name: str, registry_source: str, directory: Path | None = None
) -> InstalledPack:
    """Install a pack by name from a registry into the packs directory.

    The suite content is fetched, validated as a prompt suite, and
    written to the packs directory. Reinstalling overwrites the
    existing pack. Raises PackError on any problem.
    """
    directory = directory if directory is not None else packs_dir()
    registry = load_registry(registry_source)

    if name not in registry:
        available = ", ".join(registry) or "none"
        raise PackError(
            f"Pack '{name}' not found in registry {registry_source}. "
            f"Available: {available}"
        )

    info = registry[name]
    resolved = _resolve_url(info.url, registry_source)
    suffix = Path(urllib.parse.urlparse(resolved).path).suffix.lower() or ".json"
    if suffix not in SUITE_FILE_EXTENSIONS:
        raise PackError(
            f"Pack '{name}' points to unsupported file type '{suffix}'. "
            f"Use one of: {', '.join(SUITE_FILE_EXTENSIONS)}"
        )

    content = _fetch(resolved, f"suite for pack '{name}'")
    try:
        parse_suite_text(content, suffix, resolved)
    except SuiteFileError as exc:
        raise PackError(f"Pack '{name}' contains an invalid suite: {exc}") from exc

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}{suffix}"
    target.write_text(content, encoding="utf-8")

    manifest = _load_manifest(directory)
    manifest[name] = {
        "version": info.version,
        "description": info.description,
        "source": registry_source,
        "file": target.name,
    }
    _save_manifest(directory, manifest)

    return InstalledPack(
        name=name,
        version=info.version,
        description=info.description,
        source=registry_source,
        file=target,
    )


def list_installed(directory: Path | None = None) -> list[InstalledPack]:
    """Return installed packs recorded in the manifest, sorted by name."""
    directory = directory if directory is not None else packs_dir()
    manifest = _load_manifest(directory)
    installed = []
    for name in sorted(manifest):
        entry = manifest[name]
        file = directory / entry.get("file", "")
        if not file.is_file():
            continue
        installed.append(
            InstalledPack(
                name=name,
                version=entry.get("version", "0.0.0"),
                description=entry.get("description", ""),
                source=entry.get("source", ""),
                file=file,
            )
        )
    return installed


def remove_pack(name: str, directory: Path | None = None) -> None:
    """Remove an installed pack and its manifest entry.

    Raises PackError if the pack is not installed.
    """
    directory = directory if directory is not None else packs_dir()
    manifest = _load_manifest(directory)
    if name not in manifest:
        raise PackError(f"Pack '{name}' is not installed.")
    file = directory / manifest[name].get("file", "")
    if file.is_file():
        file.unlink()
    del manifest[name]
    _save_manifest(directory, manifest)


def _validate_pack_name(name: str) -> str:
    """Validate a pack name for publishing and return it stripped."""
    name = name.strip()
    if not name:
        raise PackError("Pack name must not be empty.")
    if any(sep in name for sep in ("/", "\\")) or name in (".", ".."):
        raise PackError(
            f"Pack name '{name}' must not contain path separators."
        )
    return name


def build_registry_entry(
    suite_path: str | Path,
    name: str | None = None,
    version: str = "1.0.0",
    description: str | None = None,
    registry_path: str | Path | None = None,
) -> tuple[str, dict[str, str]]:
    """Validate a suite file and build its registry entry.

    Returns (pack name, entry dict) where the entry has version,
    description, and url keys ready to drop into a registry's packs
    mapping. The name defaults to the suite file stem and the
    description defaults to the suite's own description. When a
    registry path is given and the suite file lives under the
    registry's directory, the url is stored relative to the registry
    so the pair stays portable. Raises PackError on any problem.
    """
    path = Path(suite_path).expanduser()
    suffix = path.suffix.lower()
    if suffix not in SUITE_FILE_EXTENSIONS:
        raise PackError(
            f"Unsupported suite file type '{suffix or path.name}'. "
            f"Use one of: {', '.join(SUITE_FILE_EXTENSIONS)}"
        )
    content = _fetch(str(path), "suite")
    try:
        suite = parse_suite_text(content, suffix, path)
    except SuiteFileError as exc:
        raise PackError(f"Suite file is not a valid prompt suite: {exc}") from exc

    pack_name = _validate_pack_name(name if name is not None else path.stem)
    version = version.strip()
    if not version:
        raise PackError("Version must not be empty.")

    url = str(path.resolve())
    if registry_path is not None and not _is_http(str(registry_path)):
        registry_dir = Path(registry_path).expanduser().resolve().parent
        try:
            url = path.resolve().relative_to(registry_dir).as_posix()
        except ValueError:
            pass

    entry = {
        "version": version,
        "description": (
            description if description is not None else suite.description or ""
        ),
        "url": url,
    }
    return pack_name, entry


def publish_pack(
    suite_path: str | Path,
    registry_path: str | Path,
    name: str | None = None,
    version: str = "1.0.0",
    description: str | None = None,
    force: bool = False,
) -> PackInfo:
    """Validate a suite and add or update its entry in a local registry file.

    Creates the registry file if it does not exist. Existing entries
    for other packs are preserved. Re-publishing an existing pack name
    requires force=True. Raises PackError on any problem.
    """
    if _is_http(str(registry_path)):
        raise PackError(
            "Cannot publish to an HTTP registry. Publish to a local registry "
            "file and host it wherever your team fetches it from."
        )

    registry_file = Path(registry_path).expanduser()
    pack_name, entry = build_registry_entry(
        suite_path, name=name, version=version,
        description=description, registry_path=registry_file,
    )

    if registry_file.is_file():
        text = registry_file.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PackError(
                f"Invalid JSON in registry {registry_file}: {exc}"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("packs"), dict):
            raise PackError(
                f"Registry {registry_file} must be a JSON object with a "
                "'packs' mapping."
            )
    else:
        data = {"packs": {}}

    if pack_name in data["packs"] and not force:
        existing = data["packs"][pack_name]
        existing_version = (
            existing.get("version", "?") if isinstance(existing, dict) else "?"
        )
        raise PackError(
            f"Pack '{pack_name}' already exists in {registry_file} "
            f"(v{existing_version}). Use --force to overwrite it."
        )

    data["packs"][pack_name] = entry
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return PackInfo(
        name=pack_name,
        version=entry["version"],
        description=entry["description"],
        url=entry["url"],
    )


def find_installed(name: str, directory: Path | None = None) -> Path | None:
    """Return the suite file path for an installed pack name, if any."""
    directory = directory if directory is not None else packs_dir()
    manifest = _load_manifest(directory)
    entry = manifest.get(name)
    if not entry:
        return None
    file = directory / entry.get("file", "")
    return file if file.is_file() else None
