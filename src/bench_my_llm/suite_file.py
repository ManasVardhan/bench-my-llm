"""Load custom prompt suites from JSON or YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .prompts import Prompt, PromptSuite

SUITE_FILE_EXTENSIONS = (".json", ".yaml", ".yml")

_ALLOWED_SUITE_KEYS = {"name", "description", "prompts"}
_ALLOWED_PROMPT_KEYS = {"text", "category", "reference", "max_tokens"}


class SuiteFileError(ValueError):
    """Raised when a custom suite file is missing, malformed, or invalid."""


def looks_like_suite_file(value: str) -> bool:
    """Return True if the value looks like a path to a suite file.

    A value counts as a file reference when it has a recognized suite
    file extension or contains a path separator.
    """
    lowered = value.lower()
    if lowered.endswith(SUITE_FILE_EXTENSIONS):
        return True
    return "/" in value or "\\" in value


def load_suite_file(path: str | Path) -> PromptSuite:
    """Load and validate a custom prompt suite from a JSON or YAML file.

    The file must contain a mapping with a non-empty "prompts" list.
    Each prompt requires "text" and may set "category", "reference",
    and "max_tokens". "name" and "description" are optional at the top
    level, with the name defaulting to the file stem.

    Raises SuiteFileError with a friendly message on any problem.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise SuiteFileError(f"Suite file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUITE_FILE_EXTENSIONS:
        raise SuiteFileError(
            f"Unsupported suite file extension '{suffix}' for {file_path}. "
            f"Use one of: {', '.join(SUITE_FILE_EXTENSIONS)}"
        )

    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SuiteFileError(f"Could not read suite file {file_path}: {exc}") from exc

    data = _parse(text, suffix, file_path)
    return _build_suite(data, file_path)


def _parse(text: str, suffix: str, file_path: Path) -> Any:
    if suffix == ".json":
        import json

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise SuiteFileError(f"Invalid JSON in {file_path}: {exc}") from exc

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SuiteFileError(
            "YAML suite files require the pyyaml package. "
            "Install it with: pip install 'bench-my-llm[yaml]'"
        ) from exc

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SuiteFileError(f"Invalid YAML in {file_path}: {exc}") from exc


def _build_suite(data: Any, file_path: Path) -> PromptSuite:
    if not isinstance(data, dict):
        raise SuiteFileError(
            f"Suite file {file_path} must contain a mapping at the top level, "
            f"got {type(data).__name__}."
        )

    unknown = set(data) - _ALLOWED_SUITE_KEYS
    if unknown:
        raise SuiteFileError(
            f"Unknown top-level key(s) in {file_path}: {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_SUITE_KEYS))}"
        )

    name = data.get("name", file_path.stem)
    if not isinstance(name, str) or not name.strip():
        raise SuiteFileError(f"'name' in {file_path} must be a non-empty string.")

    description = data.get("description", "Custom prompt suite")
    if not isinstance(description, str):
        raise SuiteFileError(f"'description' in {file_path} must be a string.")

    raw_prompts = data.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise SuiteFileError(
            f"Suite file {file_path} must contain a non-empty 'prompts' list."
        )

    prompts: list[Prompt] = []
    for i, item in enumerate(raw_prompts, 1):
        prompts.append(_build_prompt(item, i, file_path))

    return PromptSuite(name=name.strip(), description=description, prompts=prompts)


def _build_prompt(item: Any, index: int, file_path: Path) -> Prompt:
    where = f"prompt #{index} in {file_path}"

    if not isinstance(item, dict):
        raise SuiteFileError(f"{where} must be a mapping, got {type(item).__name__}.")

    unknown = set(item) - _ALLOWED_PROMPT_KEYS
    if unknown:
        raise SuiteFileError(
            f"Unknown key(s) in {where}: {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_PROMPT_KEYS))}"
        )

    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SuiteFileError(f"{where} requires a non-empty 'text' string.")

    category = item.get("category", "custom")
    if not isinstance(category, str) or not category.strip():
        raise SuiteFileError(f"'category' for {where} must be a non-empty string.")

    reference = item.get("reference", "")
    if not isinstance(reference, str):
        raise SuiteFileError(f"'reference' for {where} must be a string.")

    max_tokens = item.get("max_tokens", 512)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise SuiteFileError(f"'max_tokens' for {where} must be a positive integer.")

    return Prompt(
        text=text,
        category=category.strip(),
        reference=reference,
        max_tokens=max_tokens,
    )
