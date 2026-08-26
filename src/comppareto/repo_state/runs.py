"""Validate formal and legacy run provenance manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


@dataclass(frozen=True)
class RunValidationResult:
    path: Path
    errors: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return value


def _legacy_t1a_errors(path: Path, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    run_id = path.parent.name
    revision = payload.get("git", {}).get("revision")
    dirty = payload.get("git", {}).get("dirty")
    config_hash = payload.get("config_sha256")
    if payload.get("status") not in {"pass", "fail"}:
        errors.append(f"{run_id}: legacy status must be pass or fail")
    if dirty is not False:
        errors.append(f"{run_id}: formal run must have dirty=false")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        errors.append(f"{run_id}: legacy source revision must be 40 hex characters")
    if (
        not isinstance(config_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", config_hash) is None
    ):
        errors.append(f"{run_id}: legacy config hash must be 64 hex characters")
    if not isinstance(payload.get("environment"), dict) or not payload["environment"]:
        errors.append(f"{run_id}: legacy environment must be non-empty")
    if not isinstance(payload.get("checks"), dict) or not payload["checks"]:
        errors.append(f"{run_id}: legacy checks must be non-empty")
    return errors


def _validation_messages(error: ValidationError) -> list[str]:
    if not error.context:
        return [error.message]
    messages: list[str] = []
    for nested in error.context:
        messages.extend(_validation_messages(nested))
    return messages


def validate_run_manifest(
    path: Path,
    schema_path: Path,
    repository_root: Path,
) -> RunValidationResult:
    payload = _load_json(path)
    if (
        path.name == "t1_manifest.json"
        and "schema_version" not in payload
        and "git" in payload
    ):
        return RunValidationResult(
            path=path,
            errors=tuple(sorted(_legacy_t1a_errors(path, payload))),
        )

    schema = _load_json(schema_path)
    validation_errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    errors = [
        message
        for error in validation_errors
        for message in _validation_messages(error)
    ]
    run_id = str(payload.get("run_id") or path.parent.name)

    if payload.get("run_kind") == "formal" and payload.get("dirty") is not False:
        errors.append(f"{run_id}: formal run must have dirty=false")

    result_files = payload.get("result_files", [])
    if isinstance(result_files, list):
        for result_file in result_files:
            if isinstance(result_file, str) and not (repository_root / result_file).exists():
                errors.append(f"{run_id}: missing result file {result_file}")

    return RunValidationResult(path=path, errors=tuple(sorted(set(errors))))


def validate_run_tree(
    runs_dir: Path,
    schema_path: Path,
    repository_root: Path,
) -> list[str]:
    errors: list[str] = []
    for path in sorted(runs_dir.rglob("*manifest.json")):
        result = validate_run_manifest(path, schema_path, repository_root)
        errors.extend(f"{path}: {error}" for error in result.errors)
    return sorted(errors)
