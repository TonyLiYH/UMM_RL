"""Load machine-readable task submission acceptance contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml


@dataclass(frozen=True)
class CommandRule:
    name: str
    command: str


@dataclass(frozen=True)
class MetricRule:
    file: str
    path: str
    operator: str
    value: Any


@dataclass(frozen=True)
class Trigger:
    file: str
    path: str
    equals: Any


@dataclass(frozen=True)
class ForbiddenClaim:
    file: str
    trigger: Trigger
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceContract:
    task_id: str
    base_ref: str
    required_files: tuple[str, ...]
    commands: tuple[CommandRule, ...]
    metrics: tuple[MetricRule, ...]
    forbidden_claims: tuple[ForbiddenClaim, ...]


def load_acceptance_contract(
    path: Path,
    schema_path: Path,
) -> AcceptanceContract:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: acceptance contract must be an object")
    schema = json.loads(schema_path.read_text())
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"{path}: invalid acceptance contract: {details}")

    return AcceptanceContract(
        task_id=payload["task_id"],
        base_ref=payload["base_ref"],
        required_files=tuple(payload["required_files"]),
        commands=tuple(
            CommandRule(name=item["name"], command=item["command"])
            for item in payload["commands"]
        ),
        metrics=tuple(
            MetricRule(
                file=item["file"],
                path=item["path"],
                operator=item["op"],
                value=item["value"],
            )
            for item in payload["metrics"]
        ),
        forbidden_claims=tuple(
            ForbiddenClaim(
                file=item["file"],
                trigger=Trigger(
                    file=item["when"]["file"],
                    path=item["when"]["path"],
                    equals=item["when"]["equals"],
                ),
                phrases=tuple(item["phrases"]),
            )
            for item in payload["forbidden_claims"]
        ),
    )
