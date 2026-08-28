"""Validate a task branch before it may be submitted for local review."""

from __future__ import annotations

from fnmatch import fnmatch
import json
import operator
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .acceptance import ForbiddenClaim, MetricRule


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def path_is_allowed(path: str, allowed_paths: Iterable[str]) -> bool:
    for rule in allowed_paths:
        if any(char in rule for char in "*?["):
            if fnmatch(path, rule) or fnmatch(path, f"{rule.rstrip('/')}/*"):
                return True
        elif rule.endswith("/"):
            if path.startswith(rule):
                return True
        elif path == rule:
            return True
    return False


def changed_paths(root: Path, base_ref: str) -> list[str]:
    result = _git(root, "diff", "--name-only", f"{base_ref}...HEAD")
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"cannot diff against {base_ref}")
    return [line for line in result.stdout.splitlines() if line]


def validate_git_submission(
    *,
    root: Path,
    expected_branch: str,
    base_ref: str,
    allowed_paths: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    status = _git(root, "status", "--porcelain")
    if status.returncode != 0 or status.stdout.strip():
        errors.append("working tree is not clean")
    branch = _git(root, "branch", "--show-current").stdout.strip()
    if branch != expected_branch:
        errors.append(
            f"branch mismatch: expected {expected_branch}, found {branch or 'detached HEAD'}"
        )
    ancestor = _git(root, "merge-base", "--is-ancestor", base_ref, "HEAD")
    if ancestor.returncode != 0:
        errors.append(f"base ref {base_ref} is not an ancestor of HEAD")
    try:
        files = changed_paths(root, base_ref)
    except ValueError as error:
        errors.append(str(error))
        files = []
    for path in files:
        if not path_is_allowed(path, allowed_paths):
            errors.append(f"unauthorized changed path: {path}")
    return errors


def revision_is_ancestor(root: Path, revision: str) -> bool:
    return _git(root, "merge-base", "--is-ancestor", revision, "HEAD").returncode == 0


def check_required_files(root: Path, patterns: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    for pattern in patterns:
        if not list(root.glob(pattern)):
            errors.append(f"missing required file pattern: {pattern}")
    return errors


def _json_value(root: Path, file: str, path: str) -> Any:
    value: Any = json.loads((root / file).read_text())
    for segment in path.split("."):
        if isinstance(value, list):
            value = value[int(segment)]
        else:
            value = value[segment]
    return value


_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def check_metrics(root: Path, rules: tuple[MetricRule, ...]) -> list[str]:
    errors: list[str] = []
    for rule in rules:
        try:
            actual = _json_value(root, rule.file, rule.path)
        except (OSError, ValueError, KeyError, IndexError) as error:
            errors.append(f"{rule.file}:{rule.path}: cannot read metric: {error}")
            continue
        if not _OPERATORS[rule.operator](actual, rule.value):
            errors.append(
                f"{rule.file}:{rule.path}: expected {rule.operator} "
                f"{rule.value!r}, found {actual!r}"
            )
    return errors


def check_forbidden_claims(
    root: Path,
    rules: tuple[ForbiddenClaim, ...],
) -> list[str]:
    errors: list[str] = []
    for rule in rules:
        try:
            trigger_value = _json_value(
                root,
                rule.trigger.file,
                rule.trigger.path,
            )
            text = (root / rule.file).read_text().lower()
        except (OSError, ValueError, KeyError, IndexError) as error:
            errors.append(f"{rule.file}: cannot evaluate forbidden claim: {error}")
            continue
        if trigger_value != rule.trigger.equals:
            continue
        for phrase in rule.phrases:
            if phrase.lower() in text:
                errors.append(
                    f"{rule.file}: forbidden phrase {phrase!r} when "
                    f"{rule.trigger.file}:{rule.trigger.path} == "
                    f"{rule.trigger.equals!r}"
                )
    return errors
