"""Parse and validate authoritative Markdown task files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml


@dataclass(frozen=True)
class TaskRecord:
    path: Path
    task_id: str
    title: str
    parent: str | None
    status: str
    priority: str
    owner: str
    reviewer: str
    branch: str
    depends_on: tuple[str, ...]
    blocks: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    source_revision: str
    body: str


def _front_matter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML front matter")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"{path}: unterminated YAML front matter")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: YAML front matter must be an object")
    for key in ("created_at", "updated_at"):
        value = metadata.get(key)
        if isinstance(value, (date, datetime)):
            metadata[key] = value.isoformat()
    return metadata, parts[2].lstrip()


def _schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_task(path: Path, schema_path: Path) -> TaskRecord:
    metadata, body = _front_matter(path.read_text(), path)
    errors = sorted(
        Draft202012Validator(_schema(schema_path)).iter_errors(metadata),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"{path}: invalid task front matter: {details}")
    return TaskRecord(
        path=path,
        task_id=metadata["id"],
        title=metadata["title"],
        parent=metadata["parent"],
        status=metadata["status"],
        priority=metadata["priority"],
        owner=metadata["owner"],
        reviewer=metadata["reviewer"],
        branch=metadata["branch"],
        depends_on=tuple(metadata["depends_on"]),
        blocks=tuple(metadata["blocks"]),
        allowed_paths=tuple(metadata["allowed_paths"]),
        source_revision=metadata["source_revision"],
        body=body,
    )


def load_tasks(tasks_dir: Path, schema_path: Path) -> dict[str, TaskRecord]:
    tasks: dict[str, TaskRecord] = {}
    for path in sorted(tasks_dir.glob("T*.md")):
        task = load_task(path, schema_path)
        if task.task_id in tasks:
            raise ValueError(
                f"duplicate task id {task.task_id}: "
                f"{tasks[task.task_id].path} and {task.path}"
            )
        tasks[task.task_id] = task
    return tasks


def _cycle_errors(
    tasks: dict[str, TaskRecord],
    edges: dict[str, tuple[str, ...]],
    label: str,
) -> list[str]:
    errors: list[str] = []
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(task_id: str) -> None:
        current = state.get(task_id, 0)
        if current == 2:
            return
        if current == 1:
            start = stack.index(task_id)
            cycle = stack[start:] + [task_id]
            message = f"{label} cycle: {' -> '.join(cycle)}"
            if message not in errors:
                errors.append(message)
            return
        state[task_id] = 1
        stack.append(task_id)
        for target in edges.get(task_id, ()):
            if target in tasks:
                visit(target)
        stack.pop()
        state[task_id] = 2

    for task_id in sorted(tasks):
        visit(task_id)
    return errors


def validate_task_graph(tasks: dict[str, TaskRecord]) -> list[str]:
    errors: list[str] = []
    roots = sorted(task.task_id for task in tasks.values() if task.parent is None)
    if len(roots) != 1:
        errors.append(f"task graph must have exactly one root; found {roots}")

    for task_id, task in sorted(tasks.items()):
        if task.parent is not None and task.parent not in tasks:
            errors.append(f"{task_id}: missing parent {task.parent}")
        for dependency in task.depends_on:
            if dependency not in tasks:
                errors.append(f"{task_id}: missing dependency {dependency}")
        for blocked_task in task.blocks:
            if blocked_task not in tasks:
                errors.append(f"{task_id}: missing blocked task {blocked_task}")

        if task.status in {
            "ready",
            "running",
            "awaiting_review",
            "revision_needed",
            "blocked",
            "accepted",
        }:
            for field_name, value in (
                ("owner", task.owner),
                ("reviewer", task.reviewer),
                ("branch", task.branch),
                ("source_revision", task.source_revision),
            ):
                if not value:
                    errors.append(
                        f"{task_id}: status {task.status} requires {field_name}"
                    )

        if task.status in {"ready", "running", "awaiting_review", "accepted"}:
            for dependency in task.depends_on:
                if dependency in tasks and tasks[dependency].status != "accepted":
                    errors.append(
                        f"{task_id}: dependency {dependency} must be accepted "
                        f"before status {task.status}"
                    )

        if task.status == "accepted":
            signoff = f"Accepted by: {task.reviewer}"
            if signoff not in task.body:
                errors.append(
                    f"{task_id}: accepted task lacks local reviewer sign-off"
                )

    parent_edges = {
        task_id: (() if task.parent is None else (task.parent,))
        for task_id, task in tasks.items()
    }
    dependency_edges = {
        task_id: task.depends_on for task_id, task in tasks.items()
    }
    errors.extend(_cycle_errors(tasks, parent_edges, "parent"))
    errors.extend(_cycle_errors(tasks, dependency_edges, "dependency"))
    return sorted(set(errors))

