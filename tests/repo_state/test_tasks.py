from __future__ import annotations

from pathlib import Path

import pytest

from comppareto.repo_state.tasks import load_task, load_tasks, validate_task_graph


SCHEMA = Path("schemas/task.schema.json")
SOURCE_REVISION = "a" * 40


def write_task(
    directory: Path,
    task_id: str,
    *,
    parent: str | None,
    status: str = "planned",
    depends_on: tuple[str, ...] = (),
    branch: str | None = None,
    body: str = "## Review history\n\n- Created for test.\n",
) -> Path:
    parent_yaml = "null" if parent is None else parent
    dependency_yaml = ", ".join(depends_on)
    task_branch = branch or f"agent/{task_id.lower()}"
    path = directory / f"{task_id}-task.md"
    path.write_text(
        f"""---
id: {task_id}
title: Test task {task_id}
parent: {parent_yaml}
status: {status}
priority: P0
owner: test-owner
reviewer: local-research-agent
branch: {task_branch}
depends_on: [{dependency_yaml}]
blocks: []
allowed_paths: [tests/]
source_revision: "{SOURCE_REVISION}"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# {task_id}

{body}
"""
    )
    return path


def test_loads_task_front_matter_and_body(tmp_path: Path) -> None:
    path = write_task(tmp_path, "T000", parent=None)

    task = load_task(path, SCHEMA)

    assert task.task_id == "T000"
    assert task.parent is None
    assert task.branch == "agent/t000"
    assert "## Review history" in task.body


def test_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    write_task(tmp_path, "T000", parent=None)
    duplicate = write_task(tmp_path, "T001", parent="T000")
    duplicate.write_text(duplicate.read_text().replace("id: T001", "id: T000"))

    with pytest.raises(ValueError, match="duplicate task id T000"):
        load_tasks(tmp_path, SCHEMA)


def test_rejects_missing_parent(tmp_path: Path) -> None:
    write_task(tmp_path, "T000", parent=None)
    write_task(tmp_path, "T001", parent="T999")

    errors = validate_task_graph(load_tasks(tmp_path, SCHEMA))

    assert "T001: missing parent T999" in errors


def test_rejects_dependency_cycle(tmp_path: Path) -> None:
    write_task(tmp_path, "T000", parent=None)
    write_task(tmp_path, "T001", parent="T000", depends_on=("T002",))
    write_task(tmp_path, "T002", parent="T000", depends_on=("T001",))

    errors = validate_task_graph(load_tasks(tmp_path, SCHEMA))

    assert any("dependency cycle" in error for error in errors)


def test_rejects_ready_task_with_unaccepted_dependency(tmp_path: Path) -> None:
    write_task(tmp_path, "T000", parent=None)
    write_task(tmp_path, "T001", parent="T000", status="running")
    write_task(
        tmp_path,
        "T002",
        parent="T000",
        status="ready",
        depends_on=("T001",),
    )

    errors = validate_task_graph(load_tasks(tmp_path, SCHEMA))

    assert "T002: dependency T001 must be accepted before status ready" in errors


def test_rejects_terminal_acceptance_without_local_review(tmp_path: Path) -> None:
    write_task(tmp_path, "T000", parent=None)
    write_task(
        tmp_path,
        "T001",
        parent="T000",
        status="accepted",
        body="## Review history\n\n- Remote executor says complete.\n",
    )

    errors = validate_task_graph(load_tasks(tmp_path, SCHEMA))

    assert "T001: accepted task lacks local reviewer sign-off" in errors


def test_accepts_initial_repository_task_tree() -> None:
    tasks = load_tasks(Path("tasks"), SCHEMA)

    assert len(tasks) == 28
    assert validate_task_graph(tasks) == []
