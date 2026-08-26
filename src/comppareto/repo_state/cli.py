"""Command-line validation for task and run evidence state."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runs import validate_run_tree
from .tasks import load_tasks, validate_task_graph


def validate_repository(
    root: Path,
) -> tuple[list[str], list[str], int, int]:
    task_schema = root / "schemas" / "task.schema.json"
    run_schema = root / "schemas" / "run-manifest.schema.json"
    tasks = load_tasks(root / "tasks", task_schema)
    task_errors = validate_task_graph(tasks)
    manifest_paths = sorted((root / "runs").rglob("*manifest.json"))
    run_errors = validate_run_tree(root / "runs", run_schema, root)
    return task_errors, run_errors, len(tasks), len(manifest_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        task_errors, run_errors, task_count, manifest_count = validate_repository(root)
    except (OSError, ValueError) as error:
        print(f"research_state=fail")
        print(str(error))
        return 1

    if task_errors:
        print(f"task_tree=fail tasks={task_count}")
        for error in task_errors:
            print(error)
    else:
        print(f"task_tree=pass tasks={task_count}")

    if run_errors:
        print(f"run_manifests=fail manifests={manifest_count}")
        for error in run_errors:
            print(error)
    else:
        print(f"run_manifests=pass manifests={manifest_count}")

    if task_errors or run_errors:
        print("research_state=fail")
        return 1
    print("research_state=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
