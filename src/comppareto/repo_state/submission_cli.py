"""Command-line submission gate for one task branch."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from .acceptance import load_acceptance_contract
from .submission import (
    check_forbidden_claims,
    check_metrics,
    check_required_files,
    revision_is_ancestor,
    validate_git_submission,
)
from .tasks import load_task


def _resolve_task(root: Path, task_id: str) -> Path:
    matches = sorted((root / "tasks").glob(f"{task_id}-*.md"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one task file for {task_id}; found "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _run_commands(root: Path, commands) -> list[str]:
    errors: list[str] = []
    for rule in commands:
        completed = subprocess.run(
            rule.command,
            cwd=root,
            shell=True,
            executable="/bin/bash",
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            errors.append(
                f"command failed [{rule.name}] exit={completed.returncode}: "
                f"{rule.command}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--skip-commands",
        action="store_true",
        help="validate static evidence only; never sufficient for awaiting_review",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        task = load_task(
            _resolve_task(root, args.task),
            root / "schemas" / "task.schema.json",
        )
        contract = load_acceptance_contract(
            root / "tasks" / "contracts" / f"{args.task}.acceptance.yaml",
            root / "schemas" / "task-acceptance.schema.json",
        )
    except (OSError, ValueError) as error:
        print(f"submission_validation=fail task={args.task}")
        print(error)
        return 1

    errors: list[str] = []
    if contract.task_id != task.task_id:
        errors.append(
            f"contract task_id {contract.task_id} does not match task {task.task_id}"
        )
    if task.status != "awaiting_review":
        errors.append(
            f"task status must be awaiting_review for submission; found {task.status}"
        )
    if not revision_is_ancestor(root, task.source_revision):
        errors.append(
            f"task source_revision {task.source_revision} is not an ancestor of HEAD"
        )
    errors.extend(
        validate_git_submission(
            root=root,
            expected_branch=task.branch,
            base_ref=contract.base_ref,
            allowed_paths=task.allowed_paths,
        )
    )
    errors.extend(check_required_files(root, contract.required_files))
    errors.extend(check_metrics(root, contract.metrics))
    errors.extend(check_forbidden_claims(root, contract.forbidden_claims))
    if not args.skip_commands:
        errors.extend(_run_commands(root, contract.commands))

    if errors:
        print(f"submission_validation=fail task={args.task}")
        for error in errors:
            print(error)
        return 1
    if args.skip_commands:
        print(f"submission_validation=static-pass task={args.task}")
    else:
        print(f"submission_validation=pass task={args.task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
