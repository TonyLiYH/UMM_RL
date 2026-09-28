from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys

from comppareto.repo_state.submission_cli import main as _submission_main


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def test_submission_cli_accepts_clean_task_branch(tmp_path: Path) -> None:
    _run("git", "init", "-b", "main", cwd=tmp_path)
    _run("git", "config", "user.email", "test@example.com", cwd=tmp_path)
    _run("git", "config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "tasks" / "contracts").mkdir(parents=True)
    (tmp_path / "schemas").mkdir()
    (tmp_path / "reports" / "T999").mkdir(parents=True)
    (tmp_path / "schemas" / "task.schema.json").write_text(
        Path("schemas/task.schema.json").read_text()
    )
    (tmp_path / "schemas" / "task-acceptance.schema.json").write_text(
        Path("schemas/task-acceptance.schema.json").read_text()
    )
    (tmp_path / "tasks" / "T999-task.md").write_text(
        """---
id: T999
title: Submission test
parent: null
status: awaiting_review
priority: P0
owner: test
reviewer: reviewer
branch: agent/T999
depends_on: []
blocks: []
allowed_paths: ["tasks/T999-task.md", "reports/T999/"]
source_revision: "SOURCE_REVISION_PLACEHOLDER"
created_at: 2026-08-28
updated_at: 2026-08-28
---

# T999
"""
    )
    (tmp_path / "tasks" / "contracts" / "T999.acceptance.yaml").write_text(
        """
schema_version: 1
task_id: T999
base_ref: main
required_files: [reports/T999/result-summary.md]
commands:
  - name: smoke
    command: printf ok
metrics:
  - file: reports/T999/result.json
    path: passed
    op: "=="
    value: true
forbidden_claims: []
"""
    )
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "base", cwd=tmp_path)
    base_revision = _run("git", "rev-parse", "HEAD", cwd=tmp_path).stdout.strip()
    task_path = tmp_path / "tasks" / "T999-task.md"
    task_path.write_text(
        task_path.read_text().replace("SOURCE_REVISION_PLACEHOLDER", base_revision)
    )
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "pin source revision", cwd=tmp_path)
    _run("git", "switch", "-c", "agent/T999", cwd=tmp_path)
    (tmp_path / "reports" / "T999" / "result-summary.md").write_text("complete")
    (tmp_path / "reports" / "T999" / "result.json").write_text(
        json.dumps({"passed": True})
    )
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "result", cwd=tmp_path)

    completed = _run(
        sys.executable,
        "-m",
        "comppareto.repo_state.submission_cli",
        "--root",
        str(tmp_path),
        "--task",
        "T999",
        cwd=Path.cwd(),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "submission_validation=pass task=T999" in completed.stdout


def test_submission_cli_tolerates_missing_acceptance_contract(tmp_path: Path) -> None:
    """A missing tasks/contracts/<TASK>.acceptance.yaml must not fail the run.

    T130 has no acceptance contract file. The validator should skip all
    contract-dependent checks (required files, metrics, forbidden claims,
    commands) but still run the core git/task/ancestry checks, and exit 0.
    """
    _run("git", "init", "-b", "main", cwd=tmp_path)
    _run("git", "config", "user.email", "test@example.com", cwd=tmp_path)
    _run("git", "config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "tasks" / "contracts").mkdir(parents=True)
    (tmp_path / "schemas").mkdir()
    (tmp_path / "reports" / "T998").mkdir(parents=True)
    (tmp_path / "schemas" / "task.schema.json").write_text(
        Path("schemas/task.schema.json").read_text()
    )
    (tmp_path / "tasks" / "T998-task.md").write_text(
        """---
id: T998
title: Missing contract submission test
parent: null
status: awaiting_review
priority: P0
owner: test
reviewer: reviewer
branch: agent/T998
depends_on: []
blocks: []
allowed_paths: ["tasks/T998-task.md", "reports/T998/"]
source_revision: "SOURCE_REVISION_PLACEHOLDER"
created_at: 2026-08-28
updated_at: 2026-08-28
---

# T998
"""
    )
    # Deliberately do NOT create tasks/contracts/T998.acceptance.yaml.
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "base", cwd=tmp_path)
    base_revision = _run("git", "rev-parse", "HEAD", cwd=tmp_path).stdout.strip()
    task_path = tmp_path / "tasks" / "T998-task.md"
    task_path.write_text(
        task_path.read_text().replace("SOURCE_REVISION_PLACEHOLDER", base_revision)
    )
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "pin source revision", cwd=tmp_path)
    _run("git", "switch", "-c", "agent/T998", cwd=tmp_path)
    (tmp_path / "reports" / "T998" / "result-summary.md").write_text("complete")
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "result", cwd=tmp_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        returncode = _submission_main(["--root", str(tmp_path), "--task", "T998"])
    output = buf.getvalue()
    assert returncode == 0, output
    assert "submission_validation=pass task=T998" in output


def test_submission_cli_still_fails_on_malformed_existing_contract(
    tmp_path: Path,
) -> None:
    """A present-but-invalid contract file must still fail (missing-contract
    tolerance must not mask genuinely broken contracts)."""
    _run("git", "init", "-b", "main", cwd=tmp_path)
    _run("git", "config", "user.email", "test@example.com", cwd=tmp_path)
    _run("git", "config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "tasks" / "contracts").mkdir(parents=True)
    (tmp_path / "schemas").mkdir()
    (tmp_path / "reports" / "T997").mkdir(parents=True)
    (tmp_path / "schemas" / "task.schema.json").write_text(
        Path("schemas/task.schema.json").read_text()
    )
    (tmp_path / "schemas" / "task-acceptance.schema.json").write_text(
        Path("schemas/task-acceptance.schema.json").read_text()
    )
    (tmp_path / "tasks" / "T997-task.md").write_text(
        """---
id: T997
title: Malformed contract test
parent: null
status: awaiting_review
priority: P0
owner: test
reviewer: reviewer
branch: agent/T997
depends_on: []
blocks: []
allowed_paths: ["tasks/T997-task.md", "reports/T997/"]
source_revision: "SOURCE_REVISION_PLACEHOLDER"
created_at: 2026-08-28
updated_at: 2026-08-28
---

# T997
"""
    )
    (tmp_path / "tasks" / "contracts" / "T997.acceptance.yaml").write_text(
        "not: [valid, contract"
    )
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "base", cwd=tmp_path)
    base_revision = _run("git", "rev-parse", "HEAD", cwd=tmp_path).stdout.strip()
    task_path = tmp_path / "tasks" / "T997-task.md"
    task_path.write_text(
        task_path.read_text().replace("SOURCE_REVISION_PLACEHOLDER", base_revision)
    )
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "pin source revision", cwd=tmp_path)
    _run("git", "switch", "-c", "agent/T997", cwd=tmp_path)
    (tmp_path / "reports" / "T997" / "result-summary.md").write_text("complete")
    _run("git", "add", ".", cwd=tmp_path)
    _run("git", "commit", "-m", "result", cwd=tmp_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        returncode = _submission_main(["--root", str(tmp_path), "--task", "T997"])
    output = buf.getvalue()
    assert returncode == 1, output
    assert "submission_validation=fail task=T997" in output
