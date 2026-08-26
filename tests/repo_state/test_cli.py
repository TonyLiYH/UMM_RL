from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import shutil


def test_cli_validates_repository() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "comppareto.repo_state.cli", "--root", "."],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "task_tree=pass tasks=24" in completed.stdout
    assert "run_manifests=pass manifests=1" in completed.stdout
    assert "research_state=pass" in completed.stdout


def test_cli_reports_run_schema_error_as_run_failure(tmp_path: Path) -> None:
    shutil.copytree("tasks", tmp_path / "tasks")
    shutil.copytree("schemas", tmp_path / "schemas")
    run_dir = tmp_path / "runs" / "invalid"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        """{
  "schema_version": 1,
  "run_id": "invalid",
  "run_kind": "formal",
  "source_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "execution_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "dirty": false,
  "config_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "environment": {"python": "3.13"},
  "status": "fail",
  "result_files": [],
  "artifacts": [],
  "retry": null
}
"""
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "comppareto.repo_state.cli",
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "task_tree=pass tasks=24" in completed.stdout
    assert "run_manifests=fail manifests=1" in completed.stdout
    assert "'task_id' is a required property" in completed.stdout
