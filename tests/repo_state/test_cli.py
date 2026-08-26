from __future__ import annotations

import subprocess
import sys


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

