from __future__ import annotations

import json
from pathlib import Path
import subprocess

from comppareto.repo_state.acceptance import (
    AcceptanceContract,
    ForbiddenClaim,
    MetricRule,
    Trigger,
)
from comppareto.repo_state.submission import (
    check_forbidden_claims,
    check_metrics,
    check_required_files,
    path_is_allowed,
    validate_git_submission,
)


def test_path_is_allowed_for_exact_file_directory_and_glob() -> None:
    allowed = (
        "tasks/T999.md",
        "reports/T999/",
        "runs/example-*/",
    )

    assert path_is_allowed("tasks/T999.md", allowed)
    assert path_is_allowed("reports/T999/result-summary.md", allowed)
    assert path_is_allowed("runs/example-2026/manifest.json", allowed)
    assert not path_is_allowed("CHANGELOG.md", allowed)


def test_required_files_support_globs(tmp_path: Path) -> None:
    (tmp_path / "reports" / "T999").mkdir(parents=True)
    (tmp_path / "reports" / "T999" / "result-summary.md").write_text("ok")

    errors = check_required_files(
        tmp_path,
        ("reports/T999/result-summary.md", "runs/example-*/manifest.json"),
    )

    assert errors == ["missing required file pattern: runs/example-*/manifest.json"]


def test_metric_rules_read_nested_json(tmp_path: Path) -> None:
    result = tmp_path / "summary.json"
    result.write_text(json.dumps({"checks": {"coverage": 0.97, "failures": 0}}))
    rules = (
        MetricRule(str(result.relative_to(tmp_path)), "checks.coverage", ">=", 0.95),
        MetricRule(str(result.relative_to(tmp_path)), "checks.failures", "==", 0),
    )

    assert check_metrics(tmp_path, rules) == []


def test_forbidden_claim_fires_when_run_failed(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "fail"}))
    (tmp_path / "summary.md").write_text("Conclusion: Supports gate.")
    rule = ForbiddenClaim(
        file="summary.md",
        trigger=Trigger(file="manifest.json", path="status", equals="fail"),
        phrases=("supports gate",),
    )

    errors = check_forbidden_claims(tmp_path, (rule,))

    assert errors == [
        "summary.md: forbidden phrase 'supports gate' when manifest.json:status == 'fail'"
    ]


def test_git_submission_rejects_wrong_branch_and_changed_path(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "allowed").mkdir()
    (tmp_path / "allowed" / "base.txt").write_text("base")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "switch", "-c", "wrong-branch"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "CHANGELOG.md").write_text("out of scope")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "change"], cwd=tmp_path, check=True, capture_output=True)

    errors = validate_git_submission(
        root=tmp_path,
        expected_branch="agent/T999",
        base_ref="main",
        allowed_paths=("allowed/",),
    )

    assert "branch mismatch: expected agent/T999, found wrong-branch" in errors
    assert "unauthorized changed path: CHANGELOG.md" in errors


def test_git_submission_rejects_dirty_worktree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "allowed").mkdir()
    (tmp_path / "allowed" / "base.txt").write_text("base")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "switch", "-c", "agent/T999"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "allowed" / "dirty.txt").write_text("dirty")

    errors = validate_git_submission(
        root=tmp_path,
        expected_branch="agent/T999",
        base_ref="main",
        allowed_paths=("allowed/",),
    )

    assert "working tree is not clean" in errors
