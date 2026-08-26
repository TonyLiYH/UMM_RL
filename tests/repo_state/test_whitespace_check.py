from __future__ import annotations

from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_commit_whitespace.sh"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def initialize_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert git(repo, "init", "-b", "main").returncode == 0
    assert git(repo, "config", "user.name", "CI Test").returncode == 0
    assert git(repo, "config", "user.email", "ci@example.com").returncode == 0
    (repo / "sample.md").write_text("line\n")
    assert git(repo, "add", "sample.md").returncode == 0
    assert git(repo, "commit", "-m", "initial").returncode == 0
    return repo


def test_allows_extra_blank_line_at_end_of_file(tmp_path: Path) -> None:
    repo = initialize_repo(tmp_path)
    (repo / "sample.md").write_text("line\n\n")
    assert git(repo, "add", "sample.md").returncode == 0
    assert git(repo, "commit", "-m", "blank eof").returncode == 0

    completed = subprocess.run(
        ["bash", str(SCRIPT), "HEAD^", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_rejects_trailing_space_at_end_of_line(tmp_path: Path) -> None:
    repo = initialize_repo(tmp_path)
    (repo / "sample.md").write_text("line \n")
    assert git(repo, "add", "sample.md").returncode == 0
    assert git(repo, "commit", "-m", "trailing space").returncode == 0

    completed = subprocess.run(
        ["bash", str(SCRIPT), "HEAD^", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "trailing whitespace" in completed.stdout

