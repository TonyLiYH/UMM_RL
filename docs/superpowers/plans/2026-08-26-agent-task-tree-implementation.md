# Agent Task Tree Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing CompPareto repository into the GitHub-backed UMM_RL collaboration repository used by local planning/review agents and remote GPU execution agents.

**Architecture:** Keep research knowledge, task control, executable code, and lightweight evidence in one repository with explicit directory ownership. Markdown task files with YAML front matter are authoritative; Python validators enforce task-tree and run-manifest invariants; GitHub Actions applies the same checks on pushes and pull requests.

**Tech Stack:** Markdown, YAML, JSON Schema, Python 3.11+, PyYAML, jsonschema, pytest, GitHub Actions, Git.

---

## File map

### Project control and navigation

- Create `CHANGELOG.md`: factual repository and experiment history.
- Modify `README.md`: collaboration model, navigation, quick-start, and remote branch flow.
- Modify `AGENTS.md`: local planner/reviewer and remote executor authority.
- Modify `PROGRESS.md`: current task entry and task-tree link.
- Create `worklog/README.md`: worklog ownership.
- Create `worklog/2026-08/2026-08-26.md`: repository initialization record.
- Create `reports/README.md`: result-review contract.

### Task tree

- Create `tasks/README.md`: authoritative tree, active task table, status legend.
- Create `tasks/T000-root-research.md`.
- Create `tasks/T100-t1b-validation.md`.
- Create `tasks/T110-overlap-family.md`.
- Create `tasks/T120-independent-kkt-reference.md`.
- Create `tasks/T130-indefinite-trust-region.md`.
- Create `tasks/T140-approximation-error.md`.
- Create `tasks/T150-negotiation-audit.md`.
- Create `tasks/T200-model-admission.md`.
- Create `tasks/T210-showo2-admission.md`.
- Create `tasks/T220-uniddt-admission.md`.
- Create `tasks/T230-sensenova-u1-admission.md`.
- Create `tasks/T240-uniar-admission.md`.
- Create `tasks/T300-d0-conflict-diagnostics.md`.
- Create `tasks/T310-parameter-block-registry.md`.
- Create `tasks/T320-hypergradient-cache.md`.
- Create `tasks/T330-predictor-comparison.md`.
- Create `tasks/T340-calibration-audit.md`.
- Create `tasks/T400-e1-showo2-pilot.md`.
- Create `tasks/T410-budget-search-freeze.md`.
- Create `tasks/T420-strong-baseline-wave.md`.
- Create `tasks/T430-comppareto-wave.md`.
- Create `tasks/T440-confirmatory-evaluation.md`.
- Create `tasks/T500-e2-architecture-transfer.md`.
- Create `tasks/T600-e3-heterogeneous-posttraining.md`.
- Create `tasks/archive/README.md`.

### Schemas and validation

- Create `schemas/task.schema.json`: task front-matter field schema.
- Create `schemas/run-manifest.schema.json`: formal run provenance schema.
- Create `src/comppareto/repo_state/__init__.py`.
- Create `src/comppareto/repo_state/tasks.py`: task parser and graph validation.
- Create `src/comppareto/repo_state/runs.py`: manifest validation.
- Create `src/comppareto/repo_state/cli.py`: combined repository validation CLI.
- Create `scripts/validate_research_state.py`: portable command wrapper.
- Create `tests/repo_state/test_tasks.py`.
- Create `tests/repo_state/test_runs.py`.
- Create `tests/repo_state/test_cli.py`.
- Modify `pyproject.toml`: add PyYAML/jsonschema and console script.

### GitHub collaboration

- Create `.github/pull_request_template.md`.
- Create `.github/workflows/validate-research-state.yml`.

## Task 1: Establish task and evidence documentation

**Files:**

- Create all files listed under “Project control and navigation” and “Task tree”.
- Modify `README.md`, `AGENTS.md`, and `PROGRESS.md`.

- [ ] **Step 1: Create the task status and authority contract**

Use the exact status vocabulary:

```yaml
planned: local-defined, not executable
ready: local-authorized
running: remote-executing
awaiting_review: remote-submitted
revision_needed: local-returned
blocked: exact external blocker recorded
accepted: local-reviewed and merged
stopped: local Gate failure or deliberate termination
```

- [ ] **Step 2: Create the initial task tree**

Use the exact dependency roots:

```text
T000
├── T100 -> T110, T120, T130, T140, T150
├── T200 -> T210, T220, T230, T240
├── T300 -> T310, T320, T330, T340
├── T400 -> T410, T420, T430, T440
├── T500
└── T600
```

Set the initial executable entries to:

```text
T110 ready
T120 ready
T130 ready
T210 ready
```

Set `T100`, `T200`, and `T000` to `running`; all other tasks remain `planned`.

- [ ] **Step 3: Give every task a complete contract**

Every task file contains YAML front matter followed by these headings:

```markdown
## Research claim
## Objective
## Dependencies and inputs
## Allowed changes
## Frozen protocol
## Execution stages
## Pass/fail gate
## First report
## Required deliverables
## Artifact and provenance requirements
## Failure and retry rules
## Successor opening
## Review history
```

- [ ] **Step 4: Add project navigation and authority rules**

`README.md` links `tasks/README.md`, `CHANGELOG.md`, `worklog/`, and `reports/`. `AGENTS.md` states remote executors never push to `main` and may only advance their task to `awaiting_review` or `blocked`.

- [ ] **Step 5: Validate documentation consistency**

Run:

```bash
git diff --check
python3 - <<'PY'
from pathlib import Path
required = {"README.md", "PROJECT.md", "PROGRESS.md", "CHANGELOG.md", "AGENTS.md"}
assert all(Path(name).exists() for name in required)
assert len(list(Path("tasks").glob("T*.md"))) == 24
print("documentation structure ok")
PY
```

Expected:

```text
documentation structure ok
```

- [ ] **Step 6: Commit**

```bash
git add README.md AGENTS.md PROGRESS.md CHANGELOG.md tasks worklog reports
git commit -m "docs: add authoritative agent task tree"
```

## Task 2: Implement task-tree validation with TDD

**Files:**

- Create `schemas/task.schema.json`.
- Create `src/comppareto/repo_state/__init__.py`.
- Create `src/comppareto/repo_state/tasks.py`.
- Create `tests/repo_state/test_tasks.py`.

- [ ] **Step 1: Write failing task parser and graph tests**

Tests cover:

```python
def test_loads_task_front_matter_and_body(tmp_path): ...
def test_rejects_duplicate_task_ids(tmp_path): ...
def test_rejects_missing_parent(tmp_path): ...
def test_rejects_dependency_cycle(tmp_path): ...
def test_rejects_ready_task_with_unaccepted_dependency(tmp_path): ...
def test_rejects_remote_terminal_acceptance_without_review(tmp_path): ...
def test_accepts_initial_repository_task_tree(): ...
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/repo_state/test_tasks.py -q
```

Expected: collection fails because `comppareto.repo_state.tasks` does not exist.

- [ ] **Step 3: Implement the task parser**

Public interface:

```python
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

def load_task(path: Path, schema_path: Path) -> TaskRecord: ...
def load_tasks(tasks_dir: Path, schema_path: Path) -> dict[str, TaskRecord]: ...
def validate_task_graph(tasks: dict[str, TaskRecord]) -> list[str]: ...
```

- [ ] **Step 4: Implement graph invariants**

Return deterministic human-readable errors for:

- duplicate IDs;
- invalid root count;
- missing parent/dependency;
- parent/dependency cycles;
- `ready` with unaccepted dependencies;
- `ready` or later without branch/owner/reviewer/source revision;
- `accepted` without a local review record in the task body.

- [ ] **Step 5: Run tests and verify GREEN**

```bash
.venv/bin/python -m pytest tests/repo_state/test_tasks.py -q
```

Expected: all task validation tests pass.

- [ ] **Step 6: Commit**

```bash
git add schemas/task.schema.json src/comppareto/repo_state tests/repo_state/test_tasks.py pyproject.toml
git commit -m "feat: validate authoritative task graph"
```

## Task 3: Implement run-manifest validation with TDD

**Files:**

- Create `schemas/run-manifest.schema.json`.
- Create `src/comppareto/repo_state/runs.py`.
- Create `tests/repo_state/test_runs.py`.

- [ ] **Step 1: Write failing run validation tests**

Tests cover:

```python
def test_accepts_committed_t1a_manifest(): ...
def test_rejects_missing_task_id(tmp_path): ...
def test_rejects_dirty_formal_run(tmp_path): ...
def test_rejects_missing_result_reference(tmp_path): ...
def test_rejects_zero_sized_artifact(tmp_path): ...
def test_rejects_retry_without_failure_reason(tmp_path): ...
```

- [ ] **Step 2: Run tests and verify RED**

```bash
.venv/bin/python -m pytest tests/repo_state/test_runs.py -q
```

Expected: collection fails because `comppareto.repo_state.runs` does not exist.

- [ ] **Step 3: Implement manifest validation**

Public interface:

```python
@dataclass(frozen=True)
class RunValidationResult:
    path: Path
    errors: tuple[str, ...]

def validate_run_manifest(
    path: Path,
    schema_path: Path,
    repository_root: Path,
) -> RunValidationResult: ...

def validate_run_tree(
    runs_dir: Path,
    schema_path: Path,
    repository_root: Path,
) -> list[str]: ...
```

The validator accepts the existing T1a manifest as a legacy formal record by mapping its source revision and environment fields, while requiring the expanded schema for new manifests.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
.venv/bin/python -m pytest tests/repo_state/test_runs.py -q
```

Expected: all run validation tests pass.

- [ ] **Step 5: Commit**

```bash
git add schemas/run-manifest.schema.json src/comppareto/repo_state/runs.py tests/repo_state/test_runs.py
git commit -m "feat: validate formal run provenance"
```

## Task 4: Add the combined validation command and CI

**Files:**

- Create `src/comppareto/repo_state/cli.py`.
- Create `scripts/validate_research_state.py`.
- Create `tests/repo_state/test_cli.py`.
- Create `.github/pull_request_template.md`.
- Create `.github/workflows/validate-research-state.yml`.
- Modify `pyproject.toml`.

- [ ] **Step 1: Write the failing CLI test**

```python
def test_cli_validates_repository():
    completed = subprocess.run(
        [sys.executable, "-m", "comppareto.repo_state.cli", "--root", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "task_tree=pass" in completed.stdout
    assert "run_manifests=pass" in completed.stdout
```

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest tests/repo_state/test_cli.py -q
```

Expected: failure because the CLI module does not exist.

- [ ] **Step 3: Implement the CLI**

Command:

```bash
.venv/bin/python -m comppareto.repo_state.cli --root .
```

Successful output:

```text
task_tree=pass tasks=24
run_manifests=pass manifests=1
research_state=pass
```

Any invariant violation prints one line per error and exits nonzero.

- [ ] **Step 4: Add GitHub collaboration files**

The pull request template requires:

- task ID;
- source revision;
- changed paths;
- tests;
- run IDs;
- artifacts;
- failures/retries;
- task status `awaiting_review`;
- explicit confirmation that no frozen protocol changed.

The workflow installs `.[test]`, runs the combined validator, pytest, compileall, shell syntax checks, Markdown local-link validation, and `git diff --check`.

- [ ] **Step 5: Run tests and verify GREEN**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m comppareto.repo_state.cli --root .
```

Expected: all tests pass and `research_state=pass`.

- [ ] **Step 6: Commit**

```bash
git add src/comppareto/repo_state/cli.py scripts/validate_research_state.py tests/repo_state/test_cli.py .github pyproject.toml
git commit -m "ci: enforce research task and evidence state"
```

## Task 5: Verify, integrate, configure remote, and push

**Files:**

- Update `CHANGELOG.md`.
- Update `worklog/2026-08/2026-08-26.md`.

- [ ] **Step 1: Run full local verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m comppareto.repo_state.cli --root .
.venv/bin/python -m compileall -q src tests scripts
bash -n scripts/run_t1.sh
git diff --check
```

Expected: zero failures and `research_state=pass`.

- [ ] **Step 2: Verify repository provenance**

```bash
git status --short
git log --oneline -8
git remote -v
git ls-remote git@github.com:alexlovecoding/UMM_RL.git HEAD
```

Expected before initial push: clean feature branch; remote reachable and may have no `HEAD`.

- [ ] **Step 3: Integrate the feature branch**

From the main worktree:

```bash
git merge --no-ff feature/agent-task-tree
```

Expected: merge succeeds without conflicts.

- [ ] **Step 4: Configure the remote**

```bash
git remote add origin git@github.com:alexlovecoding/UMM_RL.git
```

If `origin` already exists, verify its URL is exactly the authorized URL instead of overwriting it.

- [ ] **Step 5: Verify main after merge**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m comppareto.repo_state.cli --root .
git status --short
```

Expected: all tests pass, repository validation passes, working tree clean.

- [ ] **Step 6: Push**

```bash
git push -u origin main
```

Expected: GitHub `main` is created and tracks `origin/main`.

- [ ] **Step 7: Verify remote state**

```bash
git ls-remote --heads origin main
git status -sb
```

Expected: remote `main` points to the local `main` commit and the branch is up to date.

