# Task Submission Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent remote task submissions from reaching review with path violations, stale branches, invalid manifests, missing artifacts, failed tests, or unsupported completion claims.

**Architecture:** Add a machine-readable acceptance contract per task and a repository-native submission validator that composes existing task/run validation with branch, changed-path, required-file, command, metric, and claim-consistency checks. Add a separate storage preflight utility for model tasks so checkpoint/cache placement and filesystem assumptions are recorded before expensive execution.

**Tech Stack:** Python 3.11+, JSON Schema, PyYAML, pytest, git subprocesses, pathlib.

---

### Task 1: Acceptance contract schema and loader

**Files:**
- Create: `schemas/task-acceptance.schema.json`
- Create: `src/comppareto/repo_state/acceptance.py`
- Create: `tests/repo_state/test_acceptance.py`

- [x] Write tests for valid contracts, missing required files, malformed metric rules, and forbidden claims.
- [x] Run the tests and verify they fail because the loader does not exist.
- [x] Implement schema validation and typed contract loading.
- [x] Run the tests and verify they pass.

### Task 2: Branch and allowed-path validation

**Files:**
- Create: `src/comppareto/repo_state/submission.py`
- Create: `tests/repo_state/test_submission.py`

- [x] Write tests for branch mismatch, base revision not being an ancestor, unauthorized changed paths, missing required files, and legal submissions.
- [x] Verify the tests fail.
- [x] Implement git-based submission checks.
- [x] Verify the tests pass.

### Task 3: Evidence and metric gates

**Files:**
- Modify: `src/comppareto/repo_state/submission.py`
- Modify: `tests/repo_state/test_submission.py`

- [x] Write tests for JSON-path metric comparisons, run-status/claim contradictions, missing artifact metadata, and failed command evidence.
- [x] Verify the tests fail.
- [x] Implement the metric and evidence checks.
- [x] Verify the tests pass.

### Task 4: Unified CLI

**Files:**
- Create: `src/comppareto/repo_state/submission_cli.py`
- Modify: `pyproject.toml`
- Create: `scripts/validate_task_submission.sh`
- Create: `tests/repo_state/test_submission_cli.py`

- [x] Write CLI tests for pass/fail exit codes and readable diagnostics.
- [x] Verify the tests fail.
- [x] Implement `comppareto-validate-submission` and the shell wrapper.
- [x] Verify CLI tests pass.

### Task 5: Model storage preflight

**Files:**
- Create: `src/comppareto/repo_state/storage_preflight.py`
- Create: `scripts/model_storage_preflight.py`
- Create: `tests/repo_state/test_storage_preflight.py`

- [x] Write tests for local-vs-network filesystem classification, free-space checks, hash comparison, and offline-cache environment checks.
- [x] Verify the tests fail.
- [x] Implement the preflight report generator and validator.
- [x] Verify the tests pass.

### Task 6: T155 and T210 contracts

**Files:**
- Create: `tasks/contracts/T155.acceptance.yaml`
- Create: `tasks/contracts/T210.acceptance.yaml`
- Modify: `tasks/T155-exact-finite-response-oracle.md`
- Modify: `tasks/T210-showo2-admission.md`
- Modify: `AGENTS.md`
- Modify: `reports/README.md`

- [x] Encode required files, commands, path policies, metrics, and forbidden claims.
- [x] Require model storage preflight for T210.
- [x] Document that `awaiting_review` requires a passing submission-validator report.
- [x] Run contract validation.

### Task 7: Full verification

- [x] Run repository validator.
- [x] Run complete pytest suite.
- [x] Run Python compilation and shell syntax checks.
- [x] Run local-link and whitespace checks.
- [x] Run the new submission validator against controlled passing and failing fixtures.
- [x] Commit and push the verified implementation.
