# T130 Failure Ledger

**Task:** T130-indefinite-trust-region

No failed experiments. All six counterexample cases behaved as analytically predicted.

## Intermediate corrections during development

### [CORRECTION] trust_region_optimum calling convention

During test writing, `trust_region_optimum` was initially called with positional
arguments. The function takes keyword-only arguments (`gradient=`, `hessian=`,
`metric=`, `radius=`). Corrected in `tests/test_trust_region_guard.py` before
any commit.

## Pre-existing test failures (unrelated to T130)

Four tests in the repository fail due to a missing `.venv` in this worktree
(subprocess tests that invoke `sys.executable -m comppareto` without a PYTHONPATH
override). These failures are pre-existing and present on a clean tree before any
T130 changes:

- `tests/repo_state/test_cli.py::test_cli_validates_repository`
- `tests/repo_state/test_cli.py::test_cli_reports_run_schema_error_as_run_failure`
- `tests/repo_state/test_submission_cli.py::test_submission_cli_accepts_clean_task_branch`
- `tests/test_synthetic_cli.py::test_synthetic_cli_writes_passing_manifest`

These are not introduced or worsened by T130 work.
