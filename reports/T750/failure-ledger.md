# T750 failure ledger

## Resolved issues (not outstanding — recorded for completeness)

### 1. PyTorch is not installed anywhere in this execution environment

- **Symptom**: an earlier (compacted) session's summary claimed a successful `import torch`
  producing `2.14.0+cpu`; this was not reproducible in this session. Every module written under the
  assumption that `torch` was available (`mock_corl.py`, `corl_adapter.py`, `equivalence.py`,
  `reconstruction.py`, plus parts of `recorder.py`) failed at import time with
  `ModuleNotFoundError: No module named 'torch'`.
- **Root cause**: an exhaustive search (every sibling worktree's `.venv`, the main repo's `.venv`,
  every system-wide `site-packages` directory) found no `torch` build for the system's `cp311`
  interpreter anywhere reachable from this container.
- **Disposition**: not a blocker. Rewrote the entire instrumentation layer and mock model in pure
  NumPy — already a declared `pyproject.toml` dependency — with hand-derived backpropagation and a
  hand-implemented AdamW/gradient-clip that reproduce `torch.optim.AdamW` /
  `torch.nn.utils.clip_grad_norm_`'s exact formulas. This is arguably a more faithful reading of the
  task's own "framework-neutral" framing than requiring torch would have been. See
  `runs/instrumentation-corl-v1/notes.md` for the full account.

### 2. `recorder.py::task_statistics` `NameError` on `norm_ratio`

- **Symptom**: `task_statistics()` called `norm_ratio(...)`, but the module's import line only
  listed `cosine, directional_derivative, gram_matrix, is_finite, l2_norm, near_zero_rate` —
  `norm_ratio` was missing, causing `NameError: name 'norm_ratio' is not defined` during the first
  smoke test.
- **Disposition**: fixed by expanding the import to include `norm_ratio`. Caught before any test
  file was written against the buggy behavior; not visible in the final committed state.

### 3. Stray `import torch` left in `experiment.py`

- **Symptom**: a leftover, unused `import torch` from an earlier (pre-pivot) draft caused
  `ModuleNotFoundError: No module named 'torch'` when running the top-level experiment orchestrator,
  even after every other module had been converted to NumPy.
- **Disposition**: fixed by removing the dead import line.

### 4. `.venv` creation race condition and accidental partial deletion

- **Symptom**: a backgrounded `python3 -m venv --system-site-packages .venv` command appeared (on a
  premature read of `pyvenv.cfg` mid-write) to have `include-system-site-packages = false`; a
  subsequent `numpy` import failed. Believing the venv was broken, `rm -rf .venv` was run — but the
  venv had actually finished successfully in the background by that point. The `rm` partially
  succeeded (deleted `bin`/`lib`/`lib64`/`include`) but failed on the top-level directory itself
  (`Directory not empty`, a lingering `pyvenv.cfg`), leaving a corrupt half-deleted state.
- **Disposition**: fixed by removing the leftover `pyvenv.cfg` and the now-empty directory, then
  recreating the venv from scratch via a backgrounded command run to full completion (confirmed via
  its own completion notification). No data or code was lost — only the disposable `.venv` directory
  was affected, never anything under `src/` or `tests/`.

### 5. `pip install -e .` failure — missing `wheel`/`bdist_wheel`

- **Symptom**: `.venv/bin/python -m pip install --no-build-isolation -e .` failed with
  `error: invalid command 'bdist_wheel'`.
- **Root cause**: the `wheel` package was not present even with `--system-site-packages`.
- **Disposition**: fixed by installing `wheel` and `setuptools` from PyPI (`pypi.org` is reachable
  from this environment), then re-running the editable install, which succeeded and also pulled in
  the previously-missing `jsonschema` dependency (needed by `comppareto.repo_state.cli`). This was
  necessary for the acceptance contract's literal `full-tests` command
  (`.venv/bin/python -m pytest -q`, run bare, no `PYTHONPATH` override) to pass, since four
  pre-existing, out-of-`allowed_paths` tests spawn `comppareto....` as a subprocess and need the
  package importable outside pytest's own `pythonpath=["src"]` rewrite.

## Non-issues actively checked and ruled out

### 1. `pytest -q`'s bare invocation picking up `pythonpath=["src"]`

- **Checked because**: the acceptance contract's literal `full-tests` command has no explicit
  `PYTHONPATH` override, and all in-development smoke testing had used
  `PYTHONPATH=src .venv/bin/python -c "..."` instead.
- **Finding**: `pyproject.toml`'s `[tool.pytest.ini_options] pythonpath = ["src"]` is sufficient on
  its own — `.venv/bin/python -m pytest -q tests/instrumentation/` passed 39/39 with no environment
  variable set.
- **Disposition**: not an issue.

### 2. Whether the instrumented-vs-uninstrumented equivalence result would be suspiciously exact
   (`0.0`)

- **Checked because**: a bit-for-bit-identical result across two independently-summed float32
  accumulation paths would be a red flag (either a trivial/degenerate test, or the two paths
  weren't actually independent).
- **Finding**: the measured errors are genuinely nonzero
  (`gradient_max_abs_error = 2.384185791015625e-07`,
  `parameter_update_max_abs_error = 7.058704565299223e-09`), consistent with real
  summation-order floating-point artifacts, and comfortably under the `1e-6` gate. The
  reconstruction check (a different comparison — logged data reconstructed against itself, not two
  independent computation paths) is legitimately exact (`0.0`), which is expected and reported as
  such, not conflated with the equivalence numbers above.
- **Disposition**: not an issue; both numbers are reported as measured, not asserted.

## Unresolved anomalies

None open at submission time. No pre-existing repository defect was found outside
`allowed_paths` — all 272 tests in the full repository suite pass after the editable-install fix
above (item 5), including the four tests that previously failed due to the missing package
install.

## Summary

- 5 issues encountered during development, all self-caught and resolved within this session (no
  human intervention required); none affected the final committed state's correctness.
- 0 unresolved anomalies at submission time.
- 2 non-issues actively checked and ruled out (both documented above for completeness, since they
  were plausible failure modes worth verifying rather than assuming).
- Stage 5 (optional T710 assets) was permissibly skipped, not a failure — see
  `reports/T750/claim-check.md`'s execution-stages table and
  `tasks/T750-corl-gradient-update-instrumentation.md`'s review history.
