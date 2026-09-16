# T750 result summary

Task: `tasks/T750-corl-gradient-update-instrumentation.md`. Branch:
`agent/T750-corl-gradient-update-instrumentation`. This is a CPU-default software-engineering task
(no third-party model admission, no GPU execution).

## Current formal run (authoritative status)

`runs/instrumentation-corl-v1/manifest.json` — `status: pass`, `run_kind: formal`, 3
hash/byte-addressed artifacts (the resolved config plus the run's own `metrics.json`/`notes.md`).
`runs/instrumentation-corl-v1/metrics.json` carries the real measured numbers below, produced by
`comppareto.instrumentation.experiment.run_experiment(seed=20260916, num_batches=5)`.

### Measured metrics (real numbers, for independent re-verification)

| Metric | Value | Gate | Result |
|---|---|---|---|
| `equivalence.gradient_max_abs_error` | `2.384185791015625e-07` | `<= 1.0e-6` | pass |
| `equivalence.parameter_update_max_abs_error` | `7.058704565299223e-09` | `<= 1.0e-6` | pass |
| `ownership.unassigned_trainable_parameters` | `0` | `== 0` | pass |
| `reconstruction.combined_gradient_pass` | `true` (max abs error `0.0`) | `== true` | pass |
| `reconstruction.realized_update_pass` | `true` (max abs error `0.0`) | `== true` | pass |
| `resources.gpu_hours` | `0.0` | `<= 1` | pass |
| `resources.device` | `"cpu"` | n/a (no silent fallback: `device.py::resolve_device` raises on any non-`"cpu"` request) | pass |
| distinct `shared_state_hashes` across 5 sequential batches | 5 (all distinct) | n/a | parameters genuinely change every combined-update step |

The two nonzero equivalence errors are genuine, measured floating-point-order artifacts (not
manufactured to sit just under the gate): the uninstrumented baseline accumulates all three tasks'
gradients into one continuously-summed float32 buffer before a single AdamW step, while the
instrumented adapter captures each task's gradient into a separate array and sums them afterward in
`combine()` — summing the same quantities in a different order/associativity produces a few ULPs of
difference. Both reconstruction errors are exactly `0.0` (a strictly stronger result than the
`<= 1e-6` gate requires) because the reconstruction functions perform the identical
sum/subtraction the recorder itself already logged, over the same arrays.

## What was built/run

- Framework-neutral instrumentation layer: `src/comppareto/instrumentation/{params,optim,device,
  stats,blocks,recorder,counters}.py`. Pure NumPy — see "Framework decision" below.
- CoRL-shaped adapter: `src/comppareto/instrumentation/corl_adapter.py::CoRLGRPOAdapter`, driving
  one combined-update step (sequential per-task gradient capture at one shared-state hash, then one
  combined clip + AdamW step) against any model satisfying the minimal `CoRLTaskModel` protocol.
- Mock CoRL-compatible model: `src/comppareto/instrumentation/mock_corl.py::MockCoRLModel` — a
  small Linear+tanh network with three block-registry-declared task blocks (`core_shared` shared by
  all three tasks, `ug_shared` shared by understanding+generation, three private blocks) and
  hand-derived backpropagation plus an analytic GRPO-style policy-gradient loss.
- Equivalence proof: `src/comppareto/instrumentation/equivalence.py::run_equivalence_check` —
  compares an uninstrumented continuous-accumulation baseline against the instrumented adapter path
  across 5 sequential batches from the same seed.
- Reconstruction check: `src/comppareto/instrumentation/reconstruction.py::check_reconstruction` —
  independently recomputes the combined gradient and realized update purely from logged per-task
  arrays and the pre-step parameter snapshot, and compares against the recorder's own numbers.
- Full test suite: `tests/instrumentation/{test_stage1_primitives,test_stage1_blocks,
  test_stage1_mock_model_and_optim,test_stage2_recorder,test_stage3_equivalence,
  test_stage4_integration}.py` — 39 tests, all passing.
- Resolved configuration: `configs/instrumentation/corl/resolved-config.yaml` — the exact toy-model
  layout, optimizer hyperparameters, clip norm, schedule, and gate thresholds used to produce the
  above numbers.
- Formal run: `runs/instrumentation-corl-v1/{manifest.json,metrics.json,notes.md}`.

## Framework decision: NumPy, not PyTorch

PyTorch is not installed anywhere reachable in this execution environment (every sibling worktree
venv, the main repo's `.venv`, and every system-wide `site-packages` location were checked — none
had a `torch` build for the system's `cp311` interpreter). Rather than install a fresh multi-hundred-MB
`torch` wheel for a task explicitly framed as "framework-neutral," the entire instrumentation layer
and mock model were implemented in pure NumPy, including hand-derived backpropagation and a
hand-implemented AdamW/gradient-clip that exactly replicate `torch.optim.AdamW` /
`torch.nn.utils.clip_grad_norm_`'s formulas (not simplified approximations). Full rationale in
`runs/instrumentation-corl-v1/notes.md` and `reports/T750/first-report.md`.

## Local validation stack (run fresh this session, from `.venv/bin/python`)

| Command | Result |
|---|---|
| `.venv/bin/python -m comppareto.repo_state.cli --root .` | `task_tree=pass tasks=40`, `run_manifests=pass manifests=9`, `research_state=pass` |
| `.venv/bin/python -m pytest -q` | `272 passed` (full repository suite, not just `tests/instrumentation/`) |
| `.venv/bin/python -m compileall -q src tests` | clean (exit 0, no output) |
| `git diff --check origin/main...HEAD` | clean (exit 0, no output) |

Note: `.venv/bin/python -m pip install --no-build-isolation -e .` initially failed
(`invalid command 'bdist_wheel'`, missing `wheel`); fixed by installing `wheel`+`setuptools` from
PyPI, then re-running the editable install (which also pulled in the previously-missing
`jsonschema` dependency). This was necessary for the bare `pytest -q` acceptance-contract command to
pass — four pre-existing, out-of-`allowed_paths` tests spawn `comppareto....` as a subprocess and
need the package importable outside pytest's own `pythonpath=["src"]` rewrite. See
`runs/instrumentation-corl-v1/notes.md` for the full account.

## Stage 5 (optional T710 assets)

Skipped, per the task brief's own "optionally... if available" wording. T710 was `status: running`
with no `runs/corl-admission-v1/` at branch-creation time (recorded in the task file's review
history) and remained so throughout this session (reconfirmed live via a peer session working
T710). Stages 1-4 are this submission's sole and sufficient evidence path.

## Conclusion

**Supports gate.** Instrumented and uninstrumented updates match within the `1e-6` dtype tolerance
(measured, not assumed); ownership is complete (`unassigned_trainable_parameters: 0`); the logs
reconstruct both the combined gradient and the realized update exactly (`0.0` error, stronger than
required); no silent device fallback is possible (`resolve_device` raises on any non-`"cpu"`
request). `runs/instrumentation-corl-v1/manifest.json` records `status: pass`.
