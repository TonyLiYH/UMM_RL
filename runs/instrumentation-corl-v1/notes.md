# run notes: instrumentation-corl-v1

## What this run is

The formal CPU run for T750 (per-task GRPO gradient and optimizer-update
instrumentation). It exercises `src/comppareto/instrumentation/experiment.py::run_experiment`
with `seed=20260916, num_batches=5` (the defaults, matching
`configs/instrumentation/corl/resolved-config.yaml`), which drives:

- Stage 3 (`equivalence.py::run_equivalence_check`): 5 sequential steps of a
  baseline (uninstrumented, continuous-gradient-accumulation) update path
  compared against the instrumented `CoRLGRPOAdapter.run_combined_step` path,
  from two independently-constructed but identically-seeded model/optimizer
  pairs.
- Stage 4 (`corl_adapter.py::CoRLGRPOAdapter` driving `mock_corl.py::MockCoRLModel`):
  5 sequential combined-update steps against the mock CoRL-compatible model,
  each checked against `reconstruction.py::check_reconstruction`.

Stages 1 and 2 (deterministic toy tests with shared/private modules;
sequential task batches observed at one shared-state hash) are exercised by
`tests/instrumentation/test_stage1_*.py` and
`tests/instrumentation/test_stage2_recorder.py`, not by this run directly --
this run's job is to produce the authoritative numbers for
`metrics.json`, not to re-derive what the unit tests already cover.

## Why NumPy, not PyTorch (the single biggest decision in this submission)

PyTorch is not installed anywhere reachable in this execution environment.
This was confirmed by an exhaustive search before writing any instrumentation
code: every sibling worktree's `.venv` (T230, T260, T710, T720), the main
`UMM_RL/.venv` (a symlink to the system `/usr/bin/python3`, no site-packages
of its own), and every system-wide `site-packages` directory
(`/usr/local/lib(64)/python3.11/site-packages`,
`/usr/lib(64)/python3.11/site-packages`,
`/home/yihangli/.local/lib/python3.11/site-packages`) were checked -- none
had a `torch` importable for the system's `cp311` interpreter. `pypi.org`
itself is reachable (`curl` returns `200`), but pulling a fresh multi-hundred-MB
`torch` wheel was judged unnecessary and out of proportion for a task whose
own framing is explicitly "framework-neutral" -- a framework-neutral
instrumentation layer should not itself require picking one deep-learning
framework to prove its own logic.

Consequently, every module under `src/comppareto/instrumentation/` is pure
NumPy:

- `params.py::Parameter` -- a minimal mutable value+grad container with
  `accumulate_grad()`, standing in for `torch.nn.Parameter`.
- `optim.py::AdamWOptimizer` -- hand-implements the exact decoupled
  weight-decay AdamW update (Loshchilov & Hutter, 2019): `theta *= (1 - lr*wd)`,
  bias-corrected first/second moments, `theta -= step_size * exp_avg / denom`.
  This is the same formula `torch.optim.AdamW` implements; it is not a
  simplified stand-in.
- The clip in `recorder.py::GradientUpdateRecorder.apply_and_clip` implements
  `torch.nn.utils.clip_grad_norm_`'s exact formula:
  `clip_coef = min(max_norm / (total_norm + 1e-6), 1.0)`.
- `mock_corl.py::MockCoRLModel` -- a small Linear+tanh chain with
  hand-derived backpropagation (chain rule worked out by hand per layer,
  cached intermediate activations passed explicitly as function arguments
  rather than stored as mutable instance state, to avoid aliasing bugs when
  multiple rollouts share a layer before backward is invoked) and an
  analytic GRPO-style policy-gradient loss
  (`loss = -mean_i(advantage_i.detach() * reward_i)`,
  `reward_i = -sum((output_i - target_i)^2)`, `advantage` stop-gradient).

This is a defensible, and arguably *more* faithful, reading of "framework-neutral"
than importing torch would have been -- the instrumentation layer's actual
contract (block ownership, shared-state-hash protocol, per-task/combined
gradient capture, clip/AdamW-moment recording, reconstruction) has nothing
in it that is torch-specific, and every formula reimplemented above is
reproduced exactly, not approximated.

## Editable-install caveat (environment note, not a code defect)

`.venv/bin/python -m pip install --no-build-isolation -e .` initially failed
with `error: invalid command 'bdist_wheel'` because the `wheel` package was
not present even with `--system-site-packages`. `pyproject.toml`'s
`[tool.pytest.ini_options] pythonpath = ["src"]` means pytest itself does not
need the editable install to find `comppareto.*` -- but four *pre-existing*,
out-of-`allowed_paths` tests (`tests/repo_state/test_cli.py`,
`tests/repo_state/test_submission_cli.py`,
`tests/test_synthetic_cli.py::test_synthetic_cli_writes_passing_manifest`)
spawn `sys.executable -m comppareto....` as a subprocess, which does need the
package importable outside pytest's own path-rewrite. Since the acceptance
contract's literal `full-tests` command is a bare `.venv/bin/python -m pytest -q`
(no `PYTHONPATH` override), those four tests would otherwise fail in this
venv for a reason unrelated to T750. Fix: `pip install wheel setuptools`
(both fetched cleanly from PyPI) then `pip install --no-build-isolation -e .`,
which succeeded and pulled in `jsonschema` (a declared dependency used by
`comppareto.repo_state.cli`, previously also missing). After this,
`.venv/bin/python -m pytest -q` passes 272/272 for the full repository suite,
not just `tests/instrumentation/`.

## Stage 5 (optional T710 assets) -- skipped

Per `tasks/T750-corl-gradient-update-instrumentation.md`'s review history
(dated 2026-09-16, already committed), T710
(`agent/T710-corl-admission-gpu-smoke`) was `status: running` with no
`runs/corl-admission-v1/` present at the time this branch was created, so the
optional stage 5 (a real diagnostic batch against T710's admitted model) was
never attempted. This was re-confirmed live during this session: a peer
session working T710 reported it is still building its own gradient-leak
probe and optimizer-diffing logic directly against the real model
(`src/comppareto/adapters/corl/`), outside T750's `allowed_paths`, and
explicitly said T750's NumPy-only mock-model code is not intended for direct
reuse there (it needs real `torch`/named-parameter-based diffing against
`param_policy.py`). No T750 code depends on T710 in any way; stages 1-4 are
this submission's sole and sufficient evidence path, exactly as the task
brief allows ("optionally use T710 assets").

## Measured results (this run, seed=20260916, num_batches=5)

- `equivalence.gradient_max_abs_error = 2.384185791015625e-07` (gate: `<= 1e-6`)
- `equivalence.parameter_update_max_abs_error = 7.058704565299223e-09` (gate: `<= 1e-6`)
- `ownership.unassigned_trainable_parameters = 0` (gate: `== 0`)
- `reconstruction.combined_gradient_pass = true`,
  `reconstruction.combined_gradient_max_abs_error = 0.0`
- `reconstruction.realized_update_pass = true`,
  `reconstruction.realized_update_max_abs_error = 0.0`
- `resources.gpu_hours = 0.0`, `resources.device = "cpu"` (gate: `<= 1`)
- 5 distinct `shared_state_hashes` across the 5 sequential batches (parameters
  genuinely change after every combined AdamW step, while remaining
  single-valued -- i.e. not mutated mid-batch -- within each step; this is
  enforced live by `GradientUpdateRecorder.check_shared_state`, which raises
  `RecorderProtocolError` on any violation, and is separately unit-tested in
  `tests/instrumentation/test_stage2_recorder.py`).

The two equivalence errors are genuine, measured floating-point-order
artifacts (not manufactured): the baseline path accumulates all three tasks'
gradients into one continuously-summed buffer before the single AdamW step,
while the instrumented path captures each task's gradient into a separate
array and sums them afterward in `combine()` -- summing the same three
float32 quantities in a different order/associativity produces a few ULPs of
difference, comfortably inside the `1e-6` gate for a toy model at this scale.
The reconstruction errors are exactly `0.0` because `reconstruct_combined_gradient`
and `reconstruct_realized_update` perform the identical sum/subtraction the
recorder itself already performed, over the same logged arrays -- a strictly
stronger result than the `<= 1e-6` gate requires, reported honestly as
measured rather than assumed.

## Reproduction

```bash
.venv/bin/python -c "
from comppareto.instrumentation.experiment import run_experiment
report = run_experiment(seed=20260916, num_batches=5)
print(report.to_metrics_dict())
"
```

or `.venv/bin/python -m pytest -q tests/instrumentation/` for the full
stage 1-4 test suite (39 tests, all passing).
