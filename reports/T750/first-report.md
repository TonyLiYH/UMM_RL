# T750 first report — CoRL gradient/optimizer-update instrumentation

Per `tasks/T750-corl-gradient-update-instrumentation.md`, this is the initial audit/design report,
published before the bulk of the instrumentation code was finalized. Branch:
`agent/T750-corl-gradient-update-instrumentation`. This is a CPU-default software-engineering task
— no GPU execution, no admission of a third-party model repository.

## Scope confirmation

The task asks for a **framework-neutral** instrumentation layer (per-task shared gradients before
clipping, combined gradients after negotiation/clipping, realized AdamW parameter updates by
block), plus a thin CoRL-shaped adapter, validated through four mandatory execution stages
(deterministic toy tests; sequential task batches at one shared-state hash; an
instrumented-vs-uninstrumented equivalence proof; a mock CoRL-compatible model) and one optional
stage (a real T710 diagnostic batch, only if T710's admitted assets are already available).

## T710 availability check (governs whether stage 5 is attempted)

Checked at branch-creation time and reconfirmed in this session: `T710`
(`agent/T710-corl-admission-gpu-smoke`) is `status: running`, and `runs/corl-admission-v1/` does
not exist. Per the task brief's own wording ("optionally use T710 assets... if available"), stage 5
is skipped for this submission. This is already recorded in
`tasks/T750-corl-gradient-update-instrumentation.md`'s review history (2026-09-16 entry). A live
peer session working T710 confirmed during this session that it is building its own
frozen-head-gradient-leak probe and optimizer-diffing logic directly against the real model, inside
T710's own `allowed_paths`, and does not intend to reuse T750's mock-model code directly (it needs
real `torch` and named-parameter-based diffing against `param_policy.py`). No T710 dependency is
introduced anywhere in this submission.

## Framework decision: NumPy, not PyTorch

Before writing any instrumentation code, this session exhaustively searched for a working `torch`
installation: every sibling worktree's `.venv` (`T230`, `T260`, `T710`, `T720`), the main
`UMM_RL/.venv` (a symlink to the system `/usr/bin/python3`, which has no site-packages of its own),
and every system-wide `site-packages` directory reachable from this container. None had `torch`
importable for the system's `cp311` interpreter. `pypi.org` itself is reachable, but installing a
fresh multi-hundred-MB `torch` wheel was judged out of proportion for a task whose own framing is
"framework-neutral" — requiring one specific deep-learning framework to validate a
framework-neutral instrumentation layer would be a scope mismatch, not a faithful implementation of
the brief.

Consequently every module under `src/comppareto/instrumentation/` is pure NumPy, including a
hand-derived backpropagation pass for the toy CoRL-shaped model
(`mock_corl.py::MockCoRLModel`, Linear+tanh chains) and hand-implemented AdamW
(`optim.py::AdamWOptimizer`, exact decoupled-weight-decay formula per Loshchilov & Hutter 2019) and
gradient-norm clipping (`recorder.py`'s `apply_and_clip`, exactly replicating
`torch.nn.utils.clip_grad_norm_`'s `clip_coef = min(max_norm / (total_norm + eps), 1.0)` formula).
Every formula reimplemented is reproduced exactly, not approximated — see
`runs/instrumentation-corl-v1/notes.md` for the full rationale.

## Design: block registry, shared-state-hash protocol, recorder

- `blocks.py::ParameterBlock`/`BlockRegistry` — each block owns a disjoint set of parameters and
  declares its owning task IDs (`overlap_tasks`); `overlap_id` is the `+`-joined sorted task-ID
  string (e.g. `"understanding+generation+auxiliary"`); `validate_ownership` raises `OwnershipError`
  on any unassigned, double-owned, or unknown-reference parameter; `unassigned_trainable_parameters`
  gives the exact completeness count the acceptance gate checks.
- `recorder.py::GradientUpdateRecorder` — `begin_batch`/`check_shared_state` implement the
  sequential-task-batches-at-one-shared-state-hash protocol (stage 2): every task in a batch must
  observe the identical shared-parameter content hash; any mid-batch mutation raises
  `RecorderProtocolError`, never silently continues. `capture_task_gradient` records per-task
  norm/cosine/norm-ratio/finite status and per-block-local norms. `combine()` produces PCGrad-ready
  zero-padded vectors and the MGDA convex-hull Gram matrix. `apply_and_clip` records pre/post-clip
  norms and the clip coefficient. `step_and_record` records AdamW moment summaries (before/after)
  and the realized `Δθ` by block, plus direction norm, near-zero rate, and directional derivative.
- `counters.py::StepCounters` — rollout/token/reward-call/backward counts and per-phase wall-time.
- `device.py::resolve_device` — the only supported device string is `"cpu"`; anything else raises
  `DeviceUnavailableError` immediately (no silent fallback), satisfying the pass/fail gate's
  explicit requirement.
- `corl_adapter.py::CoRLGRPOAdapter` — the thin CoRL-shaped glue: drives one combined-update step
  (sequential per-task forward+gradient-capture at one shared-state hash, then one combined clip +
  AdamW step) against any model satisfying the minimal `CoRLTaskModel` protocol
  (`block_registry()`, `forward_task(task_id, batch) -> (loss, RolloutInfoLike)`).
- `equivalence.py::run_equivalence_check` — stage 3: builds two independent model/optimizer pairs
  from the same seed, runs one through a continuous-accumulation baseline path and the other
  through the instrumented adapter, and reports the measured max-abs-error for both the pre-clip
  gradient and the realized parameter update.
- `reconstruction.py::check_reconstruction` — independently recomputes the combined gradient (as a
  sum over each block's captured per-task arrays) and the realized update (as post-step minus
  pre-step parameter snapshot) purely from what the recorder logged, and compares both against the
  recorder's own numbers — proving the logs are sufficient to reconstruct both required quantities.
- `experiment.py::run_experiment` — top-level orchestrator producing the exact metrics structure
  the acceptance contract checks.

## Plan for the remainder of this submission

1. Write the full `tests/instrumentation/` suite (stages 1-4). Done — 39 tests across four files,
   all passing.
2. Run the full local validation stack (`pytest -q`, `compileall -q src tests`,
   `comppareto.repo_state.cli`, `git diff --check`). Done — see `reports/T750/result-summary.md`.
3. Produce `configs/instrumentation/corl/resolved-config.yaml` and
   `runs/instrumentation-corl-v1/{manifest.json,metrics.json,notes.md}` with the real measured
   numbers. Done.
4. Write the remaining reports and set `status: awaiting_review`.

No GPU work is planned or required at any stage of this submission.
