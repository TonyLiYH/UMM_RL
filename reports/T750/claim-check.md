# T750 claim check

Each row maps a claim or pass/fail-gate bullet from
`tasks/T750-corl-gradient-update-instrumentation.md` to the evidence that supports or refutes it.

## Objective

> Implement a framework-neutral instrumentation layer, with a CoRL adapter, that records per-task
> shared gradients before clipping, combined gradients after negotiation/clipping, and realized
> AdamW parameter updates by block.

Supported. `src/comppareto/instrumentation/recorder.py::GradientUpdateRecorder` is
framework-neutral (operates only on `Parameter`/`BlockRegistry` primitives, no framework-specific
type anywhere in its signature); `corl_adapter.py::CoRLGRPOAdapter` is the thin CoRL-shaped adapter
that drives it. `capture_task_gradient` records each task's shared gradient before clipping;
`combine()` + `apply_and_clip` record the combined gradient after negotiation/clipping;
`step_and_record` records the realized AdamW `Δθ` by block. All three are exercised end-to-end in
`tests/instrumentation/test_stage2_recorder.py` and `tests/instrumentation/test_stage4_integration.py`.

## Required measurements

| Requirement | Status | Evidence |
|---|---|---|
| Per-task norm, cosine, norm ratio, finite status | **Done** | `recorder.py::task_statistics` — `shared_norm`, `cosine_vs_combined`, `norm_ratio_vs_combined`, `finite`; tested in `test_stage2_recorder.py::test_recorder_block_local_stats_and_overlap_ids`; real values in `runs/instrumentation-corl-v1/metrics.json`'s `per_batch_summaries[*].per_task_statistics` |
| Block-local statistics and overlap IDs | **Done** | `TaskGradientRecord.per_block_norm`; `ParameterBlock.overlap_id` (`"understanding+generation+auxiliary"` for `core_shared`, `"understanding+generation"` for `ug_shared`); tested literally in `test_stage2_recorder.py` |
| PCGrad-ready vectors and MGDA convex-hull inputs (Gram matrix) | **Done** | `CombinedGradientRecord.pcgrad_vectors` (zero-padded, shared-scope-only, one per task, all equal length) and `.gram_matrix` (symmetric, `len(TASK_IDS) x len(TASK_IDS)`, diagonal = squared norm); tested in `test_stage2_recorder.py`; real matrices in `metrics.json`'s `per_batch_summaries[*].gram_matrix` |
| Pre/post-clipping norms and clipping coefficient | **Done** | `ClippingRecord.{pre_clip_norm,post_clip_norm,clip_coefficient}`; tested in `test_stage2_recorder.py::test_recorder_clipping_math`; real values in `metrics.json`'s `per_batch_summaries[*].{pre_clip_norm,post_clip_norm,clip_coefficient}` |
| AdamW moment summaries and realized Δθ | **Done** | `MomentSummary` (per-block `initialized`/`step`/moment norms, before and after); `RealizedUpdateRecord.delta_theta_by_block`; tested in `test_stage2_recorder.py::test_recorder_moments_and_realized_update` |
| Direction norm, near-zero rate, directional derivatives | **Done** | `RealizedUpdateRecord.{direction_norm,near_zero_rate,directional_derivative}`; real values in `metrics.json`'s `per_batch_summaries[*].{direction_norm,near_zero_rate,directional_derivative}` |
| Rollout, token, reward-call, backward, wall-time counts | **Done** | `counters.py::StepCounters` (`rollout_count`, `token_count`, `reward_call_count`, `backward_count`, `wall_time_seconds` per phase, `total_wall_time_seconds`); tested in `test_stage1_mock_model_and_optim.py::test_counters_accumulate`; real values in `metrics.json`'s `per_batch_summaries[*].counters` |

## Execution stages

| Stage | Status | Evidence |
|---|---|---|
| 1. Deterministic toy tests with shared/private modules | **Done** | `tests/instrumentation/test_stage1_primitives.py`, `test_stage1_blocks.py`, `test_stage1_mock_model_and_optim.py` (24 tests) |
| 2. Sequential task batches at one shared-state hash | **Done** | `tests/instrumentation/test_stage2_recorder.py::test_sequential_task_batches_observe_one_shared_state_hash` + `test_check_shared_state_detects_mutation_mid_batch` (protocol violation raises, never silently continues) |
| 3. Prove instrumentation does not alter gradients or optimizer results | **Done** | `src/comppareto/instrumentation/equivalence.py::run_equivalence_check`; `tests/instrumentation/test_stage3_equivalence.py`; real measured errors in `runs/instrumentation-corl-v1/metrics.json` (`2.384185791015625e-07`, `7.058704565299223e-09`), both `<= 1e-6` |
| 4. Integrate a mock CoRL-compatible model | **Done** | `src/comppareto/instrumentation/mock_corl.py::MockCoRLModel`; `tests/instrumentation/test_stage4_integration.py`; driven end-to-end via `experiment.py::run_experiment` |
| 5. Optionally use T710 assets | **Skipped (permitted)** | T710 was `status: running` with no `runs/corl-admission-v1/` at branch creation, reconfirmed still true this session via a live peer session working T710; task brief's own wording makes this optional and conditional on availability — see `reports/T750/first-report.md` |

## Pass/fail gate

> "Instrumented and uninstrumented updates must match within dtype tolerance; ownership must be
> complete; logs must reconstruct the combined gradient and realized update; no silent device
> fallback is allowed."

| Gate bullet | Status | Evidence |
|---|---|---|
| Instrumented vs. uninstrumented match within dtype tolerance | **Pass** | `metrics.json`: `equivalence.gradient_max_abs_error = 2.384185791015625e-07 <= 1e-6`; `equivalence.parameter_update_max_abs_error = 7.058704565299223e-09 <= 1e-6` |
| Ownership complete | **Pass** | `metrics.json`: `ownership.unassigned_trainable_parameters = 0`; enforced via `registry.validate_ownership()` (raises `OwnershipError` on any gap) exercised in `test_stage1_blocks.py` and live in `experiment.py::run_experiment` |
| Logs reconstruct combined gradient and realized update | **Pass** | `metrics.json`: `reconstruction.combined_gradient_pass = true` (error `0.0`), `reconstruction.realized_update_pass = true` (error `0.0`); `reconstruction.py::check_reconstruction` independently recomputes both purely from logged data |
| No silent device fallback | **Pass** | `device.py::resolve_device` — only `"cpu"` is accepted; any other request (`"cuda:0"`, `"tpu"`, etc.) raises `DeviceUnavailableError` immediately; tested in `test_stage1_primitives.py::test_resolve_device_rejects_unsupported_without_fallback` |

## Resource envelope

> "CPU by default; optional real-model smoke at most 1 H20 GPU-hour; no persistent research
> training."

**Honored.** This entire submission ran CPU-only (`resources.device = "cpu"`, `resources.gpu_hours
= 0.0` in `metrics.json`); no GPU was requested or used; stage 5 (the only stage that could have
used GPU time) was skipped; no training loop of any kind (research or otherwise) was started.

## Required deliverables checklist

| Deliverable | Path |
|---|---|
| Resolved config | `configs/instrumentation/corl/resolved-config.yaml` |
| First report | `reports/T750/first-report.md` |
| Result summary | `reports/T750/result-summary.md` |
| Claim check | `reports/T750/claim-check.md` (this file) |
| Failure ledger | `reports/T750/failure-ledger.md` |
| Formal run manifest | `runs/instrumentation-corl-v1/manifest.json` (`status: pass`) |
| Formal run metrics | `runs/instrumentation-corl-v1/metrics.json` |
| Formal run notes | `runs/instrumentation-corl-v1/notes.md` |

## Conclusion

**Supports gate.** Every pass/fail-gate bullet is satisfied with real measured numbers, not
assumptions. `runs/instrumentation-corl-v1/manifest.json` records `status: pass` with 3
hash/byte-addressed artifacts. Stage 5 was permissibly skipped per the task brief's own conditional
wording, and this is recorded in three independent places (the task file's review history, this
claim check, and `runs/instrumentation-corl-v1/notes.md`) for consistency.
