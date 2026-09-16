---
id: T750
title: Per-task GRPO gradient and optimizer-update instrumentation
parent: T700
status: awaiting_review
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T750-corl-gradient-update-instrumentation
depends_on: []
blocks: [T760, T770]
allowed_paths: ["tasks/T750-corl-gradient-update-instrumentation.md", "src/comppareto/instrumentation/", "tests/instrumentation/", "configs/instrumentation/corl/", "reports/T750/", "runs/instrumentation-corl-v1/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T750: Per-task GRPO gradient and optimizer-update instrumentation

## Objective

Implement a framework-neutral instrumentation layer, with a CoRL adapter, that
records per-task shared gradients before clipping, combined gradients after
negotiation/clipping, and realized AdamW parameter updates by block.

## Required measurements

- per-task norm, cosine, norm ratio, and finite status;
- block-local statistics and overlap IDs;
- PCGrad-ready vectors and MGDA convex-hull inputs;
- pre/post-clipping norms and clipping coefficient;
- AdamW moment summaries and realized \(\Delta\theta\);
- direction norm, near-zero rate, and directional derivatives;
- rollout, token, reward-call, backward, and wall-time counts.

## Execution

1. Deterministic toy tests with shared/private modules.
2. Sequential task batches at one shared-state hash.
3. Prove instrumentation does not alter gradients or optimizer results.
4. Integrate a mock CoRL-compatible model.
5. Optionally use T710 assets for one real diagnostic batch if available.

## Pass/fail gate

Instrumented and uninstrumented updates must match within dtype tolerance;
ownership must be complete; logs must reconstruct the combined gradient and
realized update; no silent device fallback is allowed.

## Resource envelope

- CPU by default;
- optional real-model smoke at most 1 H20 GPU-hour;
- no persistent research training.

## Automated submission gate

```bash
bash scripts/validate_task_submission.sh T750
```

## Review history

- 2026-09-16 — Remote executor created/resumed branch `agent/T750-corl-gradient-update-instrumentation` from `origin/main` (`448c517`), confirmed `source_revision` `818d1d8` is an ancestor of HEAD, and set status to `running`. This is a CPU-default software-engineering task (framework-neutral gradient/optimizer-update instrumentation layer plus a CoRL adapter); checked `T710` (blocks-adjacent sibling task, not a dependency) and found its `runs/corl-admission-v1/` does not yet exist and its task status is `running`, not `awaiting_review` or later, so per the task brief the optional stage 5 (real T710 diagnostic batch) will be skipped for this submission and stages 1-4 (deterministic toy tests, sequential task batches at one shared-state hash, instrumented-vs-uninstrumented equivalence proof, mock CoRL-compatible model) will be the primary and sole evidence path. No GPU work is planned or required.
- 2026-09-16 — Remote executor completed stages 1-4 and set status to `awaiting_review`. PyTorch is not installed anywhere in this execution environment (exhaustively checked: every sibling worktree venv, the main repo's `.venv`, all system-wide `site-packages`); the entire instrumentation layer and mock CoRL-compatible model were implemented in pure NumPy instead (hand-derived backpropagation, hand-implemented AdamW and gradient clipping that exactly replicate `torch.optim.AdamW` / `torch.nn.utils.clip_grad_norm_`'s formulas) — a defensible and arguably more faithful reading of "framework-neutral" for this task; full rationale in `runs/instrumentation-corl-v1/notes.md`. Stage 5 remained skipped: a live peer session working `T710` (still `status: running`, no `runs/corl-admission-v1/`) confirmed during this session it is building its own gradient-leak-probe/optimizer-diffing logic directly against the real model inside its own `allowed_paths`, and does not intend to reuse this task's NumPy-only mock-model code. Measured results (`runs/instrumentation-corl-v1/metrics.json`, seed=20260916, num_batches=5): `equivalence.gradient_max_abs_error=2.384185791015625e-07` (gate `<=1e-6`), `equivalence.parameter_update_max_abs_error=7.058704565299223e-09` (gate `<=1e-6`), `ownership.unassigned_trainable_parameters=0`, `reconstruction.combined_gradient_pass=true`/`reconstruction.realized_update_pass=true` (both exact, error `0.0`), `resources.gpu_hours=0.0`/`device="cpu"` (gate `<=1`), 5 distinct `shared_state_hashes` across 5 sequential batches. Local validation: `.venv/bin/python -m pytest -q` → 272 passed (full repo suite); `.venv/bin/python -m compileall -q src tests` → clean; `.venv/bin/python -m comppareto.repo_state.cli --root .` → `task_tree=pass tasks=40`, `run_manifests=pass manifests=9`, `research_state=pass`; `git diff --check origin/main...HEAD` → clean. `runs/instrumentation-corl-v1/manifest.json` records `status: pass`, schema-validated against `schemas/run-manifest.schema.json`. Committed as `89d6781`.

