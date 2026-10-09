---
id: T712
title: First complete corrected CoRL training run
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T712-corl-complete-training
depends_on: [T711]
blocks: [T725, T755]
allowed_paths: ["tasks/T712-corl-complete-training.md", "configs/corl/complete-training-v1/", "runs/corl-complete-training-v1/", "reports/T712/", "src/comppareto/adapters/corl/", "tests/adapters/corl/"]
source_revision: "e001548976ede8c8dd9805e5602fc2ab9ac6274b"
created_at: 2026-10-09
updated_at: 2026-10-09
---

# T712: First complete corrected CoRL training run

## Objective

Establish that the T711 corrected CoRL protocol can finish a declared
understanding-generation training horizon with valid held-out evaluation,
authorized updates, and reproducible checkpoint restoration. This is an
engineering and learning-headroom run, not a paper-result reproduction.

## Entry and protocol freeze

T711 must be locally accepted. Before GPU execution, commit a resolved manifest
pinning the base checkpoint and hashes, corrected-protocol revision, one seed,
grouping and sampling settings, parameter mask, `x2x_rft_22k` train/dev record
IDs and split rule, reward model and evaluator revisions, disjoint frozen U/G
dev inputs, checkpoint schedule, stopping rule, and numerical tolerances.
The dev set must never supply training rewards or tuning feedback. Keep the
paper-described, upstream-executable, corrected, and reduced-run identities
separate in the report.

## Execution

1. Calibrate throughput, reward variation, evaluator success, and checkpoint
   size on a short declared window. Record a cost projection before the main
   run. Do not silently change the manifest after seeing dev metrics.
2. Complete the frozen horizon, bounded above by 128 update windows. Save and
   evaluate base, middle, and final snapshots on both U and G dev suites.
3. Continue once uninterrupted and once from the middle checkpoint with the
   same subsequent batches. Compare U/G log probabilities, model weights,
   optimizer moments, scheduler, RNG, and next-step result at the frozen
   tolerance. Report any nondeterministic operations explicitly.
4. Log reward distributions, usable and rejected groups, effective U/G tokens,
   rollouts, reward calls, backward calls, authorized parameter deltas, peak
   VRAM, elapsed time, GPU-hours, and checkpoint bytes.

## Acceptance and stop gate

Pass requires a complete finite run at the declared horizon, valid U and G
held-out metrics at each snapshot, no update outside the frozen mask, and
resume equivalence within the preregistered tolerance. Improvement is an
observed outcome, not a pass requirement. If fewer than 20% of groups have
usable reward variation, either evaluator fails, or the calibrated full run
exceeds the resource cap, stop the main run and diagnose the failure.

## Resource envelope and authorization

Planning ceiling: one H20 and eight H20-equivalent GPU-hours, including
calibration, evaluation, retries, and checkpointing; at most 128 windows.
Reserve local storage for at least three measured full checkpoints including
optimizer state. This ceiling is a proposed task bound, not authorization to
start: the local review side must accept T711, freeze this task's manifest and
resume tolerance, and update the programme GPU budget before marking T712
`ready`.

## First report and deliverables

The first report must include the resolved manifest, split leakage checks,
calibration and projected cost, first finite updates, evaluator validity,
checkpoint size, and any blocker before the full horizon. Deliver the runnable
configuration, command/log manifest, metrics and curves, checkpoint hashes,
resume comparison, resource ledger, and a claim check against each acceptance
item under `reports/T712/` and `runs/corl-complete-training-v1/`.
