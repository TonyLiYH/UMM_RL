---
id: T216
title: Show-o2 compute-matched alternating-protocol diagnostic
parent: T200
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T216-showo2-alternating-protocol-diagnostic
depends_on: [T210]
blocks: [T300, T310]
allowed_paths: ["tasks/T216-showo2-alternating-protocol-diagnostic.md", "configs/alternating/showo2/", "runs/alternating-showo2-v1/", "reports/T216/", "src/comppareto/adapters/showo2_alternating/", "tests/adapters/showo2_alternating/"]
source_revision: "2358267c14dddc3754e7a80e9b681308c6bcd0f7"
created_at: 2026-09-15
updated_at: 2026-09-15
---

# T216: Show-o2 compute-matched alternating-protocol diagnostic

## Research claim

A reversible Show-o2 diagnostic can determine whether simple
shared-then-private alternating updates already capture useful private
compensation, and whether private-then-shared commit gradients improve the
choice of shared direction under matched data and private-step budgets.

This task tests feasibility and attribution. It does not claim final model
quality, convergence speedup, or authorization for persistent joint training.

## Objective

On one admitted understanding path and one admitted generation path:

1. implement simultaneous, shared-then-private (SP), private-only control, and
   private-then-shared commit (PS) protocols;
2. generate shared candidate directions with raw SUM, normalized SUM, PCGrad,
   and two-task MGDA;
3. measure compute-matched controlled post-adaptation loss changes;
4. determine whether raw zero/near-zero or conflicting directions become
   jointly useful after private adaptation;
5. record gradient evaluations, data use, wall clock, peak memory, and complete
   rollback evidence.

## Dependencies and inputs

- accepted T210 Show-o2 source/checkpoint and local-SSD execution layout;
- mathematical definitions in `docs/math/`;
- no dependency on T215 code or its failed full-rerun implementation.

The executor may independently reimplement the small reusable snapshot and
gradient utilities needed inside the allowed T216 paths. Do not merge or
silently depend on an unaccepted task branch.

## Frozen protocol

- Use the exact accepted T210 model revision and checkpoint.
- Use the T210 shared/private block registry to select one declared shared
  subspace and one private subspace per task.
- Begin with \(K=1\); run \(K=3\) only if rollback, finiteness, and resource
  gates pass at \(K=1\).
- Use the same initial parameter state, optimizer state, batches, RNG state,
  data order, and private-step budget for treatment and control branches.
- All shared and private updates are virtual and fully rolled back.
- No full AdamW trajectory differentiation and no persistent joint training.
- Run model assets and caches from verified local SSD.

## Protocols

### P0: simultaneous raw baseline

Compute shared and private gradients at the original state. Apply both virtual
updates without letting the private update observe the new shared state.

### P1: private-only control

Keep shared parameters fixed and execute exactly \(K\) private updates:

\[
s_{i}^{0,K}=A_i^K(P_i\theta;s_i).
\]

### P2: shared-then-private (SP)

Construct \(d\) from raw shared gradients, apply
\(\theta'=\theta+\eta d\), freeze \(\theta'\), then execute exactly \(K\)
private steps from the original private snapshot:

\[
s_i^{d,K}=A_i^K(P_i\theta';s_i).
\]

The controlled treatment effect is

\[
\Delta_i^{controlled}(d)
=
L_i(P_i\theta',\pi_\phi s_i^{d,K})
-
L_i(P_i\theta,\pi_\phi s_i^{0,K}).
\]

### P3: private-then-shared commit (PS)

Execute \(K\) virtual private steps at the original shared state, compute the
post-adaptation stop-gradient, construct \(d\), and evaluate the same
compute-matched control/treatment branches. The virtual pre-adaptation state
must not persist across candidate directions.

## Shared candidate directions

At minimum:

- raw SUM;
- raw per-task normalized SUM;
- raw PCGrad with a frozen task order and a separately declared reversed-order
  sensitivity check;
- exact two-task MGDA convex-combination solution.

For PS, repeat SUM and MGDA with commit gradients. PCGrad/normalized SUM for PS
are optional only after mandatory rows pass.

All directions must be normalized to a declared common trust-region metric
before comparing step scales.

## Step-scale and attribution design

- Predeclare at least three symmetric logarithmic step scales around a pilot
  scale selected without observing treatment outcomes.
- Evaluate treatment and control with common random numbers.
- Report absolute loss change, controlled loss change, and normalized
  controlled change for both tasks.
- Report whether each direction produces:
  - both tasks improved;
  - one improved and one degraded;
  - neither improved;
  - an indeterminate result due to numerical or sampling noise.
- Do not call a direction beneficial merely because its uncorrected loss after
  extra private training is lower.

## Execution stages

1. Commit and push a first report before GPU execution.
2. Implement exact snapshot/restore for parameters, optimizer state, RNG,
   buffers, gradient scaler, and data order.
3. Unit-test protocol ordering on a deterministic toy shared/private model.
4. Run local-SSD preflight and artifact verification.
5. Execute \(K=1\) P0--P3 mandatory rows on both task paths.
6. Verify complete rollback and finite losses/gradients.
7. Run \(K=3\) only if the \(K=1\) gate passes.
8. Emit row-level results, attribution table, resource table, and failure
   ledger.

## Pass/fail gate

The task passes feasibility only if:

- every mandatory treatment row has a compute-matched private-only control;
- all mandatory P0--P3 rows execute on both task paths;
- parameter tensors restore within declared dtype tolerance, while counters,
  RNG, data order, and discrete state restore exactly;
- all mandatory reported losses and shared directions are finite;
- repeated same-seed executions match declared deterministic tolerances;
- controlled effects, not uncorrected post-private losses, drive conclusions;
- total execution remains within the resource envelope;
- persistent updates equal zero.

A finding that SP does not improve over the control is a valid scientific
result and does not by itself fail the task. Missing controls, rollback
failure, nonfinite mandatory rows, or selective omission do fail it.

## First report

Before GPU execution, report:

- exact model/checkpoint revision and local-SSD paths;
- selected shared/private module paths and parameter counts;
- optimizer state inventory;
- equations and pseudocode for P0--P3;
- direction normalization and step-scale grid;
- task batches, seeds, RNG coupling, and data-order protocol;
- expected gradient-evaluation count, memory, runtime, and GPU hours;
- exact commands and artifact paths.

## Required deliverables

- `reports/T216/first-report.md`
- `reports/T216/result-summary.md`
- `reports/T216/claim-check.md`
- `reports/T216/failure-ledger.md`
- resolved config and storage/artifact verification;
- run manifest, metrics, and notes;
- toy protocol-ordering and rollback tests;
- controlled-effect table by protocol, negotiator, task, \(K\), and step scale;
- resource accounting by row.

## Resource envelope

- at most two H20 GPUs;
- at most eight H20-equivalent GPU-hours;
- at most 24 wall-clock hours excluding queue time;
- \(K=3\) conditional on the \(K=1\) gate;
- no full-backbone unroll and no persistent parameter update.

## Automated submission gate

Before setting `awaiting_review`, run:

```bash
bash scripts/validate_task_submission.sh T216
```

## Review policy

Only the local review side can mark this task accepted. Report negative and
indeterminate rows without replacement or post-hoc tuning.

