---
id: T120
title: Independent KKT and direct-solver reference
parent: T100
status: awaiting_review
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T120-independent-kkt-reference
depends_on: []
blocks: [T150]
allowed_paths: ["src/comppareto/", "tests/", "configs/t1b/", "runs/t1b-kkt-*/", "reports/T120/", "tasks/T120-independent-kkt-reference.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-09-28
---

# T120: Independent KKT reference

## Research claim

Schur elimination, trust-region attainable gain, and negotiation solutions require an independently implemented reference.

## Objective

Implement direct joint KKT/reference solves without calling the production Schur or negotiation functions.

## Dependencies and inputs

T1a problem definitions and mathematical constraints.

## Allowed changes

Reference solver, tests, configs, runs, report, and this task file.

## Frozen protocol

Reference code must not import the production solution functions under test.

## Execution stages

Direct joint solve; KKT residual calculation; comparison suite; failure-case capture.

## Pass/fail gate

Parameter, objective, feasibility, and KKT residual thresholds pass on fixed and random convex cases.

## First report

Return solver choice, equations, independence argument, tolerances, and CPU estimate.

## Required deliverables

Reference implementation, tests, manifest, residual tables, summary, and failure ledger.

## Artifact and provenance requirements

Record solver/library version, config hash, source revision, and every case identifier.

## Failure and retry rules

Solver non-convergence remains a result and cannot be silently replaced.

## Successor opening

Acceptance contributes to T150 and T100.

## Review history

- 2026-08-26 — Authorized for remote execution; no result submitted.
- 2026-09-28 — Remote execution complete. Submitted for review.
  - Deliverables: `src/comppareto/kkt_reference.py` (684 lines, 6 solve paths),
    `src/comppareto/kkt_compare.py` (comparison suite), `tests/test_kkt_reference.py`
    (29 tests), `tests/test_kkt_compare.py` (8 tests), `configs/t1b/kkt-reference-suite.json`,
    `runs/t1b-kkt-20260928-suite/` (config.yaml, notes.md, summary.json).
  - Suite result (seed=20260928): 111/111 cases passed, 0 failures, 2.15 s.
  - Max residuals vs. thresholds: private_response_param 1.11e-15/1e-08,
    schur_param 1.78e-15/1e-08, exact_rational_param 8.88e-16/1e-08,
    trust_region_eigen_param 1.78e-15/1e-06, trust_region_blackbox_param 4.27e-06/5e-03,
    negotiation_param 9.90e-06/5e-03, negotiation_kkt_complementarity 2.56e-05/1e-04.
  - All 336 repository tests pass (307 pre-existing + 29 new T120 tests).
  - Frozen protocol preserved: kkt_reference.py has zero import statements pulling in
    comppareto.quadratic (verified by AST analysis in test_module_does_not_import_production).
  - Two non-trivial bugs found and fixed: (1) late-binding closure in negotiation
    constraint loop (stationarity residual 2.6 → 1e-13 after fix); (2) non-PSD Schur
    complement from independently-sampled random test generator (fixed by joint-PD-block
    sampling).
  - Validator note: `tasks/contracts/T120.acceptance.yaml` does not exist and is outside
    allowed_paths; `validate_task_submission.sh T120` exits 1 solely due to this missing
    contract. This is an upstream sequencing gap (all existing contracts were added on the
    local-review track, not by agent branches) — it is not a deficiency in the deliverables.
- 2026-09-29 — Local review: the sequencing gap above is confirmed by direct evidence and is
  now recorded as an infrastructure defect. The acceptance-contract mechanism was introduced
  on 2026-08-28 (`d7106f5 feat: enforce task submission acceptance gates`); this task file was
  created on 2026-08-26, two days earlier, so the task predates the mechanism and was never
  backfilled. `tasks/contracts/` lies outside this task's `allowed_paths`, so the contract
  cannot be authored on the executor branch; per AGENTS.md the local planning/review side owns
  Gates, and the executor only *reads* the contract from authorized `main`.
  - **Escalated consequence.** Local review has now accepted the T130 change to
    `src/comppareto/repo_state/submission_cli.py`, under which a missing contract exits 0
    instead of 1. Once that change reaches `main`, this task will satisfy
    `scripts/validate_task_submission.sh` with **zero substantive checks**: required_files,
    metrics, forbidden_claims and commands are all skipped, leaving only branch-name,
    clean-tree and ancestry checks. The gate result would be vacuous — so the backfill is now
    strictly more urgent than when the executor flagged it.
  - **Required action (local review side).** Backfill `tasks/contracts/T120.acceptance.yaml`
    derived from this task's own Frozen protocol / Pass/fail gate / Required deliverables
    sections — explicitly *not* from the observed 111/111 result. Required before this task
    may be set to `accepted`.

