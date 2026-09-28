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

