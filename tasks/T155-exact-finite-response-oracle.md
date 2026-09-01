---
id: T155
title: Exact finite-response oracle benchmark
parent: T100
status: accepted
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T155-exact-oracle
depends_on: []
blocks: [T160]
allowed_paths: ["docs/theory/oracle-spec.md", "src/comppareto/oracle/", "tests/oracle/", "configs/oracle/", "runs/oracle-*/", "reports/T155/", "tasks/T155-exact-finite-response-oracle.md"]
source_revision: "432ea002fb71a66bbecc22e2a1d59b74777a1769"
created_at: 2026-08-27
updated_at: 2026-09-01
---

# T155: Exact finite-response oracle benchmark

## Research claim

A small structured shared/private multi-task system can provide exact finite-response trajectories, hypergradients, realized loss changes, and high-accuracy Pareto references without claiming to simulate the full behavior of a unified multimodal model.

## Objective

Specify and implement a structured linear-quadratic oracle with controllable task-block overlap, private compensation, conflict geometry, stochastic noise, and finite SGD/momentum response. Cross-check analytic results against automatic differentiation, finite differences, and direct numerical evaluation.

## Dependencies and inputs

The corrected finite-response semantics in `docs/theory/formulation.md`, the theory audit in `docs/theory/2026-08-27-theory-breakthrough-audit.md`, and the existing quadratic utilities. T110–T130 may run in parallel and are not prerequisites for the first oracle implementation.

## Allowed changes

Only the declared oracle specification, code, tests, configurations, run evidence, report, and this task file.

## Frozen protocol

The oracle is a mechanism-validation benchmark, not a surrogate claim about real UMM behavior. Randomization is structured through declared ranges for overlap, gradient angle, scale ratio, curvature spectrum, coupling rank, noise covariance, and optimizer memory. The primary operational target is rerun-response \(F_i^{K,\mathrm{rerun}}\); commit-response quantities are separate labeled references.

The first implementation covers exact SGD and momentum. A diagonal Adam-like state is a stretch deliverable and may start only after the SGD and momentum gates pass.

## Execution stages

1. Publish the complete mathematical specification and resolved parameter ranges.
2. Implement deterministic task/block selectors and structured data generation.
3. Implement exact \(K\)-step SGD finite response and analytic sensitivity.
4. Implement exact augmented-state momentum response and sensitivity.
5. Cross-check analytic sensitivities against full automatic differentiation and central finite differences.
6. Compute exact realized rerun-response loss changes for candidate shared directions.
7. Emit a seeded baseline dataset and oracle manifest for T160/T170.
8. Add the Adam-like extension only if the core gate passes within the declared CPU budget.

## Minimum model specification

- two to eight tasks;
- four to sixty-four global parameter blocks;
- task-private parameter dimensions from two to thirty-two;
- disjoint, partial, star, chain, random-sparse, and full-overlap selectors;
- train and disjoint meta/evaluation matrices;
- quadratic losses with explicit private regularization;
- controllable shared-gradient cosine, gradient scale ratio, curvature condition number, coupling rank, Gaussian and block-correlated noise;
- \(K\in\{1,3,5,10\}\) for SGD and momentum;
- stable and deliberately unstable response configurations.

The first report may narrow dimensions for runtime, but it must preserve every mechanism above and state the reason for each final range.

## Pass/fail gate

For every accepted stable seeded case:

- analytic finite-response state matches the independently unrolled state with relative error at most \(10^{-10}\);
- analytic hypergradient matches full automatic differentiation with relative error at most \(10^{-9}\) when the reference norm exceeds \(10^{-10}\), or absolute error at most \(10^{-11}\) near zero;
- central finite-difference directional derivatives match within a preregistered step-size stability envelope;
- direct rerun-response loss changes match the oracle evaluation;
- selectors satisfy the single-lift and no-duplicate-coordinate contract;
- all failed or unstable seeds remain in the ledger;
- full-overlap and disjoint regimes behave as declared boundary controls.

A systematic mismatch, an unexplained seed-dependent failure, or a generated family that cannot realize the declared conflict and coupling ranges blocks acceptance.

## First report

Before implementation, publish:

- exact equations for train and meta losses;
- state definitions for SGD and momentum;
- rerun/commit response definitions;
- analytic state and sensitivity recurrences;
- selector and graph-family construction;
- parameter ranges and seed policy;
- independent autodiff, finite-difference, and direct-evaluation references;
- numerical tolerances and near-zero handling;
- expected CPU time, memory, and output size;
- expected handoff artifacts for T160 and T170.

## Required deliverables

Oracle specification, typed implementation, analytic and numerical references, unit/property tests, resolved configurations, a deterministic run manifest, per-seed result table, boundary and failure cases, summary report, and failure ledger.

## Artifact and provenance requirements

Every generated case records graph family, selector hashes, dimensions, seed, optimizer, horizon, spectra, coupling rank, gradient angle, noise model, source revision, configuration hash, exact state/hypergradient/loss-change references, and output hash.

## Failure and retry rules

Do not remove hard seeds, tune generation ranges after inspecting method performance, or describe the oracle as a realistic UMM simulator. Numerical tolerance changes require local review. Infrastructure retries preserve the same configuration and seed.

## Automated submission gate

Before setting `awaiting_review`, run:

```bash
bash scripts/validate_task_submission.sh T155
```

The machine-readable contract is `tasks/contracts/T155.acceptance.yaml`. A
formal run with `status: fail`, any stable-case failure, any Pareto-reference
failure, any independent high-precision-reference failure, a full-suite test
failure, or an unauthorized changed path blocks submission.

## Successor opening

Accepted T155 contributes the exact benchmark required by T160. T160 retains its other declared dependencies.

## Review history

- 2026-08-27 — Authorized for CPU execution as the exact-oracle entry point.
- 2026-08-27 — Remote executor set status to `running` on `agent/T155-exact-oracle`; first report drafted before implementation.
- 2026-08-27 — Remote executor implemented the oracle core and sweep infrastructure (`src/comppareto/oracle/`, `configs/oracle/baseline.yaml`), ran 52 unit tests (all passing) and the resolved 288-case baseline sweep (`runs/oracle-20260827-baseline/`, 286/288 passed). Submitted evidence under `reports/T155/` (`result-summary.md`, `claim-check.md`, `failure-ledger.md`) and set status to `awaiting_review`. Two unresolved `loss_change`-tolerance failures (case_index 41, 287) are flagged in `reports/T155/failure-ledger.md` for local-reviewer decision; no tolerance was changed and no seed was dropped.
- 2026-08-28 — Local review of `e1cb034` requested revision. Blocking items:
  invalid formal run manifest, missing serialized detailed references, missing
  Pareto reference, inaccurate autodiff terminology, two unresolved unstable
  loss-change checks, and an out-of-scope `CHANGELOG.md` edit. Full review:
  `reports/T155/local-review.md`.
- 2026-08-28 — Remote executor addressed local-review R1-R6 and set status to
  `awaiting_review` on `agent/T155-exact-oracle` at `b8a1f6b`. R1:
  `manifest.json` is now a schema-valid envelope object per
  `schemas/run-manifest.schema.json`; the per-case array moved to
  `case-records.json`. R2: `case_record()` now serializes each task's full
  `detail` (state/momentum trajectories, per-step `J_k`/`B_k`, sensitivity
  trajectories, exact local gradient, `Q_i^K`, selector/case identifiers),
  independently re-verified in `tests/oracle/test_case.py`. R3: added
  `src/comppareto/oracle/pareto.py`, an independent exact
  active-set-enumeration common-descent/Pareto reference over the tasks' real
  lifted gradients (not random probe directions), cross-checked against an
  independent Frank-Wolfe solver; a scale-dependent tolerance bug found in
  this new code during regeneration (spurious rejection of the true optimum
  on large-Gram-magnitude case_index 47) was fixed by selecting the
  minimum-objective lambda-feasible candidate, provably the exact optimum for
  this convex QP. R4: renamed "automatic differentiation" claims to
  "independently implemented reverse-mode differentiation" throughout (no AD
  library is a dependency). R5: added `src/comppareto/oracle/highprecision.py`
  for a higher-precision/conditioning-aware recheck of case_index 41/287
  without relaxing the frozen 1e-9 tolerance; both failures reproduce
  bit-for-bit at extended precision. R6: reverted the out-of-scope
  `CHANGELOG.md` edit. Regenerated the full 288-case sweep on the clean
  `b8a1f6b` revision (`run_kind: formal`, `dirty: false`); 286/288 still pass
  (same 2 documented failures as before). `tests/oracle/` 100/100 pass; full
  repo suite 129/130 pass, with the 1 failure
  (`tests/repo_state/test_cli.py::test_cli_validates_repository`) being a
  pre-existing, out-of-scope test-assertion staleness unrelated to this
  remediation (see `reports/T155/result-summary.md`). Evidence:
  `reports/T155/result-summary.md`, `claim-check.md`,
  `runs/oracle-20260827-baseline/notes.md`.
- 2026-08-28 — Second local review of `0ba8bf8` retained
  `revision_needed`: the long-double tests fail and do not establish a
  cancellation-only explanation; the Frank–Wolfe Pareto cross-check has
  material scale-normalized errors and is not included in the run Gate. Merge
  current `origin/main` and complete R7–R9 in `reports/T155/local-review.md`.
- 2026-08-28 — Remote executor merged current `origin/main` and addressed
  local-review R7-R9 on `agent/T155-exact-oracle` at `432ea00`. R7: replaced
  the platform-dependent longdouble-only extended-precision check with a
  genuinely independent, platform-independent `decimal.Decimal` (precision=100)
  literal-recurrence reconstruction of the full affine-transition/sensitivity/
  quadratic-model/loss-change chain (`src/comppareto/oracle/highprecision.py`);
  both known float64 tolerance failures (case_index 41/task 4, case_index
  287/task 1) remain in `failure_ledger.json` unmodified and are now confirmed
  `pure_cancellation: true` (Decimal relative error 2.37e-13 vs. float64's
  2.73e-9 for case 41; 4.72e-16 vs. 1.29e-8 for case 287), persisted in the new
  `high_precision_recheck.json` artifact. R8: added `min_norm_point_scipy_qp`
  (SciPy SLSQP), a genuinely independent constrained-QP Pareto/min-norm-point
  reference, alongside the exact active-set enumeration; Frank-Wolfe is
  retained as an optional diagnostic only. Fixed an SLSQP false-convergence
  bug (unnormalized objective on this oracle's up-to-1e13 Gram-matrix scale
  caused `ftol` to falsely report convergence) by normalizing by
  `scale = max(diag(Gram), 1)`. Defined five preregistered scale-aware
  thresholds (all 1e-6) evaluated as `pareto_reference.independent_check`,
  now gating `CaseResult.all_passed`; zero independent-check failures across
  the full 288-case sweep. R9: wired both checks into `sweep.py`'s
  summary/manifest generation (`stable_failed_cases`, `pareto_failed_cases`,
  `high_precision_failed_cases` in `summary.json`); per the task's exact
  pass/fail gate wording ("for every accepted **stable** seeded case ... all
  failed or unstable seeds remain in the ledger"), `manifest.json status` is
  now `pass` (all three new fields are 0; the raw `failed_cases: 2` count is
  unchanged and both known unstable-regime failures remain in the ledger).
  Regenerated the full 288-case sweep on the clean `432ea00` revision
  (`run_kind: formal`, `dirty: false`, `status: pass`). `tests/oracle/`
  111/111 pass; full repo suite 156/156 pass (the previously-flagged
  `test_cli_validates_repository` staleness no longer reproduces on the
  merged tree). Evidence: `reports/T155/result-summary.md`, `claim-check.md`,
  `failure-ledger.md`, `runs/oracle-20260827-baseline/notes.md`. Set status
  back to `awaiting_review`.
- 2026-09-01 — Integrated verification passed on `main`: task/run validation,
  156 tests, local artifact hashes, stable-case Gate, Decimal high-precision
  checks, and independent Pareto QP checks. Accepted by: local-research-agent.
