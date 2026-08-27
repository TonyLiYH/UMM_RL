---
id: T160
title: Finite-horizon optimizer-response posterior certificate
parent: T100
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T160-finite-response-certificate
depends_on: [T110, T120, T130, T155]
blocks: [T170]
allowed_paths: ["docs/theory/", "src/comppareto/", "tests/", "configs/t1c/", "runs/t1c-response-*/", "reports/T160/", "tasks/T160-finite-response-certificate.md"]
source_revision: "c481c0ee25245ebace0d6d50d881145a972046cf"
created_at: 2026-08-27
updated_at: 2026-08-27
---

# T160: Finite-horizon optimizer-response posterior certificate

## Research claim

A prescribed finite task-native optimizer response can be differentiated and audited as a finite-horizon state trajectory without assuming that it approximates a globally optimal private response.

## Objective

Formalize rerun-response and commit-response semantics; implement exact finite-horizon sensitivities for linear-quadratic SGD and momentum; derive and test a posteriori hypergradient error bounds from tangent residuals and finite-horizon propagation gains.

## Dependencies and inputs

Accepted overlap families, independent reference solver, indefinite-curvature
behavior from T110–T130, the accepted T155 exact oracle, and the corrected
protocol definitions in `docs/theory/formulation.md`.

## Allowed changes

Theory documents, finite-response oracle code, tests, T1c configurations and manifests, reports, and this task file only.

## Frozen protocol

The operational target is \(F_i^{K,\mathrm{rerun}}\) unless a test explicitly names the commit-response counterfactual. Finite \(K\) is not labeled inner error. Exact unrolling is the reference; every approximation reports its tangent residual, propagated bound, realized error, cost, and all failed seeds.

## Execution stages

1. State the finite-horizon sensitivity recurrence and norm conventions.
2. Import and independently verify the accepted T155 exact references.
3. Implement approximate sensitivities and per-step tangent residuals.
4. Bound or estimate finite-horizon propagation gains.
5. Compare posterior bounds with realized hypergradient error and issue a claim check.

## Pass/fail gate

Across the frozen seeded families, the posterior bound must cover the exact finite-response hypergradient error at the preregistered rate, remain finite on accepted trajectories, detect deliberately unstable trajectories, and be tighter than a single global worst-case bound on at least one nontrivial sparse family. A bound that is valid but too loose to distinguish any candidate estimator is reported as a negative result and does not open T170.

## First report

Publish the exact state definitions, rerun/commit semantics, optimizer equations, norms, residual formulas, propagation-gain estimator, task-family dimensions, seeds, tolerances, and expected CPU cost before implementation.

## Required deliverables

Proof note, exact and approximate trajectory code, unit/property tests, resolved configuration, run manifest, coverage/tightness tables, unstable-trajectory counterexamples, and a claim-check report.

## Artifact and provenance requirements

Every result maps to optimizer, state dimension, horizon, seed, approximation, source revision, config hash, residual sequence, propagation estimate, exact error, and bound.

## Failure and retry rules

Do not replace exact finite-response error with a longer-horizon proxy. Do not drop clipping/switching failures; record them as explicit unsupported or defect cases. Tolerance or norm changes require local review.

## Successor opening

Acceptance opens T170.

## Review history

- 2026-08-27 — Planned from the theory breakthrough audit; dependencies remain incomplete.
