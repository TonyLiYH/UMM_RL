---
id: T170
title: Graph-localized robust descent and resource allocation
parent: T100
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T170-overlap-robust-certificate
depends_on: [T150, T160]
blocks: [T300]
allowed_paths: ["docs/theory/", "src/comppareto/", "tests/", "configs/t1c/", "runs/t1c-robust-*/", "reports/T170/", "tasks/T170-overlap-robust-certificate.md"]
source_revision: "c481c0ee25245ebace0d6d50d881145a972046cf"
created_at: 2026-08-27
updated_at: 2026-08-27
---

# T170: Graph-localized robust descent and resource allocation

## Research claim

Task-block incidence can localize uncertainty in finite-response hypergradients and support a computable simultaneous-descent certificate whose worst width can be reduced by principled sample and differentiation-compute allocation.

## Objective

Prove and validate a joint high-probability finite-step descent certificate using task-block error radii, then solve the minimax resource-allocation problem for reducing the widest normalized certificate.

## Dependencies and inputs

Accepted negotiation diagnostics from T150 and accepted finite-horizon posterior bounds from T160.

## Allowed changes

Theory documents, robust-certificate and allocation code, tests, T1c configurations and manifests, reports, and this task file only.

## Frozen protocol

Use a block-diagonal reference metric. The certificate must remain valid for a direction selected from estimated gradients, using independent certification data, cross-fitting, or a uniform confidence set. Parameter overlap is not assumed to be a stochastic dependency graph. Compare task-query and edge/block-query oracle models separately.

## Execution stages

1. Define the joint confidence event and task-block dual norms.
2. Prove the simultaneous finite-step descent condition.
3. Implement global, support-local, covariance-aware, and empirical-Bernstein radii.
4. Formulate the minimax certificate-width allocation problem and verify its KKT solution against an independent convex solver.
5. Run disjoint, chain, star, grid, random-sparse, and full-overlap families.
6. Measure coverage, false-safe rate, certified-step recall, width, cost, and allocation regret.

## Pass/fail gate

The localized certificate must satisfy preregistered simultaneous coverage, keep false-safe rate below the declared level, improve sample or differentiation cost over the global bound on sparse families, lose that advantage under full overlap, and produce an allocation close to the exact oracle. Otherwise the graph-localized result is demoted or rejected.

## First report

Publish the probability model, covariance assumptions, confidence construction, graph families, oracle semantics, optimization program, rounding rule, baselines, thresholds, and expected CPU cost.

## Required deliverables

Proof note, certificate implementation, independent allocation solver, tests, frozen configs, manifests, coverage and allocation-regret plots, failure ledger, and a novelty/claim-check report.

## Artifact and provenance requirements

Every result maps to graph, incidence matrix, covariance model, task/block dimensions, seed, budget, confidence level, source revision, configuration hash, and solver residual.

## Failure and retry rules

Do not infer stochastic independence from parameter disjointness. Do not replace simultaneous coverage with per-task marginal coverage. Do not count parallel speedup as reduced total oracle calls. Any change to the confidence construction or query model after results are visible requires local review.

## Successor opening

Accepted T170 contributes to T100 and is required before T300 can be promoted to ready.

## Review history

- 2026-08-27 — Planned from the theory breakthrough audit; dependencies remain incomplete.
