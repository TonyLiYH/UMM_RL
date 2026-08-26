---
id: T140
title: CG, unroll, diagonal, and low-rank approximation error curves
parent: T100
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T140-approximation-error
depends_on: [T110, T130]
blocks: []
allowed_paths: ["src/comppareto/", "tests/", "configs/t1b/", "runs/t1b-approx-*/", "reports/T140/", "tasks/T140-approximation-error.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T140: Approximation error

## Research claim

Scalable compensation estimators are useful only if their error and cost are measured against accepted references.

## Objective

Compare CG, unroll-1/3/5, diagonal, and low-rank estimators across accepted random task families.

## Dependencies and inputs

Accepted T110 and T130 outputs plus the T120 reference when available.

## Allowed changes

Approximation code, tests, configs, runs, report, and this task file.

## Frozen protocol

Report error versus compute without selecting only favorable dimensions or coupling ranks.

## Execution stages

Implement estimators; calibrate fixed tolerances; run full matrix; plot error/cost curves.

## Pass/fail gate

At least one scalable estimator meets the frozen accuracy/cost target without unsafe indefinite behavior.

## First report

Publish estimator definitions, matrix, budgets, metrics, and CPU/GPU estimate before execution.

## Required deliverables

Code, tests, manifests, curves, tabular errors, costs, summary, and failure ledger.

## Artifact and provenance requirements

Every curve point maps to method, seed, dimension, rank, iteration budget, and revision.

## Failure and retry rules

No estimator-specific tolerance changes after results are visible.

## Successor opening

Acceptance contributes to T100.

## Review history

- 2026-08-26 — Planned; T110 and T130 are not accepted.

