---
id: T120
title: Independent KKT and direct-solver reference
parent: T100
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T120-independent-kkt-reference
depends_on: []
blocks: [T150]
allowed_paths: ["src/comppareto/", "tests/", "configs/t1b/", "runs/t1b-kkt-*/", "reports/T120/", "tasks/T120-independent-kkt-reference.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
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

