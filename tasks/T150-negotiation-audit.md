---
id: T150
title: Negotiation feasibility, KKT, and independent-reference audit
parent: T100
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T150-negotiation-audit
depends_on: [T120]
blocks: []
allowed_paths: ["src/comppareto/", "tests/", "configs/t1b/", "runs/t1b-negotiation-*/", "reports/T150/", "tasks/T150-negotiation-audit.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T150: Negotiation audit

## Research claim

The retained-gain solver must satisfy feasibility, stationarity, and independent-reference checks beyond an optimizer `success` flag.

## Objective

Add constraint violation, KKT residual, iteration, and independent-grid/convex-reference comparisons.

## Dependencies and inputs

Accepted T120 reference solver and current negotiation implementation.

## Allowed changes

Negotiation diagnostics, tests, configs, manifests, report, and this task file.

## Frozen protocol

Use fixed convex cases, boundary cases, infeasible certificate cases, and independent task rescaling.

## Execution stages

Define diagnostics; implement references; run suite; emit residual and reference-error tables.

## Pass/fail gate

All accepted solutions meet frozen feasibility/KKT/reference thresholds; failures are explicit.

## First report

Return equations, reference strategy, thresholds, and CPU estimate.

## Required deliverables

Diagnostics, tests, manifest, tables, summary, and failure ledger.

## Artifact and provenance requirements

Record optimizer options, iterations, residuals, reference solver, and revision.

## Failure and retry rules

No solver swap after seeing comparative results without local approval.

## Successor opening

Acceptance contributes to T100.

## Review history

- 2026-08-26 — Planned; T120 is not accepted.

