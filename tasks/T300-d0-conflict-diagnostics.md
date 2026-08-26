---
id: T300
title: D0 compensation-aware conflict diagnostics
parent: T000
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T300-d0-conflict-diagnostics
depends_on: [T100, T210]
blocks: [T400]
allowed_paths: ["src/comppareto/", "tests/", "configs/d0/", "runs/d0-*/", "reports/T300/", "tasks/T3*.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T300: D0 conflict diagnostics

## Research claim

Optimizer-state-aware compensation should predict realized task changes better than raw first- or second-order Taylor models under identical finite private responses.

## Objective

Build the block registry, shared hypergradient cache, predictor comparison, and held-out calibration audit.

## Dependencies and inputs

Accepted T100 and T210, paired semantic batches, and frozen diagnostic splits.

## Allowed changes

Show-o2 diagnostic adapters, logging, prediction metrics, tests, and D0 reports.

## Frozen protocol

Raw cosine is descriptive only; the Gate compares against the strongest raw Taylor predictor.

## Execution stages

T310, T320, T330, then T340.

## Pass/fail gate

Held-out compensation-aware prediction improves over the strongest raw Taylor baseline under equal compute and meets the frozen calibration threshold.

## First report

Publish block partitions, data split unit, cached-gradient schema, estimator budgets, and expected GPU hours.

## Required deliverables

Diagnostic datasets, cached gradients, predictor metrics, calibration plots, and a Gate report.

## Artifact and provenance requirements

Every record maps to checkpoint, optimizer state, batch group, parameter blocks, and source revision.

## Failure and retry rules

No transition to training if D0 fails; a diagnostic-only paper route requires a local decision.

## Successor opening

Accepted T300 opens T400.

## Review history

- 2026-08-26 — Planned; dependencies not yet accepted.

