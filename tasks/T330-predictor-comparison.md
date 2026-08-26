---
id: T330
title: Raw Taylor and compensation-aware predictor comparison
parent: T300
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T330-predictor-comparison
depends_on: [T310, T320]
blocks: [T340]
allowed_paths: ["src/comppareto/", "tests/", "configs/d0/", "runs/d0-predictor-*/", "reports/T330/", "tasks/T330-predictor-comparison.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T330: Predictor comparison

## Research claim

Compensation-aware predictors should improve realized loss-change prediction beyond raw first- and second-order Taylor models.

## Objective

Evaluate raw cosine, first-order Taylor, damped second-order Taylor, stop-gradient, finite unroll, and implicit/Schur predictors.

## Dependencies and inputs

Accepted T310/T320 and frozen paired semantic batch groups.

## Allowed changes

Predictor/evaluation code, tests, D0 configs/runs, report, and this task file.

## Frozen protocol

Calibration and held-out split by checkpoint/step-window/semantic-group tuple; cosine is never the Gate baseline.

## Execution stages

Generate predictions; measure fresh realized changes; calculate errors and ranking/calibration metrics.

## Pass/fail gate

No data leakage; compensation-aware predictor exceeds the strongest raw Taylor baseline on frozen held-out metrics.

## First report

Return split counts, step proposals, curvature budgets, prediction metrics, and expected GPU cost.

## Required deliverables

Prediction table, metrics, tests, manifests, report, and failure ledger.

## Artifact and provenance requirements

Every row maps to immutable cache records and realized evaluation batches.

## Failure and retry rules

Do not change the held-out split or primary metric after results are visible.

## Successor opening

Acceptance contributes to T340.

## Review history

- 2026-08-26 — Planned; T310/T320 not accepted.

