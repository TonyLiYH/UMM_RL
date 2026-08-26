---
id: T340
title: Held-out calibration and common-descent certificate audit
parent: T300
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T340-calibration-audit
depends_on: [T330]
blocks: [T400]
allowed_paths: ["src/comppareto/", "tests/", "configs/d0/", "runs/d0-calibration-*/", "reports/T340/", "tasks/T340-calibration-audit.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T340: Calibration and certificate audit

## Research claim

Stochastic common-descent signals are useful only when their confidence and realized safety are calibrated on held-out steps.

## Objective

Measure sign precision/recall, Brier score, Spearman correlation, confidence coverage, and measured-loss acceptance.

## Dependencies and inputs

Accepted T330 outputs and frozen calibration/held-out partitions.

## Allowed changes

Calibration code, tests, configs, runs, reports, and this task file.

## Frozen protocol

Repeated microbatches estimate confidence; uncertified steps remain uncertified rather than relabeled safe.

## Execution stages

Fit calibration on calibration split; evaluate held-out; audit certificate failures; issue D0 decision report.

## Pass/fail gate

Frozen predictive and safety thresholds pass against the strongest raw Taylor baseline.

## First report

Return calibration method, interval construction, thresholds, sample count, and GPU estimate.

## Required deliverables

Calibration artifacts, held-out metrics, certificate table, decision report, and failure ledger.

## Artifact and provenance requirements

Record split manifest/hash, calibration parameters, predictions, realized changes, and revisions.

## Failure and retry rules

Held-out results cannot be used to retune calibration.

## Successor opening

Acceptance completes T300 and opens T400.

## Review history

- 2026-08-26 — Planned; T330 not accepted.

