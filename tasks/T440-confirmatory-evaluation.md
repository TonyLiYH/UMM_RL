---
id: T440
title: Confirmatory seeds and capability-slice evaluation
parent: T400
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T440-confirmatory-evaluation
depends_on: [T420, T430]
blocks: [T500, T600]
allowed_paths: ["src/comppareto/", "tests/", "configs/e1/", "runs/e1-confirm-*/", "reports/T440/", "tasks/T440-confirmatory-evaluation.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T440: Confirmatory evaluation

## Research claim

The E1 claim requires power-selected training seeds, hierarchical uncertainty, and counting/spatial/OCR/composition guardrails.

## Objective

Run confirmatory seeds for frozen selected configurations and issue the E1 Gate decision package.

## Dependencies and inputs

Accepted T420/T430 and frozen validation-selected methods.

## Allowed changes

Confirmatory configs, evaluation code/tests, runs, reports, and this task file.

## Frozen protocol

Seed count follows the preregistered power calculation; test metrics do not influence configuration selection.

## Execution stages

Power calculation; confirmatory training; capability evaluation; hierarchical statistics; failure audit.

## Pass/fail gate

Worst-task interval, retained gain, hypervolume, and negative-transfer guardrails meet the E1 thresholds.

## First report

Return variance estimate, detectable effect, selected seed count, model configs, and total GPU-hour bound.

## Required deliverables

Per-seed runs, capability metrics, statistics, costs, failures, and E1 claim-check report.

## Artifact and provenance requirements

All selected configurations map to the exploratory lineage and immutable evaluation manifests.

## Failure and retry rules

Missing seeds remain missing; no example-only bootstrap substitutes for training-seed uncertainty.

## Successor opening

Acceptance completes T400 and opens T500; local review may open T600.

## Review history

- 2026-08-26 — Planned; T420/T430 not accepted.

