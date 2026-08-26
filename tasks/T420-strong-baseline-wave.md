---
id: T420
title: Strong baseline training wave
parent: T400
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T420-strong-baseline-wave
depends_on: [T410]
blocks: [T440]
allowed_paths: ["src/comppareto/", "tests/", "configs/e1/", "experiments/", "runs/e1-baseline-*/", "reports/T420/", "tasks/T420-strong-baseline-wave.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T420: Strong baseline wave

## Research claim

CompPareto is meaningful only against tuned scalarization and general negotiation methods consuming the identical finite-response cache.

## Objective

Run base, single-task oracles, isolation, scheduling, scalarization, MGDA/MOBLO, normalized Chebyshev, Nash, ConsMTL, and expert-integration baselines.

## Dependencies and inputs

Accepted T410 budgets and immutable T320-style cache interface.

## Allowed changes

Baseline implementations/tests/configs/runs/reports and this task file.

## Frozen protocol

All baselines use the same task data, seeds, cache where applicable, and search budget.

## Execution stages

Smoke; exploratory trials; validation selection; preregistered baseline report.

## Pass/fail gate

Required baselines complete or have an accepted, exact incompatibility report; no weak substitute is chosen after results.

## First report

Return implementation versions, cache inputs, resolved budgets, smoke results, and expected GPU hours.

## Required deliverables

Code/tests, manifests, validation results, costs, failures, and baseline report.

## Artifact and provenance requirements

Every selected baseline checkpoint maps to its full search lineage.

## Failure and retry rules

Numerical divergence is a completed trial; only frozen infrastructure retries apply.

## Successor opening

Acceptance contributes to T440.

## Review history

- 2026-08-26 — Planned; T410 not accepted.

