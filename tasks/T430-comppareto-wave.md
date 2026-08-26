---
id: T430
title: CompPareto estimator and negotiation wave
parent: T400
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T430-comppareto-wave
depends_on: [T410]
blocks: [T440]
allowed_paths: ["src/comppareto/", "tests/", "configs/e1/", "experiments/", "runs/e1-comppareto-*/", "reports/T430/", "tasks/T430-comppareto-wave.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T430: CompPareto wave

## Research claim

Compensation-aware estimation plus attainable-gain negotiation should improve the equal-budget joint frontier.

## Objective

Run no-compensation, stop-gradient, unroll-1/3, and approved scalable variants under frozen budgets.

## Dependencies and inputs

Accepted T410 and immutable diagnostic/cache interfaces.

## Allowed changes

Proposed method code/tests/configs/runs/reports and this task file.

## Frozen protocol

The method receives no extra persistent private updates or hidden data beyond the declared protocol.

## Execution stages

Smoke; estimator ablations; exploratory trials; validation selection; method report.

## Pass/fail gate

Runs satisfy budget/provenance validation and produce all preregistered primary and guardrail metrics.

## First report

Return estimator implementation, resolved budgets, smoke acceptance, overhead, and GPU-hour estimate.

## Required deliverables

Code/tests, manifests, metrics, overhead, failures, and method report.

## Artifact and provenance requirements

Record virtual/persistent state semantics and every cache/budget hash.

## Failure and retry rules

Do not commit virtual inner updates or alter data accounting without a new decision.

## Successor opening

Acceptance contributes to T440.

## Review history

- 2026-08-26 — Planned; T410 not accepted.

