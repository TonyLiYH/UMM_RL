---
id: T410
title: E1 budget, search, and run-contract freeze
parent: T400
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T410-budget-search-freeze
depends_on: [T300]
blocks: [T420, T430]
allowed_paths: ["configs/e1/", "schemas/", "src/comppareto/repo_state/", "tests/repo_state/", "reports/T410/", "tasks/T410-budget-search-freeze.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T410: Budget and search freeze

## Research claim

E1 comparisons require executable equal-FLOP, equal-wall-clock, and equal-shared-update accounting fixed before training results.

## Objective

Resolve method-level budgets, search spaces, surplus allocations, retry rules, and runtime hard-fail validation.

## Dependencies and inputs

Accepted T300 and measured Show-o2 smoke capacity.

## Allowed changes

E1 budget configs, schemas, validators, tests, report, and this task file.

## Frozen protocol

At least extra-private-step and extra-shared-step surplus controls are preregistered; search trial counts and ranges are fixed.

## Execution stages

Profile smoke; create resolved budgets; hash configs; validate counters; publish freeze decision.

## Pass/fail gate

Every method has a validated immutable allocation and the runtime aborts on budget overrun.

## First report

Return profiler evidence, proposed allocations, search matrix, and total GPU-hour upper bound.

## Required deliverables

Resolved budgets, hashes, validator tests, freeze report, and failure ledger.

## Artifact and provenance requirements

Budget files map to model, hardware, source revision, and profiling runs.

## Failure and retry rules

Budget changes require a new local decision and invalidate unstarted comparisons.

## Successor opening

Acceptance opens T420 and T430.

## Review history

- 2026-08-26 — Planned; D0 not accepted.

