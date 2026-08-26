---
id: T110
title: Random disjoint, partial, and full overlap quadratic families
parent: T100
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T110-overlap-family
depends_on: []
blocks: [T140]
allowed_paths: ["src/comppareto/", "tests/", "configs/t1b/", "runs/t1b-overlap-*/", "reports/T110/", "tasks/T110-overlap-family.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T110: Random overlap families

## Research claim

The selector and compensation formulation must behave correctly for disjoint, partial, and full task overlap rather than only the T1a identity-selector case.

## Objective

Generate seeded PSD quadratic families and verify block lifting, objective changes, and safe-set relations across overlap regimes.

## Dependencies and inputs

Existing `QuadraticTask`, T1a tests, and the theory assumptions in `docs/theory/formulation.md`.

## Allowed changes

Only the declared synthetic code, tests, configs, runs, report, and this task file.

## Frozen protocol

Use at least 100 seeded cases per overlap regime, dimensions 2–32, and store every failing seed.

## Execution stages

Define generators; add independent direct evaluation; run deterministic tests; emit summary distributions.

## Pass/fail gate

Zero unexplained selector or objective mismatches; all failing seeds either become regression tests or stop the task.

## First report

Before implementation, return generator parameter ranges, seed policy, direct-reference calculation, and CPU estimate.

## Required deliverables

Generator code, tests, resolved config, run manifest, result summary, and failure ledger.

## Artifact and provenance requirements

Record source revision, config hash, Python/NumPy/SciPy versions, seed ranges, and output hash.

## Failure and retry rules

Do not drop difficult seeds or relax tolerance without local review.

## Successor opening

Acceptance contributes to T140 and T100.

## Review history

- 2026-08-26 — Authorized for remote execution; no result submitted.

