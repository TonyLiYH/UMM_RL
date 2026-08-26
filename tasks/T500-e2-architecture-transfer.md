---
id: T500
title: E2 cross-architecture validation
parent: T000
status: planned
priority: P1
owner: unassigned
reviewer: local-research-agent
branch: agent/T500-e2-architecture-transfer
depends_on: [T400]
blocks: []
allowed_paths: ["src/comppareto/", "tests/", "configs/e2/", "runs/e2-*/", "reports/T500/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T500: E2 architecture transfer

## Research claim

The method effect should transfer beyond Show-o2 and vary predictably with architecture and objective sharing.

## Objective

Validate on accepted UniDDT and SenseNova-U1 paths, with UniAR as a more homogeneous boundary control.

## Dependencies and inputs

Accepted T400 and accepted model-admission tasks.

## Allowed changes

Model adapters, E2 configs, tests, run metadata, and cross-architecture reports.

## Frozen protocol

Models are selected by admission order, not by observed CompPareto performance.

## Execution stages

Deep-sharing validation, native pixel/MoT stress test, homogeneous-objective boundary control, and joint analysis.

## Pass/fail gate

Positive worst-task result on at least two admitted architecture families with all attempted models reported.

## First report

Publish exact admitted models, block mapping, estimated resources, and evaluation compatibility.

## Required deliverables

Per-model runs, block maps, normalized metrics, costs, failures, and architecture-effect analysis.

## Artifact and provenance requirements

Use each admission manifest as the immutable model source.

## Failure and retry rules

Admission failure is not method failure; training failure after admission remains reported.

## Successor opening

No automatic successor; local review decides paper-scale expansion.

## Review history

- 2026-08-26 — Planned; E1 is not accepted.

