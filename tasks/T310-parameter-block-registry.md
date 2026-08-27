---
id: T310
title: Show-o2 shared/private parameter-block registry
parent: T300
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T310-parameter-block-registry
depends_on: [T100, T210, T215]
blocks: [T320, T330]
allowed_paths: ["src/comppareto/adapters/showo2/", "tests/adapters/", "configs/d0/", "reports/T310/", "tasks/T310-parameter-block-registry.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T310: Parameter-block registry

## Research claim

Partial-overlap diagnostics require an explicit, testable mapping from task objectives to shared and private parameter blocks.

## Objective

Expose Show-o2 block IDs, task activity, shapes, trainability, optimizer ownership, and allowed sharing-depth variants.

## Dependencies and inputs

Accepted T100, T210, and T215.

## Allowed changes

Show-o2 adapter registry, tests, D0 configs, report, and this task file.

## Frozen protocol

Tokenizer/VAE remain frozen; 25%, 50%, and 100% eligible shared-depth variants use deterministic block selection.

## Execution stages

Inventory parameters; define selectors; add shape/activity tests; export registry.

## Pass/fail gate

Every trainable parameter belongs to exactly one declared block and each task's active/private mapping is reproducible.

## First report

Return module tree, proposed block boundaries, parameter counts, and ambiguous cases.

## Required deliverables

Registry code, tests, exported table, report, and failure ledger.

## Artifact and provenance requirements

Bind the registry to the accepted Show-o2 source/checkpoint revision.

## Failure and retry rules

Ambiguous ownership blocks are escalated; do not assign them arbitrarily.

## Successor opening

Acceptance contributes to T320 and T330.

## Review history

- 2026-08-26 — Planned; dependencies not accepted.
