---
id: T320
title: Identical-A_i^K hypergradient cache
parent: T300
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T320-hypergradient-cache
depends_on: [T310]
blocks: [T330]
allowed_paths: ["src/comppareto/", "tests/", "configs/d0/", "runs/d0-cache-*/", "reports/T320/", "tasks/T320-hypergradient-cache.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T320: Identical finite-response cache

## Research claim

Negotiation methods can be compared fairly only when they consume exactly the same finite private transition and cached hypergradient.

## Objective

Define a serialized cache for raw, stop-gradient, unrolled, and implicit estimates plus optimizer-state provenance.

## Dependencies and inputs

Accepted T310 block registry and frozen D0 batches.

## Allowed changes

Cache schemas, code, tests, D0 configs/runs, report, and this task file.

## Frozen protocol

The cache stores task, state, block, batch, inner-step, optimizer, scale, and source identifiers.

## Execution stages

Define schema; implement writer/reader; add round-trip/hash tests; run smoke cache.

## Pass/fail gate

MGDA, Chebyshev, Nash, and CompPareto can read identical immutable records with matching hashes.

## First report

Return schema, estimated size, precision, compression, and privacy/storage constraints.

## Required deliverables

Schema, code, tests, smoke manifest, report, and failure ledger.

## Artifact and provenance requirements

Every cache shard records content hash, task/model revisions, optimizer state, and batch-group ID.

## Failure and retry rules

No method-specific cache regeneration after baseline selection.

## Successor opening

Acceptance contributes to T330.

## Review history

- 2026-08-26 — Planned; T310 not accepted.

