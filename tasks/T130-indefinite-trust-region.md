---
id: T130
title: Overall-indefinite curvature and trust-region rejection
parent: T100
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T130-indefinite-trust-region
depends_on: []
blocks: [T140]
allowed_paths: ["src/comppareto/", "tests/", "configs/t1b/", "runs/t1b-indefinite-*/", "reports/T130/", "tasks/T130-indefinite-trust-region.md"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T130: Indefinite curvature and trust-region rejection

## Research claim

Positive private curvature alone is insufficient; the implementation must detect or reject misleading overall-indefinite local models.

## Objective

Construct counterexamples with positive private blocks and indefinite joint/effective curvature, then test measured trust-region acceptance.

## Dependencies and inputs

Theory failure cases and existing trust-region helper.

## Allowed changes

Synthetic counterexample code, tests, configs, runs, report, and this task file.

## Frozen protocol

Include analytically known negative-curvature directions and fresh measured-objective acceptance checks.

## Execution stages

Generate counterexamples; implement rejection/acceptance contract; add regression tests; summarize false-accept rates.

## Pass/fail gate

All known unsafe steps are rejected or reduced; no silent convex projection is reported as a proof for indefinite cases.

## First report

Return counterexample equations, acceptance rule, tolerances, and CPU estimate.

## Required deliverables

Counterexample suite, tests, manifests, summary, claim check, and failure ledger.

## Artifact and provenance requirements

Every counterexample records matrices, eigenvalues, proposed step, measured change, and source revision.

## Failure and retry rules

Unexpected acceptance becomes a blocking regression case.

## Successor opening

Acceptance contributes to T140 and T100.

## Review history

- 2026-08-26 — Authorized for remote execution; no result submitted.

