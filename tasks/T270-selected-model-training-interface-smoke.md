---
id: T270
title: Selected-checkpoint post-training interface smoke
parent: T200
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T270-selected-model-training-interface-smoke
depends_on: [T250, T260]
blocks: [T300, T400]
allowed_paths: ["tasks/T270-selected-model-training-interface-smoke.md", "configs/training-smoke/selected-model/", "runs/training-smoke-selected-model-v1/", "reports/T270/", "src/comppareto/adapters/selected_model/", "tests/adapters/selected_model/"]
source_revision: "45c54ba403a2b5c95985a49206243449436717c8"
created_at: 2026-09-15
updated_at: 2026-09-15
---

# T270: Selected-checkpoint post-training interface smoke

## Research claim

The checkpoint selected by T250 can consume T260 data and execute reversible
understanding-private, generation-private, and shared gradient steps before
formal D0/E1 training is authorized.

## Objective

Confirm loss construction, trainable ownership, optimizer-state creation,
interleaved task batches, restore, and resource cost on the selected model.

## Authorization boundary

This task remains planned until T250 and T260 are accepted and the local side
replaces generic paths with a model-specific contract. No remote executor may
begin from this placeholder.

## Planned gate

- one understanding training loss/backward;
- one generation training loss/backward;
- one same-shared-version negotiation window;
- exact rollback of parameters, optimizer state, RNG, buffers, and data order;
- zero persistent updates;
- measured memory/wall clock;
- all rows traceable to accepted T260 manifests.

## Successor opening

Accepted T270 contributes to opening D0 and the formal controlled pilot.

