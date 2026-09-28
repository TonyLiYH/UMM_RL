---
id: T755
title: Real CoRL per-task gradient and optimizer instrumentation
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T755-real-corl-instrumentation
depends_on: [T711, T750]
blocks: [T730, T740, T760, T770]
allowed_paths: ["tasks/T755-real-corl-instrumentation.md", "configs/instrumentation/corl-real/", "runs/instrumentation-corl-real-v1/", "reports/T755/", "src/comppareto/adapters/corl/", "src/comppareto/instrumentation/", "tests/adapters/corl/", "tests/instrumentation/"]
source_revision: "1fcb9e9964c678a330f32514f211828d551dc48c"
created_at: 2026-09-28
updated_at: 2026-09-28
---

# T755: Real CoRL per-task gradient and optimizer instrumentation

## Objective

Integrate T750 measurements into the semantically corrected CoRL PyTorch
path. Demonstrate that U-only, G-only, combined, clipped, and AdamW-updated
parameter tensors can be captured and reconstructed without changing the
actual real-model update.

## Required evidence

- actual U-only/G-only gradients on identical anchored trajectories;
- semantic parameter ownership map, including text output head and weight tying;
- pre/post clipping and actual optimizer parameter membership;
- full-tensor update reconstruction;
- checkpoint/resume U/G log-probability equality;
- per-group advantages, candidate IDs, rollout counts, tokens, reward calls,
  backward count, wall time, and peak memory.

## Authorization boundary

Planned until T711 passes semantic correctness. This task remains bounded to
instrumentation and diagnostic batches; it does not claim task-level learning.

