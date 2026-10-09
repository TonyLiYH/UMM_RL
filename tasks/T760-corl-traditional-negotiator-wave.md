---
id: T760
title: CoRL traditional multi-task negotiator wave
parent: T700
status: planned
priority: P2
owner: unassigned
reviewer: local-research-agent
branch: agent/T760-corl-traditional-negotiator-wave
depends_on: [T725, T730, T740, T755]
blocks: [T756, T770]
allowed_paths: ["tasks/T760-corl-traditional-negotiator-wave.md", "configs/corl/negotiators-v1/", "runs/corl-negotiators-v1/", "reports/T760/", "src/comppareto/negotiators/", "tests/negotiators/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-10-09
---

# T760: CoRL traditional multi-task negotiator wave

## Objective

Screen conventional alternatives under the corrected CoRL GRPO data, reward,
and evaluator protocol. Hold task objectives, parameter masks, tuning
allowance, and checkpoint selection fixed. The first wave compares raw U+G
sum, tuned scalar weights, explicit per-task token/advantage normalization,
U/G alternating updates, and shared-then-private. Give every eligible method
the same private-step allowance and include private-only when private modules
train. The second wave implements PCGrad, MGDA, CAGrad, and GradNorm on audited
shared parameters, with task-specific private gradient routing. Carry the
strongest conventional negotiator into the final method screen.

On fixed batches, reconstruct the sum update and log gradients before and
after clipping. Test inactive blocks, zero gradients, mixed precision, and
optimizer-state continuity. Then run a bounded one-seed screen with both
matched task exposure and matched accelerator time. Record reward coupling,
loss reduction, parameter masks, and optimizer rule as separate axes.

## Authorization boundary

Planned until T725, T730, T740, and T755 establish the reference, learning
headroom, and audited updates. The local side freezes practical margins,
per-arm horizon, compute and search budgets, evaluator, and comparison masks
before opening. Response-opportunity evidence is evaluated in T756 afterward.
