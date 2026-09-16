---
id: T760
title: CoRL traditional multi-task negotiator wave
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T760-corl-traditional-negotiator-wave
depends_on: [T730, T740, T750]
blocks: [T770]
allowed_paths: ["tasks/T760-corl-traditional-negotiator-wave.md", "configs/corl/negotiators-v1/", "runs/corl-negotiators-v1/", "reports/T760/", "src/comppareto/negotiators/", "tests/negotiators/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T760: CoRL traditional multi-task negotiator wave

## Objective

Compare official CoRL sum, independent-advantage sum, token-normalized
weighting, MGDA, PCGrad, and CAGrad under frozen rollouts, rewards, trainable
parameters, tuning budget, and evaluation protocol.

## Authorization boundary

Planned until official Unified-RL, single-task oracles, and instrumentation are
accepted. The local side freezes compute and search budgets before opening.

