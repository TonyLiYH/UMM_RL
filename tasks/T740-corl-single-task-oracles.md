---
id: T740
title: CoRL understanding-only and generation-only GRPO oracles
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T740-corl-single-task-oracles
depends_on: [T710]
blocks: [T760]
allowed_paths: ["tasks/T740-corl-single-task-oracles.md", "configs/corl/oracles-v1/", "runs/corl-oracles-v1/", "reports/T740/", "src/comppareto/adapters/corl/", "tests/adapters/corl/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T740: CoRL single-task GRPO oracles

## Objective

Run compute-matched understanding-only and generation-only GRPO from the same
checkpoint and data universe to define negative transfer relative to
single-task training.

## Authorization boundary

Planned until T710 is accepted and the local side freezes matched rollout,
token, reward-call, optimizer-step, seed, and evaluation budgets.

