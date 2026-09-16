---
id: T730
title: Official CoRL Unified-RL exploratory reproduction
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T730-corl-unified-exploratory-reproduction
depends_on: [T710]
blocks: [T760]
allowed_paths: ["tasks/T730-corl-unified-exploratory-reproduction.md", "configs/corl/unified-v1/", "runs/corl-unified-v1/", "reports/T730/", "src/comppareto/adapters/corl/", "tests/adapters/corl/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T730: Official CoRL Unified-RL exploratory reproduction

## Objective

After T710 freezes a corrected executable protocol, run a bounded Unified-RL
training reproduction from Janus-Pro-1B through checkpoint evaluation on both
understanding and generation.

## Authorization boundary

Planned only. The local reviewer must freeze dataset size, optimizer steps,
evaluation subset, seed count, corrected/upstream path, and GPU budget after
T710. No executor may start from this placeholder.

