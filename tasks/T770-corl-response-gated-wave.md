---
id: T770
title: Response-gated shared update for Unified GRPO
parent: T700
status: planned
priority: P0
owner: unassigned
reviewer: local-research-agent
branch: agent/T770-corl-response-gated-wave
depends_on: [T756, T760]
blocks: []
allowed_paths: ["tasks/T770-corl-response-gated-wave.md", "configs/corl/response-gated-v1/", "runs/corl-response-gated-v1/", "reports/T770/", "src/comppareto/response_gate/", "tests/response_gate/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T770: Response-gated shared update for Unified GRPO

## Objective

Run locked multi-seed confirmation of the simplest response-informed shared
update that passes T756/T760. Compare against the locked strongest conventional
comparator at matched per-task exposure and matched total compute.

## Authorization boundary

Planned. Open only if T756 establishes a reproducible decision-quality
opportunity and T760 leaves practically meaningful joint headroom. A baseline
failure is not itself required.
