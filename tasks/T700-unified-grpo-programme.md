---
id: T700
title: Unified understanding-generation GRPO programme
parent: T000
status: running
priority: P0
owner: local-research-agent
reviewer: user
branch: main
depends_on: []
blocks: []
allowed_paths: ["PROJECT.md", "PROGRESS.md", "tasks/", "docs/plans/", "reports/"]
source_revision: "818d1d83ecf6b7b6fca924e8b1b8f7a214b0a7e5"
created_at: 2026-09-16
updated_at: 2026-09-16
---

# T700: Unified understanding-generation GRPO programme

## Research claim

The success or failure of joint understanding-generation reinforcement
learning depends on data/reward coupling, normalization, trainable parameter
ownership, and the realized shared update. A response-aware intervention is
useful only after these factors are controlled and a reproducible failure mode
is observed.

## Objective

Use Janus-Pro-1B and the public CoRL stack as the first training platform.
Reproduce the official Unified-RL path, establish single-task oracles and
training-dynamics instrumentation, then test conventional negotiators and the
proposed shared/private response gate.

## Frozen scope

- Main paper setting: unified understanding-generation GRPO.
- SFT is an implementation/headroom diagnostic, not a second full paper track.
- DPO is a bounded mechanism control, not a co-equal main setting.
- OPD is related/future work.
- No response-aware comparison starts before the official CoRL path and
  single-task controls are reproducible.

## Child sequence

```text
T710 CoRL assets/code/admission + GPU optimizer smoke
  ├── T730 Official Unified-RL exploratory reproduction
  ├── T740 Single-task GRPO oracle wave
  └── T750 Per-task gradient/update instrumentation
T720 Janus-Pro-R1 reusable training-stack audit + GPU smoke
T730 + T740 + T750
  └── T760 Traditional negotiator wave
       └── T770 Shared/private response-gated wave
```

## Successor policy

Only the local review side opens T730/T740/T760/T770 after prerequisites and
comparison budgets are frozen.

