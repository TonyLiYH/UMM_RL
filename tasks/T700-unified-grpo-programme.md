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
T710 bounded CoRL smoke and defect ledger
        ↓
T711 semantic correctness + fixed-anchor admission
        ↓
T755 real CoRL instrumentation
   ┌────┴─────────┐
   ↓              ↓
T730 joint/private-enabled headroom pilots
T740 U-only/G-only reference pilots
   └────┬─────────┘
        ↓
T756 same-snapshot response-opportunity probes
        ↓
T760 compact conventional-versus-response method screen
        ↓
T770 locked multi-seed response-method confirmation

T720 Janus-Pro-R1 reusable SFT/GRPO stack audit is an engineering sidecar.
```

## Successor policy

Only the local review side opens T730/T740/T756/T760/T770 after prerequisites,
practical-effect margins, and comparison budgets are frozen.
