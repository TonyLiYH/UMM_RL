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

The active UMM PT claim is that intervention-defined ownership of shared
backbone groups predicts pruning damage and cross-task transfer or
interference under restricted CoRL post-training. See
[`docs/plans/2026-10-09-umm-parameter-ownership.md`](../docs/plans/2026-10-09-umm-parameter-ownership.md)
for the proposed measurement protocol and gates. CoRL reproduction remains
the required platform and reference result. The following optimizer claim is
a conditional follow-up rather than the core paper hypothesis.

The success or failure of joint understanding-generation reinforcement
learning depends on data/reward coupling, normalization, trainable parameter
ownership, and the realized shared update. A response-aware intervention is
useful only after these factors are controlled and a reproducible failure mode
is observed.

## Objective

Use Janus-Pro-1B and the public CoRL stack as the first training platform.
Reproduce the semantically admitted Unified-RL path and establish single-task
oracles. Map shared structured groups, then test selective pruning and
restricted post-training under matched budgets. Only a measured interference
pattern may motivate conventional negotiators or the shared/private response
gate.

## Frozen scope

- Main paper setting: unified understanding-generation GRPO.
- SFT is an implementation/headroom diagnostic, not a second full paper track.
- DPO is a bounded mechanism control, not a co-equal main setting.
- OPD is related/future work.
- No ownership-guided training comparison starts before the corrected CoRL
  path, stable importance map, and single-task controls are reproducible.
- Gradient negotiators and response-aware methods require a separate gate
  after the ownership intervention pilot.

## Child sequence

```text
T710 bounded CoRL smoke and defect ledger
        ↓
T711 semantic correctness + fixed-anchor admission
        ↓
T712 complete corrected CoRL run → T725 Unified-only reproduction
   ├── T740 U-only/G-only reference pilots
   └── shared-group registry and intervention-map pilot (task proposed)
        ↓
matched pruning + restricted post-training (tasks proposed)
        ↓
locked confirmation and second-architecture test, if gates pass

T755 real CoRL instrumentation supports update attribution.
T730/T760/T756/T770 are conditional mechanism follow-ups.

T720 Janus-Pro-R1 reusable SFT/GRPO stack audit is an engineering sidecar.
```

## Successor policy

Only the local review side opens successor tasks after prerequisites,
practical-effect margins, and comparison budgets are frozen. Existing task
statuses are unchanged by this research reprioritization.
