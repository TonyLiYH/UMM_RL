# UMM_RL authoritative task tree

Task files are the unique source of truth for task status. `PROGRESS.md` summarizes milestones; GitHub Issues or Projects may mirror active tasks but never override these files.

Status legend:

- `planned`: defined, not executable;
- `ready`: locally authorized;
- `running`: executor or local coordinator is actively working;
- `awaiting_review`: remote result submitted;
- `revision_needed`: local review requires repair;
- `blocked`: exact external blocker recorded;
- `accepted`: locally reviewed and merged;
- `stopped`: local Gate failure or deliberate termination.

Remote executors may set `running`, `awaiting_review`, or `blocked` on their task branch. Only the local review side may set `ready`, `revision_needed`, `accepted`, or `stopped` on `main`.

## Task tree

```text
T000  CompPareto / UMM_RL research [root]  running
├── T100  T1b independent solver and approximation validation  running
│   ├── T110  Random overlap quadratic families  ready
│   ├── T120  Independent KKT/direct reference  ready
│   ├── T130  Indefinite curvature and trust-region rejection  ready
│   ├── T140  CG/unroll/diagonal/low-rank error curves  planned
│   ├── T150  Negotiation feasibility and KKT audit  planned
│   ├── T155  Exact finite-response oracle benchmark  accepted
│   ├── T160  Finite-horizon optimizer-response posterior certificate  planned
│   └── T170  Graph-localized robust descent and resource allocation  planned
├── T200  Public-model admission programme  running
│   ├── T210  Show-o2 admission  accepted
│   ├── T215  Show-o2 finite-response diagnostic feasibility  ready
│   ├── T216  Show-o2 compute-matched alternating-protocol diagnostic  ready
│   ├── T220  UniDDT admission  accepted (read-only scope)
│   ├── T230  SenseNova-U1 admission  accepted
│   ├── T240  UniAR boundary-control admission  accepted
│   ├── T250  Post-training starting-checkpoint selection audit  accepted
│   └── T270  Selected-checkpoint post-training interface smoke  planned
├── T260  Joint post-training dataset admission and frozen manifests  accepted
├── T300  D0 compensation-aware conflict diagnostics  planned
│   ├── T310  Shared/private parameter-block registry  planned
│   ├── T320  Identical-A_i^K hypergradient cache  planned
│   ├── T330  Raw Taylor and compensated predictor comparison  planned
│   └── T340  Held-out calibration and certificate audit  planned
├── T400  E1 Show-o2 controlled pilot  planned
│   ├── T410  Budget and search freeze  planned
│   ├── T420  Strong baseline wave  planned
│   ├── T430  CompPareto estimator/negotiation wave  planned
│   └── T440  Confirmatory seeds and capability slices  planned
├── T500  E2 cross-architecture validation  planned
├── T600  E3 heterogeneous preference/RL validation  planned
└── T700  Unified understanding-generation GRPO programme  running
    ├── T710  CoRL assets, implementation audit, and GPU optimizer smoke  ready
    ├── T720  Janus-Pro-R1 reusable SFT and GRPO stack smoke  ready
    ├── T730  Official CoRL Unified-RL exploratory reproduction  planned
    ├── T740  CoRL single-task GRPO oracles  planned
    ├── T750  Per-task gradient/update instrumentation  ready
    ├── T760  Traditional multi-task negotiator wave  planned
    └── T770  Response-gated Unified GRPO wave  planned
```

## Current remote execution entry points

| Task | Priority | Branch | Scope |
|---|---|---|---|
| [T110](T110-overlap-family.md) | P0 | `agent/T110-overlap-family` | CPU synthetic task families |
| [T120](T120-independent-kkt-reference.md) | P0 | `agent/T120-independent-kkt-reference` | CPU independent reference solver |
| [T130](T130-indefinite-trust-region.md) | P0 | `agent/T130-indefinite-trust-region` | CPU failure and acceptance tests |
| [T215](T215-showo2-finite-response-feasibility.md) | P0 | `agent/T215-showo2-finite-response-feasibility` | Reversible Show-o2 finite-response diagnostics |
| [T216](T216-showo2-alternating-protocol-diagnostic.md) | P0 | `agent/T216-showo2-alternating-protocol-diagnostic` | Compute-matched SP vs PS/commit reversible diagnostics |
| [T710](T710-corl-admission-and-gpu-smoke.md) | P0 | `agent/T710-corl-admission-gpu-smoke` | Download CoRL assets and run bounded Unified-GRPO optimizer smoke |
| [T720](T720-janus-pro-r1-stack-smoke.md) | P1 | `agent/T720-janus-pro-r1-stack-smoke` | Download Janus-Pro-R1 assets and run SFT+GRPO stack smokes |
| [T750](T750-corl-gradient-update-instrumentation.md) | P0 | `agent/T750-corl-gradient-update-instrumentation` | Per-task gradients, clipping, AdamW update and cost instrumentation |

Only the bounded optimizer smokes in T710 and T720 are newly authorized.
T730/T740/T760/T770 remain closed. T215/T216 remain legacy reversible
diagnostics and are no longer the first-paper critical path.

## Active task table

| ID | Parent | Status | Priority | Owner | Reviewer |
|---|---|---|---|---|---|
| T000 | root | running | P0 | local-research-agent | user |
| T100 | T000 | running | P0 | local-research-agent | user |
| T110 | T100 | ready | P0 | remote-gpu-agent | local-research-agent |
| T120 | T100 | ready | P0 | remote-gpu-agent | local-research-agent |
| T130 | T100 | ready | P0 | remote-gpu-agent | local-research-agent |
| T155 | T100 | accepted | P0 | remote-gpu-agent | local-research-agent |
| T200 | T000 | running | P0 | local-research-agent | user |
| T210 | T200 | accepted | P0 | remote-gpu-agent | local-research-agent |
| T215 | T200 | ready | P0 | remote-gpu-agent | local-research-agent |
| T216 | T200 | ready | P0 | remote-gpu-agent | local-research-agent |
| T220 | T200 | accepted | P1 | remote-gpu-agent | local-research-agent |
| T230 | T200 | accepted | P1 | remote-gpu-agent | local-research-agent |
| T240 | T200 | accepted | P1 | remote-gpu-agent | local-research-agent |
| T250 | T200 | accepted | P0 | remote-gpu-agent | local-research-agent |
| T260 | T000 | accepted | P0 | remote-gpu-agent | local-research-agent |
| T270 | T200 | planned | P0 | unassigned | local-research-agent |
| T160 | T100 | planned | P0 | unassigned | local-research-agent |
| T170 | T100 | planned | P0 | unassigned | local-research-agent |
| T700 | T000 | running | P0 | local-research-agent | user |
| T710 | T700 | ready | P0 | remote-gpu-agent | local-research-agent |
| T720 | T700 | ready | P1 | remote-gpu-agent | local-research-agent |
| T730 | T700 | planned | P0 | unassigned | local-research-agent |
| T740 | T700 | planned | P0 | unassigned | local-research-agent |
| T750 | T700 | ready | P0 | remote-gpu-agent | local-research-agent |
| T760 | T700 | planned | P0 | unassigned | local-research-agent |
| T770 | T700 | planned | P0 | unassigned | local-research-agent |
