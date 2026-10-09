# UMM_RL authoritative task tree

Task files are the unique source of truth for task status. `PROGRESS.md` summarizes milestones. `ready` authorizes the stated work; `planned` requires a later local gate; `revision_needed` requires repair and review; `accepted` means locally reviewed and merged; `stopped` records a deliberate end to that task. Remote executors may submit evidence but may not accept their own work.

## Current mainline: parameter ownership and plasticity

| Step | Task | Status | Role |
|---|---|---|---|
| 0 | [T780](T780-ownership-protocol-freeze.md) | ready, P0 | Freeze Janus-Pro group units, U/G evaluators, splits, controls, thresholds, and pilot budget; CPU planning only |
| 1 | [T711](T711-corl-semantic-correctness-admission.md) | revision_needed, P0 | Repair CoRL scoring, sampling/score relation, rewards, masks, parameter updates, and resume; no full training |
| 2 | [T712](T712-corl-complete-training.md) | planned, P0 | Complete one corrected finite training run after T711 |
| 3 | [T725](T725-corl-unified-reference.md) | planned, P0 | Freeze and reproduce the Unified-only reference after T712 |
| 4 | [T781](T781-ownership-intervention-pilot.md) | planned, P0 | Map reversible shared-group U/G effects after T780 and T711 |
| 5 | [T740](T740-corl-single-task-oracles.md) | planned, P0 | Establish U-only and G-only headroom and interference controls |
| 6 | [T782](T782-ownership-pruning-test.md) | planned, P1 | Test held-out pruning predictions at matched removed capacity |
| 7 | [T783](T783-ownership-restricted-posttraining.md) | planned, P1 | Test restricted U/G/joint training at matched budget |
| 8 | [T784](T784-ownership-cross-architecture.md) | planned, P2 | Repeat decisive tests on a second admitted UMM if first-model gates pass |

[T700](T700-unified-grpo-programme.md) coordinates this sequence. [T755](T755-real-corl-instrumentation.md) is planned support for real update attribution and T740; [T750](T750-corl-gradient-update-instrumentation.md) is accepted for mock scope only. [T710](T710-corl-admission-and-gpu-smoke.md) is stopped as a separate task; its unresolved defects are owned by T711. [T720](T720-janus-pro-r1-stack-smoke.md) is accepted engineering evidence.

T780 is the only newly `ready` task. T711 is a bounded repair, not permission for persistent CoRL training. T712/T725 and T781–T784 remain closed until the dependencies and their written budgets pass local review. Source revisions and precise allowed paths are in each task file.

## Conditional optimizer programme

The earlier CompPareto theory and Show-o2 execution route remains in the repository for provenance and possible follow-up. It is outside the first parameter-ownership paper's critical path:

- [T100](T100-t1b-validation.md) and T110–T170: finite-response and certificate theory; P2, planned except accepted T155.
- [T200](T200-model-admission.md) retains accepted model/data admission evidence; [T215](T215-showo2-finite-response-feasibility.md), [T216](T216-showo2-alternating-protocol-diagnostic.md), and [T270](T270-selected-model-training-interface-smoke.md) are P2, planned.
- [T300](T300-d0-conflict-diagnostics.md), [T400](T400-e1-showo2-pilot.md), [T500](T500-e2-architecture-transfer.md), and [T600](T600-e3-heterogeneous-posttraining.md), including their descendants, are P2, planned.
- [T730](T730-corl-unified-exploratory-reproduction.md), [T756](T756-corl-same-snapshot-response-probes.md), [T760](T760-corl-traditional-negotiator-wave.md), and [T770](T770-corl-response-gated-wave.md) are P2, planned. Fixed-anchor finite-response validation belongs to T756 if that route opens.

These tasks require an observed interference pattern that simpler ownership-guided restrictions do not resolve, plus a new budget and comparison gate before promotion. Accepted evidence remains accepted; lowering priority does not erase it.

## Current execution entry points

| Task | Owner | Next action |
|---|---|---|
| T780 | local-research-agent | Prepare the reviewable protocol and ambiguity ledger without GPU work |
| T711 | remote-gpu-agent under local review | Rebase repair work on latest authorized `main`, implement and test A–E, submit evidence against the revised acceptance contract |
