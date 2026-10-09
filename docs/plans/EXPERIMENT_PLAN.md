# UMM post-training experiment plan — 2026-10-09

## Updated research priority

The core subdirection is now intervention-defined shared-parameter ownership:
test whether group importance predicts pruning damage and transfer or
interference from restricted CoRL post-training. The proposed definitions,
controls, and gates are in
[`2026-10-09-umm-parameter-ownership.md`](2026-10-09-umm-parameter-ownership.md).
CoRL semantic admission and Unified-only reproduction below remain P0
prerequisites. After those, the next proposed experiments are a shared-group
registry, perturbation-map stability pilot, matched pruning, and restricted
post-training. The T760/T756/T770 sequence below records the earlier optimizer
route; it is conditional on a measured interference pattern and a separate
authorization decision. Existing task statuses do not change here.

## Decision and evidence boundary

The immediate objective is to make one semantically valid CoRL Unified-GRPO
training run complete from a pinned Janus-Pro-1B checkpoint through held-out
understanding and generation evaluation and exact resume. Next reproduce the
Unified-only reference under a frozen recipe. Implement and compare basic
multi-task methods before any response-aware method trial.

This plan proposes successor tasks. It does not change task authorization:
`tasks/*.md` remains the status source. In particular, T711 is awaiting local
review on its remote branch; T730/T740/T755/T756/T760/T770 remain closed on
authoritative `main` until their contracts are revised or accepted.

Do not equate three evidence levels:

1. **End-to-end training:** a real policy changes, resumes, and receives both
   held-out evaluations. No quality-gain claim follows from this alone.
2. **CoRL Unified-only reference:** a pinned, paper-described or explicitly
   corrected recipe runs to its frozen horizon and is compared with the same
   base checkpoint on the declared benchmark subset. The report distinguishes
   the corrected protocol from the published recipe.
3. **Method comparison:** alternatives share data, rewards, trainable masks,
   evaluators, task exposure, and a declared compute/search budget.

## Claims served

| ID | Testable claim | Evidence required |
|---|---|---|
| C0 | The corrected CoRL stack is a usable training framework. | One complete run, finite U/G updates, fixed held-out evaluation, checkpoint/resume equivalence, and resource log. |
| C1 | The Unified-only reference has measurable U/G learning headroom. | Frozen base-vs-trained U/G metrics and curves, plus single-task reference trajectories; outcome may be negative. |
| C2 | Basic update rules differ under a common CoRL protocol. | Matched-budget training curves and capability slices for strong simple baselines. |
| C3 | A private-response method adds value beyond those baselines. | Same-snapshot decision evidence, then locked training comparison. This claim is deferred until C0–C2 have evidence. |

## Execution order and proposed task contracts

```text
T711 independent local review and any semantic repair
  -> T712 first end-to-end corrected CoRL training (new; proposed)
      -> T725 Unified-only reference reproduction (new; proposed)
      -> T755 real CoRL instrumentation
          -> T740 U-only and G-only references
          -> T730 joint and private-enabled headroom pilots
              -> T760 basic-method implementation and training wave (revise)
                  -> T756 same-snapshot response-opportunity probes (revise order)
                      -> T770 response method only if its gate passes
```

T725 and T755 can proceed in parallel after T712. T740 and T730 can proceed in
parallel after T755 and the reference protocol is frozen. Their outputs must be
reviewed together before a negative-transfer claim.

### T711 — accept a valid training estimand

Local review checks the task's complete A–F contract against code, tests, and
run artifacts. Current remote report records a failed CFG=1 probe, no complete
save/resume equality test, and incomplete fixed-anchor state. Resolve these
items before marking T711 accepted. A successful task-validator result or
`confirmed=true` field alone is not semantic evidence.

**Gate:** causal image scoring, sampling/scoring policy, mixed QA rewards,
padding, actual optimizer state restoration, and counterfactual anchors meet
the preregistered tolerances. Keep paper-described, upstream-executable,
corrected, and reduced-smoke identities separate.

### T712 — first complete training run

Use the accepted corrected protocol, one pinned checkpoint, a frozen train/dev
split from `x2x_rft_22k`, and one exploratory seed. Freeze the U/G evaluator
inputs and reward model revisions before running. Start with a short throughput
and evaluation calibration; then run up to 128 update windows, saving base,
middle, and final snapshots. Resume from the middle snapshot and compare the
continued run with an uninterrupted control on fixed U/G log probabilities,
optimizer moments, scheduler, and RNG. Evaluate both capabilities at every
snapshot, including the base checkpoint. Record rejected groups, reward
variation, effective tokens, rollouts, reward calls, backward calls, peak
memory, elapsed time, and checkpoint bytes.

**Pass:** the declared horizon completes with finite updates and reproducible
resume; both held-out evaluators return valid metrics; all parameter changes
fall inside the frozen mask. Improvement is measured, not required to call the
framework executable. If reward variation is below the frozen 20% usable-group
gate or either evaluator fails, diagnose the cause before a longer run.

**Provisional resource cap:** one H20, at most 8 H20 GPU-hours, no more than
128 windows. The T710 four-step smoke took 106 seconds, so the training-only
extrapolation is about 0.94 GPU-hours for 128 steps; evaluation, checkpointing,
and longer outputs can dominate. Re-estimate the full cost from the first
calibration batch before the main run. Reserve storage from measured checkpoint
bytes, with space for at least three snapshots and optimizer state. These are
planning bounds, not approved task budgets. T712 is a new stage outside the
existing 20-H20-GPU-hour P0 cap for T711 plus T755; update the fast-path budget
record before authorizing it.

### T725 — Unified-only reference reproduction

Freeze the exact paper-described recipe, public executable recipe, corrected
protocol delta, horizon, batch exposure, evaluation subset, and tolerance
before training. Run one seed from the same base checkpoint through the
declared Unified-only horizon. Evaluate understanding and generation on the
same checkpoint. Compare against the base and paper-reported Unified-only
figures, where protocols are comparable. Report every discrepancy and the
cost of achieving the reference. Treat a mismatch as a finding to diagnose;
do not tune on the test set to match a paper number.

**Gate:** all declared metrics and costs are reported. “Framework reproduced”
requires the corrected implementation, frozen recipe, and complete run to be
auditable. “Paper result reproduced” additionally requires preregistered
metric tolerances to pass. Freeze GPU and storage caps after T712 throughput
and evaluation measurements; the present evidence does not support a reliable
full-run estimate.

### T755, T740, T730 — measurement and headroom

Integrate real PyTorch U-only/G-only gradients, clipping, AdamW membership,
and realized updates without altering the CoRL update (T755). Run U-only and
G-only from the same base under both matched task exposure and matched total
GPU budget (T740). Run faithful frozen-private joint training and a separately
audited small-private-mask configuration (T730). Use the same evaluator and
checkpoint-selection policy. Keep task-private capacity identical across any
later methods that use it.

**Gate:** both single-task paths have measured headroom, or the project records
why a capability lacks headroom. A single-task reference is empirical, not an
optimum. Persistent method comparisons require stable, nondegenerate rewards
and measured U/G results.

### T760 — basic methods before response-aware methods

Revise T760 so that its basic-method wave depends on accepted T725, T755,
T740, and T730, but not on T756. Keep all methods inside the same corrected
GRPO data/reward/evaluator protocol. Implement in two small waves:

1. **Simple training controls:** raw U+G sum, tuned scalar weights, explicit
   per-task token/advantage normalization, U/G alternating updates, and
   shared-then-private with the same private-step allowance given to every
   eligible method. Include private-only when private modules train.
2. **Gradient negotiators:** PCGrad, MGDA, CAGrad, and GradNorm. Apply each
   rule only to audited shared parameters; route private gradients by task.
   Compare one strongest conventional negotiator in the final method screen.

First verify every implementation on fixed batches: reconstruct the original
sum update; log task gradients before and after clipping; test inactive blocks,
zero gradients, mixed precision, and optimizer-state continuity. Then run a
bounded one-seed screen. Freeze a common tuning allowance and compare both
matched task exposure and matched total accelerator time. Keep reward coupling,
loss reduction, parameter masks, and optimizer rule as separately logged axes;
do not change several at once and attribute the result to a negotiator.

**Gate:** each method either completes the frozen horizon or has a recorded
failure. Report all U/G development metrics, capability slices, costs, and
rejected groups. Select the strongest cheap comparator by a preregistered
joint criterion; reserve held-out test data for the locked comparison.

### T756 and T770 — later response-method decision

After the basic wave, test same-snapshot response opportunity against the
strongest cheap predictor and simple alternating control (T756). Freeze
trajectories, behavior probabilities, rewards, masks, optimizer state, RNG,
and a disjoint audit batch within each counterfactual window. Advance to
response-gated training (T770) only if response information improves held-out
update decisions by the frozen practical margin and yields favorable
controlled shared gain at equal private and total compute.

## Resource and decision ledger

| Stage | GPU planning basis | Main decision |
|---|---|---|
| T711 | Existing cap: 16 H20 GPU-hours; review itself is local. | Accept or return specific semantic gaps. |
| T712 | Proposed cap: 8 H20 GPU-hours; recalibrate after short run. | Is one corrected CoRL training complete and reproducible? |
| T725 | Budget set from T712 throughput and published horizon. | Is the Unified-only reference auditable, and are paper metrics comparable? |
| T755 | Existing task is bounded instrumentation. | Do logged gradients reconstruct the actual update? |
| T740/T730 | Freeze after T725/T755. | Is there U/G headroom and an audited private-response opportunity? |
| T760 | Freeze per-arm horizon, tuning allowance, and total GPU cap after baseline timing. | Which basic method is the strongest fair comparator? |
| T756/T770 | Existing gates; authorize only after preceding evidence. | Does response information improve decisions and final results? |

The critical dependency chain is T711 -> T712 -> T755 -> T740/T730 -> T760 ->
T756 -> T770, with T725 joining before T760. T725 can run beside T755. Its
result must be reviewed before claims that any method improves on a reproduced
CoRL reference.

## Repository actions before dispatch

1. Review the remote T711 submission against its full task contract and
   record `accepted` or `revision_needed` on authoritative `main`.
2. Create T712 and T725 task files, acceptance contracts, branch names,
   `allowed_paths`, first-report requirements, and task-tree entries. Freeze
   T712's train/dev split, evaluator, horizon, resume tolerance, and cap.
3. After T712, update T755/T730/T740 contracts with the measured protocol,
   artifacts, and resource envelope.
4. Revise T760's dependencies and method matrix; move T756 after the basic
   wave. Keep all successor tasks `planned` until their prerequisites pass.
