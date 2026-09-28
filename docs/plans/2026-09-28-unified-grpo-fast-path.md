# Unified GRPO fast-path experiment plan

## Precise research claim

> Under a fixed private-adaptation budget and a semantically valid Unified-GRPO
> surrogate, response information can identify shared updates whose incremental
> value is misjudged by current-state gradients, and can improve joint outcomes
> over tuned alternating optimization at equal total compute.

This claim is narrower than generic bilevel optimization, does not assert
update-order acceleration, and applies first to Janus-Pro-1B Unified GRPO.

## Phase P0: correctness and measurement

**Tasks:** T711, then T755.  
**Budget cap:** 20 H20 GPU-hours combined.

Required outcomes:

1. causal image-token scoring parity;
2. matched behavior/scoring policy or an explicitly labeled surrogate;
3. repaired per-row MC/OE dispatch;
4. fixed reward/reduction/mask semantics;
5. frozen behavior anchors for local counterfactuals;
6. actual PyTorch per-task gradients and realized AdamW updates.

Any failure stops research training. It is an implementation/estimand failure,
not a negative result about the response idea.

## Phase P1: short learning and headroom pilots

Open only after P0 passes. Use one base checkpoint, a frozen common data
universe, independently scored held-out capability metrics, and an exploratory
single seed.

| Arm | Trainable scope | Purpose |
|---|---|---|
| U reference | shared policy mask | understanding headroom |
| G reference | shared policy mask | generation headroom |
| Faithful joint | official frozen-private policy mask | public-method reference |
| Joint-private | same joint objective + audited small private modules | private capacity |
| Simultaneous-private | independent task objectives + same private mask | simple joint baseline |
| Shared-then-private | same objectives and private mask, \(K=1\) | strongest simple response explanation |

Use a maximum of 128 update windows, with one preregistered extension to 256
only if both single-task references show learning but joint comparison is
inconclusive.

Measure both:

1. matched task exposure;
2. matched total accelerator budget.

Single-task runs are empirical references, not mathematical oracles.

## Phase P2: same-snapshot response opportunity

Use approximately 24 preregistered windows distributed across base, early, and
later P1 checkpoints. Each window freezes:

- trajectories and candidate ordering;
- old policy log probabilities;
- masks, rewards, advantages, and evaluator snapshot;
- parameters, optimizer moments, scheduler, and RNG;
- a disjoint audit batch.

Evaluate five states:

\[
L_{00},\quad L_{10},\quad L_{01},\quad L_{11},\quad L_c,
\]

corresponding to neither update, shared-only, private-only, shared-then-private,
and commit-endpoint-held-fixed.

Primary quantities:

\[
\Delta^{private}=L_{01}-L_{00},
\]

\[
\Delta^{controlled}=L_{11}-L_{01},
\]

\[
I=L_{11}-L_{10}-L_{01}+L_{00},
\]

\[
R=L_{11}-L_c.
\]

Response opportunity exists only when a response-informed candidate produces
better held-out incremental shared decisions than tuned raw/SP candidates,
after accounting for equal-private and equal-total-compute controls.

## Phase P3: compact method screen

Open only if P1 shows headroom and P2 shows a differential response signal.

Screen:

1. tuned scalarization;
2. simultaneous private-enabled training;
3. shared-then-private;
4. persistent private-then-shared;
5. strongest conventional negotiator;
6. raw-direction line search with equal validation budget;
7. simplest response-informed candidate.

Use equal declared tuning expenditure. Advance only the strongest conventional
competitor and the simplest response method.

## Phase P4: locked confirmation

At least three independent training seeds, with the seed count and endpoint
frozen before results are examined. Report every seed, independent capability
metrics, task exposure, total compute, rollout count, effective tokens, reward
calls, backward work, rejected updates, peak memory, and GPU time.

## Pre-registered gates

| Gate | Advance condition | Stop/reframe |
|---|---|---|
| Correctness | P0 passes all semantic tests | no research training |
| Signal | at least 20% of prompt groups have usable within-group reward variation | diagnose reward/data once |
| Headroom | both U and G references improve development metric by ≥1 point at fixed horizon | one extension, then stop setup |
| Response | ≥20% lower held-out update-decision MAE than strongest cheap predictor and favorable controlled gain relative to SP | stop response-complexity branch |
| Efficacy | >1 point practical improvement on one primary metric with ≤1 point degradation on the other, versus locked strongest comparator | noninferiority/benefit not demonstrated |
| Gate utility | accepted updates create progress after rejected-work cost | simplify or stop if acceptance <20% twice |

## Deferred work

- Broad DPO/SFT/OPD claims;
- second UMM before a first-model effect survives controls;
- exhaustive reward factorials;
- full AdamW trajectory unrolling;
- full Janus-Pro-R1 scientific reproduction.

