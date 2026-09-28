# Unified GRPO fast-path auto review

**Date:** 2026-09-28  
**Review mode:** hard, multi-agent adversarial review  
**Scope:** the shared/private response idea, T710/T720/T750 evidence, and the
fastest credible route to a Unified-GRPO paper.

## Round 1: current evidence

### Review outcome

| Reviewer focus | Score | Verdict | Central finding |
|---|---:|---|---|
| CoRL implementation correctness | 3/10 | not ready | Public Unified-GRPO execution is not yet a semantically verified training baseline. |
| Fast publishable evidence | 3/10 | not ready | Engineering smokes exist, but there is no headroom, response-opportunity, or method-benefit evidence. |
| Shared/private theory and controls | 4/10 | not ready | The viable claim is finite-budget decision quality, not update-order acceleration. |

### Sustained strengths

1. The project has real-model training components, pinned assets, local-SSD
   execution, bounded optimizer smokes, and checkpoint evidence.
2. The quadratic analysis correctly rejects the overclaim that reversing
   persistent shared/private updates automatically accelerates convergence.
3. The private-only and controlled-shared-gain decomposition is a useful
   attribution foundation.

### Blocking findings

1. **Causal image-token scoring is unverified.** The current probe compares
   unshifted versus circularly shifted labels, rather than comparing
   teacher-forced and cached next-token distributions.
2. **Batched MC reward dispatch is incorrect.** A list-valued question-type
   argument takes the OE path even for MC-only batches.
3. **Sampling and scoring policies differ.** CFG image sampling and
   conditional-only scoring, plus implicit text truncation, need an explicit
   protocol decision.
4. **The public executable, paper-described recipe, corrected research
   protocol, and reduced smoke are currently conflated.**
5. **CoRL freezes the obvious task-private policy modules.** A response method
   has no mechanism opportunity unless an explicitly audited trainable-private
   configuration is introduced and applied equally to every comparator.
6. **GRPO counterfactual values require frozen behavior anchors.** Candidate
   directions must share trajectories, old log probabilities, masks,
   advantages, evaluator state, optimizer state, and RNG within a diagnostic
   window.
7. **Mock instrumentation is not real-model instrumentation.**

## Required repair and experimental decision

The programme is narrowed to the following falsifiable claim:

> Under a fixed private-adaptation budget and a semantically valid Unified-GRPO
> surrogate, response information can identify shared updates whose incremental
> value is misjudged by current-state gradients, and can improve joint outcomes
> beyond tuned alternating optimization at equal total compute.

The claim is rejected if:

- no audited trainable-private response opportunity exists;
- shared-then-private captures all measurable benefit;
- response estimates improve prediction but not held-out shared-update
  decisions;
- the apparent benefit vanishes under equal total compute or a raw-direction
  line-search control.

## Fast-path DAG

```text
T711 semantic correctness admission
        ↓
T755 real CoRL instrumentation
        ↓
P1 short U-only / G-only / joint / private-enabled pilots
        ├── same-snapshot response-opportunity probes
        └── equal-private and equal-total-compute controls
        ↓
headroom + differential-response gate
        ├── no opportunity → stop/reframe optimizer claim
        └── opportunity → small strong-baseline screen
                         ↓
                     locked multi-seed confirmation
```

## Explicitly deferred work

- Full Janus-Pro-R1 reproduction as a scientific baseline;
- broad DPO/SFT/OPD programmes;
- full-model AdamW unrolling;
- a second UMM before a first-model effect survives controls;
- exhaustive reward ablations.

## Next review trigger

Re-run hard review after T711 and T755 provide:

1. causal scorer and behavior-policy parity tests;
2. repaired per-row question-type dispatch;
3. resolved trainer/generation/reward configuration;
4. actual PyTorch CoRL gradient, clipping, optimizer-membership, and
   save/resume instrumentation;
5. an audited task-private parameter mask or an explicit conclusion that the
   Janus/CoRL configuration offers no private-response mechanism.

