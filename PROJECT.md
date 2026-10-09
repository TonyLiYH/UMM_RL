# Project charter

## Research problem

A unified multimodal model may perform image understanding with autoregressive cross-entropy and image generation with flow matching, diffusion preference optimization, or reward-based training. These tasks can share a backbone or semantic pathway while retaining private encoders, decoders, heads, samplers, rewards, and optimizer states.

Standard multi-task learning treats the losses as simultaneous functions of one shared parameter vector. That abstraction omits three defining features of unified multimodal post-training:

1. **Heterogeneous native updates:** task objectives and gradient estimators are not commensurate.
2. **Partial overlap:** each task touches a different subset of parameter blocks.
3. **Private compensation:** task-specific components can adapt after a shared update and absorb part of its cost.

## Primary hypothesis

The UMM post-training programme now centers on **shared parameter ownership
and plasticity**: intervention-defined groups may selectively support image
understanding, image generation, both, or neither under a specified test.
The project tests whether this map predicts pruning damage and cross-task
effects of restricted post-training, and whether acting on it improves joint
outcomes at matched compute. The analysis units, definitions, controls, and
decision gates are in
[`docs/plans/2026-10-09-umm-parameter-ownership.md`](docs/plans/2026-10-09-umm-parameter-ownership.md).

Janus-Pro-1B is the first base model; the CoRL framework and data support the
main joint-GRPO setting. A faithful, semantically admitted CoRL reproduction
remains a P0 objective before ownership-guided training claims.

### Earlier optimizer hypothesis (conditional follow-up)

Optimizing a local model of each task *after private adaptation* yields shared updates that preserve more single-task improvement and produce fewer negative-transfer capability slices than raw-gradient surgery or loss scalarization.

The decisive controlled question uses the same finite response \(A_i^K\) for every method: at equal compute, does response information identify shared updates whose incremental value is misjudged by current-state gradients, and does acting on it improve independently evaluated joint outcomes beyond tuned alternating optimization and a strong conventional negotiator? This prevents gains from being attributed merely to extra private updates, line search, or a different evaluation budget.

Before promoting a response-aware method, the project must test the simpler
shared-then-private alternating baseline: select a shared direction from current
gradients, apply it virtually, then freeze the new shared state and run the same
finite private response granted to every method. A response-aware method is
scientifically justified only if it improves controlled post-adaptation
outcomes after matching private steps, samples, gradient evaluations, and wall
clock. The compute-matched private-only branch is mandatory so that ordinary
private training gains are not attributed to the shared update.

For GRPO response diagnostics, trajectories, behavior log probabilities, masks,
advantages, evaluator state, private optimizer state, and RNG must remain
fixed within every counterfactual window. Persistent training refreshes
on-policy rollouts between windows. No finite-response theorem is applied to a
surrogate whose behavior anchors are recomputed after every candidate
perturbation.

## Initial scope

- Modalities: text and image.
- Tasks: image understanding and text-to-image generation.
- Main paper setting: joint understanding-generation GRPO on Janus-Pro-1B,
  starting from the public CoRL training stack.
- SFT is used for implementation/headroom diagnostics; DPO is a bounded
  mechanism comparison; OPD is outside the main experimental scope.
- Training: post-training only; pretrained tokenizers and generative autoencoders remain frozen unless a named ablation changes this.
- Parameters: shared-backbone full-parameter updates are required in the main pilot; LoRA is an efficiency ablation, not the only setting.
- Models: Janus-Pro-1B is the first ownership-mapping and joint-training model;
  a second admitted UMM is a later generality check. Show-o2, UniDDT,
  SenseNova-U1, and UniAR retain their earlier admission evidence for
  conditional optimizer or architecture studies.

The formal cross-architecture starting checkpoint is selected separately from
inference admission. Prefer a checkpoint that already supports understanding
and generation but precedes the target preference/RL stage; an SFT checkpoint
is normally preferable to either an unusable raw base model or a final
preference-optimized model. SenseNova-U1-SFT is the accepted transfer
candidate; Show-o2 remains an engineering diagnostic.

The first empirical route uses Janus-Pro-1B because CoRL provides a public
successful Unified-GRPO baseline and paired data. The project first reproduces
that path, establishes single-task oracles, and instruments per-task gradients
and realized optimizer updates. Traditional negotiators and the response-aware
method open only after those controls pass.

## Out of scope until the core claim passes

- Pretraining a unified model from scratch.
- Video, audio, and action modalities.
- Building a new multimodal tokenizer.
- Claiming one optimizer is universally optimal for every task mixture.
- Reproducing every benchmark reported by each base model.

## Intended contributions

1. A reproducible, intervention-defined map of structured shared groups for
   image understanding and generation, with uncertainty and unresolved cases.
2. A test of whether that map predicts held-out pruning damage and the
   direction of cross-task effects under restricted post-training.
3. A matched-budget decision rule for selecting trainable groups, promoted
   only if it improves a prespecified joint criterion over random and
   conventional-importance controls.
4. A later cross-architecture test of the decisive mechanism, if the first
   model shows a reproducible effect.

The earlier compensation-aware value-function, finite-response, negotiation,
and certificate contributions remain a separate conditional optimizer
programme. They require a measured interference pattern and a new comparison
gate before becoming the paper's method claim.

## Falsification conditions

The ownership claim should be rejected or reframed if group effects are
unstable across disjoint calibration splits, fail to predict held-out pruning
or training effects better than the frozen controls, or lose any joint benefit
after matching parameter count, task exposure, and compute. A zero-gate
dependency that does not predict learning plasticity remains a useful
mechanistic result, but it does not support an ownership-guided training
method. The conditional optimizer programme keeps its own earlier
falsification criteria in its task contracts.
