# Project charter

## Research problem

A unified multimodal model may perform image understanding with autoregressive cross-entropy and image generation with flow matching, diffusion preference optimization, or reward-based training. These tasks can share a backbone or semantic pathway while retaining private encoders, decoders, heads, samplers, rewards, and optimizer states.

Standard multi-task learning treats the losses as simultaneous functions of one shared parameter vector. That abstraction omits three defining features of unified multimodal post-training:

1. **Heterogeneous native updates:** task objectives and gradient estimators are not commensurate.
2. **Partial overlap:** each task touches a different subset of parameter blocks.
3. **Private compensation:** task-specific components can adapt after a shared update and absorb part of its cost.

## Primary hypothesis

Optimizing a local model of each task *after private adaptation* yields shared updates that preserve more single-task improvement and produce fewer negative-transfer capability slices than raw-gradient surgery or loss scalarization.

The decisive controlled question uses the same finite response \(A_i^K\) for every method: at equal compute, does retaining optimizer state and normalizing by attainable single-task gain improve prediction or joint optimization over general MOBLO/MGDA, normalized Chebyshev, and Nash negotiation? This prevents gains from being attributed merely to giving CompPareto extra inner updates.

Before promoting a response-aware method, the project must test the simpler
shared-then-private alternating baseline: select a shared direction from current
gradients, apply it virtually, then freeze the new shared state and run the same
finite private response granted to every method. A response-aware method is
scientifically justified only if it improves controlled post-adaptation
outcomes after matching private steps, samples, gradient evaluations, and wall
clock. The compute-matched private-only branch is mandatory so that ordinary
private training gains are not attributed to the shared update.

## Initial scope

- Modalities: text and image.
- Tasks: image understanding and text-to-image generation.
- Main paper setting: joint understanding-generation GRPO on Janus-Pro-1B,
  starting from the public CoRL training stack.
- SFT is used for implementation/headroom diagnostics; DPO is a bounded
  mechanism comparison; OPD is outside the main experimental scope.
- Training: post-training only; pretrained tokenizers and generative autoencoders remain frozen unless a named ablation changes this.
- Parameters: shared-backbone full-parameter updates are required in the main pilot; LoRA is an efficiency ablation, not the only setting.
- Models: Show-o2 for the first executable pilot; UniDDT and SenseNova-U1 for cross-architecture validation; UniAR as a more homogeneous-objective boundary control.

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

1. A value-function formulation for heterogeneous post-training with partial parameter overlap and task-private best responses.
2. A compensation-aware local surrogate based on implicit differentiation or finite unrolling.
3. A conditionally loss-scale-invariant max-min retained-gain negotiation objective and a deterministic common-descent certificate when one exists.
4. A diagnostic protocol testing whether the certificate predicts realized joint changes.
5. Evidence across architectures with shallow, deep, and more homogeneous sharing.

The optimizer contribution is staged: simultaneous raw training,
shared-then-private alternating optimization, private-then-shared commit
gradients, and only then explicit trajectory or reduced-space response models.
The simplest stage that survives compute-matched controls is the preferred
method.

The first four mathematical ingredients overlap with multi-objective bilevel optimization and personalized-learning literature. Until a new overlap-specific complexity or approximation theorem is proved, the intended novelty is the optimizer-state-aware UMM method, diagnostic evidence, and compute-matched protocol—not the generic existence of a bilevel value function.

## Falsification conditions

The core claim should be rejected or reframed if any of the following holds after the preregistered pilot:

- post-compensation diagnostics fail to improve prediction of realized two-task changes over raw gradients;
- CompPareto does not improve worst-task normalized gain over the best budget-matched scalarization;
- improvements disappear after equalizing optimizer steps, tokens, samples, and baseline search budgets;
- the method helps only one model or only one hand-picked capability slice;
- compute overhead makes the method infeasible relative to the measured benefit.
