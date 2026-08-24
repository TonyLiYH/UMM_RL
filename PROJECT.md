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

## Initial scope

- Modalities: text and image.
- Tasks: image understanding and text-to-image generation.
- Training: post-training only; pretrained tokenizers and generative autoencoders remain frozen unless a named ablation changes this.
- Parameters: shared-backbone full-parameter updates are required in the main pilot; LoRA is an efficiency ablation, not the only setting.
- Models: Show-o2 for the first executable pilot; UniDDT and SenseNova-U1 for cross-architecture validation; UniAR as a more homogeneous-objective boundary control.

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

The first four mathematical ingredients overlap with multi-objective bilevel optimization and personalized-learning literature. Until a new overlap-specific complexity or approximation theorem is proved, the intended novelty is the optimizer-state-aware UMM method, diagnostic evidence, and compute-matched protocol—not the generic existence of a bilevel value function.

## Falsification conditions

The core claim should be rejected or reframed if any of the following holds after the preregistered pilot:

- post-compensation diagnostics fail to improve prediction of realized two-task changes over raw gradients;
- CompPareto does not improve worst-task normalized gain over the best budget-matched scalarization;
- improvements disappear after equalizing optimizer steps, tokens, samples, and baseline search budgets;
- the method helps only one model or only one hand-picked capability slice;
- compute overhead makes the method infeasible relative to the measured benefit.
