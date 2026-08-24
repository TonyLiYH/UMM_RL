# Decision 0001: validate image understanding and image generation first

- Date: 2026-08-24
- Status: accepted

## Decision

The first paper claim targets joint post-training of image understanding and text-to-image generation. Video, audio, editing, and pretraining are excluded until the core diagnostic and optimization claims pass.

## Rationale

This pair already contains the essential heterogeneity: autoregressive semantic objectives, flow/diffusion objectives, different output lengths and gradient scales, partial component sharing, and private decoding paths. It is sufficient to test the method without confounding the first result with modality-specific engineering.

## Consequence

The method is formulated for more than two tasks, but all early numerical and model experiments must work for the two-task case and expose a closed-form or one-dimensional negotiation diagnostic where possible.

