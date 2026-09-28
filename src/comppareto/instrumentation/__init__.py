"""Framework-neutral per-task GRPO gradient and optimizer-update instrumentation.

This package (T750) records, for a multi-task shared/private GRPO-style update:

- per-task shared gradients before clipping (norm, cosine, norm ratio, finite status);
- block-local statistics and overlap IDs (:mod:`comppareto.instrumentation.blocks`);
- PCGrad-ready vectors and MGDA convex-hull inputs (Gram matrix);
- pre/post-clipping norms and the realized clipping coefficient;
- AdamW moment summaries and realized parameter deltas (Delta theta);
- update-direction norm, near-zero rate, and directional derivatives;
- rollout, token, reward-call, backward, and wall-time counts.

The instrumentation core (:mod:`comppareto.instrumentation.recorder`) is a plain
PyTorch layer with no dependency on any specific training framework. The CoRL
adapter (:mod:`comppareto.instrumentation.corl_adapter`) specializes the core to
the CoRL-style shared/private multi-task interface without importing any CoRL
vendor code (which belongs to T710's separate ``allowed_paths``).
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
