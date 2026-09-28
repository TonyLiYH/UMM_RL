"""Reusable Janus-Pro-R1 SFT/GRPO stack adapter (T720).

This package wraps the vendored upstream code under ``vendor/janus-pro-r1/``
(``janus-sft/``, ``janus-rl/``) with a thin, testable, env-var-driven layer:

- :mod:`comppareto.adapters.janus_pro_r1.env` -- path resolution for the
  vendored source tree, the base checkpoint, the reward-model checkpoint, and
  a ``require_cuda()`` guard (no silent CPU fallback for GPU-required code).
- :mod:`comppareto.adapters.janus_pro_r1.param_inventory` -- pure
  trainable/frozen parameter classification, independent of any specific
  model class (operates on any ``named_parameters()``-shaped iterable).
- :mod:`comppareto.adapters.janus_pro_r1.grpo_math` -- pure-numpy
  group-relative-advantage and policy-gradient loss, independent of torch.
- :mod:`comppareto.adapters.janus_pro_r1.sft_smoke` /
  :mod:`comppareto.adapters.janus_pro_r1.grpo_smoke` -- GPU-requiring smoke
  runners that import torch lazily (inside functions, not at module import
  time) so this package remains importable -- and its pure-logic modules
  remain unit-testable -- on a CPU-only development machine without torch
  installed.

Nothing in this package claims Janus-Pro-R1 is a CoRL reproduction (T720
frozen protocol); it is a standalone reusability/admission smoke for a
public third-party SFT+GRPO stack.
"""

from __future__ import annotations
