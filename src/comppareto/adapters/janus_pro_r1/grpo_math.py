"""Pure-numpy GRPO group-relative-advantage and policy-gradient loss (T720).

Scope note (deliberate, documented reduction -- see
``configs/janus-pro-r1/admission/environment-rl-lock.md`` and
``reports/T720/claim-check.md``): the T720 GRPO smoke performs exactly one
optimizer step per rollout batch, matching upstream's own per-step
generate-then-train loop
(``vendor/janus-pro-r1/janus-rl/src/open_r1/grpo_t2i.py``). Because the
policy that generated the rollout is *by construction* the same policy the
loss is computed under (no stale behavior policy from a replay buffer, no
multiple inner epochs over one rollout batch), the PPO-style
probability-ratio ``pi_theta(a|s) / pi_theta_old(a|s)`` is exactly 1 for
every token on the very step it is computed -- there is nothing for a
ratio-clip to clip. The loss below is therefore the group-relative-advantage
REINFORCE estimator with no clipping term and, as documented, no separate
frozen-reference-model KL penalty (that omission trades off exactly the
~14.84GB a second co-resident bf16 7B policy copy would cost against the
smoke's memory budget, and is reported as a scope reduction, not hidden).

Kept independent of torch: all math here is plain ``numpy`` so the formulas
themselves are unit-testable against hand-computed values on a CPU-only,
torch-free development machine. The real GPU smoke runner
(:mod:`comppareto.adapters.janus_pro_r1.grpo_smoke`) converts
torch tensors to/from numpy at the boundary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def group_relative_advantage(rewards: NDArray[np.floating], *, eps: float = 1e-6) -> NDArray[np.floating]:
    """Per-group standardized advantage: ``(r_i - mean(group)) / (std(group) + eps)``.

    ``rewards`` has shape ``(n_prompts, n_generations)`` -- one row per
    prompt, one column per sampled completion for that prompt (upstream's
    ``num_generations``). The mean/std are computed *within* each row
    (``axis=-1``), matching upstream's own GRPO group-relative baseline: a
    completion's advantage is relative only to the other completions sampled
    for the *same* prompt, never across prompts.

    ``eps`` guards the degenerate case where every completion in a group
    received an identical reward (std == 0) -- without it the advantage
    would be ``0/0 = nan`` instead of the correct, honest answer "no signal
    in this group", which is ``0`` (numerator is also 0 in that case).
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (n_prompts, n_generations); got shape {rewards.shape}")
    mean = rewards.mean(axis=-1, keepdims=True)
    std = rewards.std(axis=-1, keepdims=True)
    return (rewards - mean) / (std + eps)


def policy_gradient_loss(
    log_probs: NDArray[np.floating],
    advantages: NDArray[np.floating],
) -> float:
    """Group-relative-advantage REINFORCE loss (no ratio-clip, no KL term).

    ``loss = -mean(log_probs * advantages)`` over every (prompt, generation)
    entry, i.e. gradient ascent on ``advantage``-weighted log-likelihood
    implemented as gradient *descent* on this scalar (the sign upstream's
    own trainer and every REINFORCE-family estimator uses).

    Both arguments must have identical shape ``(n_prompts, n_generations)``
    -- one scalar (summed-over-tokens) log-probability and one advantage per
    sampled completion. Per-token log-probabilities should already have been
    summed by the caller before this function is invoked; this function
    performs no sequence-length normalization decision of its own (that
    choice -- sum vs. mean over tokens -- belongs to the caller, matching
    upstream's own token-sum convention in ``grpo_t2i.py``).
    """

    log_probs = np.asarray(log_probs, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    if log_probs.shape != advantages.shape:
        raise ValueError(
            f"log_probs and advantages must have the same shape; got {log_probs.shape} vs {advantages.shape}"
        )
    return float(-np.mean(log_probs * advantages))
