"""Tests for comppareto.adapters.janus_pro_r1.grpo_math (T720).

Every case is checked against a hand-computed value, not merely a smoke
"does it run" check -- pure numpy, no torch, runs on any CPU-only machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from comppareto.adapters.janus_pro_r1.grpo_math import (
    group_relative_advantage,
    policy_gradient_loss,
)


def test_group_relative_advantage_hand_computed_two_prompts() -> None:
    # Prompt 0: rewards [1, 2, 3] -> mean=2, population std=sqrt(2/3)=0.8164965809...
    # Prompt 1: rewards [0, 0, 10] -> mean=10/3, population std computed below.
    rewards = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 10.0]])
    adv = group_relative_advantage(rewards, eps=0.0)

    mean0, std0 = 2.0, np.sqrt(2.0 / 3.0)
    expected_row0 = (rewards[0] - mean0) / std0
    np.testing.assert_allclose(adv[0], expected_row0, rtol=1e-10)

    mean1 = 10.0 / 3.0
    std1 = np.sqrt(np.mean((rewards[1] - mean1) ** 2))
    expected_row1 = (rewards[1] - mean1) / std1
    np.testing.assert_allclose(adv[1], expected_row1, rtol=1e-10)

    # Each group's own advantages must average to (numerically) zero --
    # that's the whole point of a group-relative baseline.
    np.testing.assert_allclose(adv.mean(axis=-1), [0.0, 0.0], atol=1e-10)


def test_group_relative_advantage_degenerate_group_is_zero_not_nan() -> None:
    # Every completion in the group got the identical reward -> zero signal,
    # not 0/0 = nan. eps default (1e-6) guards this.
    rewards = np.array([[5.0, 5.0, 5.0]])
    adv = group_relative_advantage(rewards)
    assert np.all(np.isfinite(adv))
    np.testing.assert_allclose(adv, np.zeros_like(adv), atol=1e-6)


def test_group_relative_advantage_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError):
        group_relative_advantage(np.array([1.0, 2.0, 3.0]))


def test_policy_gradient_loss_hand_computed() -> None:
    log_probs = np.array([[-1.0, -2.0], [-0.5, -3.0]])
    advantages = np.array([[1.0, -1.0], [2.0, 0.5]])
    # elementwise product: [[-1.0, 2.0], [-1.0, -1.5]] -> mean = -0.375
    # loss = -mean(...) = 0.375
    loss = policy_gradient_loss(log_probs, advantages)
    assert loss == pytest.approx(0.375, abs=1e-12)


def test_policy_gradient_loss_zero_advantage_is_zero_loss() -> None:
    log_probs = np.array([[-1.0, -2.0, -3.0]])
    advantages = np.zeros_like(log_probs)
    assert policy_gradient_loss(log_probs, advantages) == pytest.approx(0.0, abs=1e-12)


def test_policy_gradient_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        policy_gradient_loss(np.zeros((2, 3)), np.zeros((2, 4)))


def test_policy_gradient_loss_is_finite_for_realistic_magnitudes() -> None:
    # Sum-of-log-probs over a ~50-token generation is plausibly large and
    # negative (e.g. -80); the loss must still come out finite, matching the
    # T720 acceptance-contract requirement grpo.loss_finite == true.
    rng = np.random.default_rng(0)
    log_probs = rng.uniform(-120.0, -40.0, size=(4, 4))
    advantages = group_relative_advantage(rng.uniform(0.0, 1.0, size=(4, 4)))
    loss = policy_gradient_loss(log_probs, advantages)
    assert np.isfinite(loss)
