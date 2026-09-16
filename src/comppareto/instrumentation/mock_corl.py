"""Deterministic mock CoRL-compatible multi-task model (T750 stage 4).

This is *not* the real CoRL/Janus-Pro stack (that vendor code and its adapter
belong to T710's own ``allowed_paths``). It is a small, deterministic,
numpy-only stand-in that exposes exactly the shape a CoRL-style
understanding/generation GRPO model exposes: a shared backbone, a partially
shared "junction" block used by only some tasks, and per-task private heads
-- enough to exercise every measurement this task must instrument without
downloading or executing any real checkpoint.

Because this instrumentation layer is framework-neutral (no autograd engine
of any kind, see :mod:`comppareto.instrumentation.params`), gradients for
this tiny fixed network are hand-derived and computed exactly via ordinary
calculus (chain rule through affine + tanh layers, and an analytic
derivative of the GRPO advantage-weighted objective with respect to each
rollout's output) rather than traced by an autograd graph. For a network
this small the two approaches are equivalent; deriving it by hand is what
lets this module have zero third-party ML-framework dependencies.

Task layout (mirrors CoRL's actual U/G split plus one auxiliary task, so the
block registry has one fully-shared block, one *partially*-shared block, and
three private blocks -- i.e. a genuine partial-overlap case, not just
shared-vs-private):

- ``core_shared``  -- touched by every task (``understanding``, ``generation``,
  ``auxiliary``).
- ``ug_shared``    -- touched only by ``understanding`` and ``generation``
  (the CoRL-style understanding/generation junction); ``auxiliary`` bypasses it.
- ``private_understanding`` / ``private_generation`` / ``private_auxiliary``
  -- each touched by exactly one task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blocks import BlockRegistry, ParameterBlock
from .params import Parameter

TASK_IDS: tuple[str, ...] = ("understanding", "generation", "auxiliary")

CORE_DIM = 8
JUNCTION_DIM = 6
OUTPUT_DIM = 4
ROLLOUTS_PER_TASK = 4
TOKENS_PER_ROLLOUT = 16


def _linear_params(rng: np.random.Generator, out_dim: int, in_dim: int) -> tuple[Parameter, Parameter]:
    weight = Parameter(rng.normal(size=(out_dim, in_dim)).astype(np.float32) * 0.2)
    bias = Parameter(rng.normal(size=(out_dim,)).astype(np.float32) * 0.1)
    return weight, bias


def _linear_forward(weight: Parameter, bias: Parameter, x: np.ndarray) -> np.ndarray:
    return weight.value @ x + bias.value


def _linear_backward(
    weight: Parameter, bias: Parameter, grad_out: np.ndarray, cache_x: np.ndarray
) -> np.ndarray:
    weight.accumulate_grad(np.outer(grad_out, cache_x))
    bias.accumulate_grad(grad_out)
    return weight.value.T @ grad_out


def _tanh_forward(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_backward(grad_out: np.ndarray, tanh_x: np.ndarray) -> np.ndarray:
    return grad_out * (1.0 - tanh_x**2)


@dataclass
class RolloutInfo:
    rollout_count: int
    token_count: int
    reward_call_count: int


class MockCoRLModel:
    """A tiny deterministic shared/private multi-task GRPO-style model."""

    def __init__(self, *, seed: int = 20260916) -> None:
        rng = np.random.default_rng(seed)
        self.core_shared_w, self.core_shared_b = _linear_params(rng, CORE_DIM, CORE_DIM)
        self.ug_shared_w, self.ug_shared_b = _linear_params(rng, JUNCTION_DIM, CORE_DIM)
        self.private_understanding_w, self.private_understanding_b = _linear_params(
            rng, OUTPUT_DIM, JUNCTION_DIM
        )
        self.private_generation_w, self.private_generation_b = _linear_params(
            rng, OUTPUT_DIM, JUNCTION_DIM
        )
        self.private_auxiliary_w, self.private_auxiliary_b = _linear_params(
            rng, OUTPUT_DIM, CORE_DIM
        )

    def parameters(self) -> tuple[Parameter, ...]:
        return (
            self.core_shared_w,
            self.core_shared_b,
            self.ug_shared_w,
            self.ug_shared_b,
            self.private_understanding_w,
            self.private_understanding_b,
            self.private_generation_w,
            self.private_generation_b,
            self.private_auxiliary_w,
            self.private_auxiliary_b,
        )

    def block_registry(self) -> BlockRegistry:
        return BlockRegistry(
            blocks=(
                ParameterBlock(
                    "core_shared",
                    (self.core_shared_w, self.core_shared_b),
                    TASK_IDS,
                ),
                ParameterBlock(
                    "ug_shared",
                    (self.ug_shared_w, self.ug_shared_b),
                    ("understanding", "generation"),
                ),
                ParameterBlock(
                    "private_understanding",
                    (self.private_understanding_w, self.private_understanding_b),
                    ("understanding",),
                ),
                ParameterBlock(
                    "private_generation",
                    (self.private_generation_w, self.private_generation_b),
                    ("generation",),
                ),
                ParameterBlock(
                    "private_auxiliary",
                    (self.private_auxiliary_w, self.private_auxiliary_b),
                    ("auxiliary",),
                ),
            )
        )

    # -- manual forward / backward (see module docstring) --------------------

    def _rollout_forward(self, task_id: str, x: np.ndarray) -> tuple[np.ndarray, dict]:
        core_post = _tanh_forward(_linear_forward(self.core_shared_w, self.core_shared_b, x))
        if task_id == "understanding":
            junction_post = _tanh_forward(_linear_forward(self.ug_shared_w, self.ug_shared_b, core_post))
            output = _linear_forward(self.private_understanding_w, self.private_understanding_b, junction_post)
            return output, {"x": x, "core_post": core_post, "junction_post": junction_post}
        if task_id == "generation":
            junction_post = _tanh_forward(_linear_forward(self.ug_shared_w, self.ug_shared_b, core_post))
            output = _linear_forward(self.private_generation_w, self.private_generation_b, junction_post)
            return output, {"x": x, "core_post": core_post, "junction_post": junction_post}
        if task_id == "auxiliary":
            output = _linear_forward(self.private_auxiliary_w, self.private_auxiliary_b, core_post)
            return output, {"x": x, "core_post": core_post}
        raise ValueError(f"unknown task_id {task_id!r}")

    def _rollout_backward(self, task_id: str, grad_output: np.ndarray, cache: dict) -> None:
        if task_id == "understanding":
            d_junction_post = _linear_backward(
                self.private_understanding_w, self.private_understanding_b, grad_output, cache["junction_post"]
            )
            d_junction_pre = _tanh_backward(d_junction_post, cache["junction_post"])
            d_core_post = _linear_backward(self.ug_shared_w, self.ug_shared_b, d_junction_pre, cache["core_post"])
            d_core_pre = _tanh_backward(d_core_post, cache["core_post"])
            _linear_backward(self.core_shared_w, self.core_shared_b, d_core_pre, cache["x"])
            return
        if task_id == "generation":
            d_junction_post = _linear_backward(
                self.private_generation_w, self.private_generation_b, grad_output, cache["junction_post"]
            )
            d_junction_pre = _tanh_backward(d_junction_post, cache["junction_post"])
            d_core_post = _linear_backward(self.ug_shared_w, self.ug_shared_b, d_junction_pre, cache["core_post"])
            d_core_pre = _tanh_backward(d_core_post, cache["core_post"])
            _linear_backward(self.core_shared_w, self.core_shared_b, d_core_pre, cache["x"])
            return
        if task_id == "auxiliary":
            d_core_post = _linear_backward(
                self.private_auxiliary_w, self.private_auxiliary_b, grad_output, cache["core_post"]
            )
            d_core_pre = _tanh_backward(d_core_post, cache["core_post"])
            _linear_backward(self.core_shared_w, self.core_shared_b, d_core_pre, cache["x"])
            return
        raise ValueError(f"unknown task_id {task_id!r}")

    def forward_task(
        self,
        task_id: str,
        batch: tuple[np.ndarray, np.ndarray],
    ) -> tuple[float, RolloutInfo]:
        """Run ``ROLLOUTS_PER_TASK`` GRPO-style rollouts for one task,
        computing and accumulating this task's exact gradient contribution
        into the parameters it touches (via hand-derived backprop; see the
        module docstring), and returning the scalar loss value.

        ``batch`` is ``(inputs, targets)`` with a leading rollout-group
        dimension of size ``ROLLOUTS_PER_TASK``. A mock reward function
        (negative squared error to the target, one call per rollout)
        produces group-normalized advantages exactly as GRPO does; the loss
        is the advantage-weighted rollout objective.
        """
        inputs, targets = batch
        n = inputs.shape[0]
        outputs = []
        caches = []
        rewards = np.empty(n, dtype=np.float32)
        for i in range(n):
            output, cache = self._rollout_forward(task_id, inputs[i])
            outputs.append(output)
            caches.append(cache)
            rewards[i] = _mock_reward(output, targets[i])

        mean = rewards.mean()
        std = rewards.std()
        advantage = (rewards - mean) / (std + 1e-6)  # detached, as GRPO does
        loss = float(-np.mean(advantage * rewards))

        # d loss / d output_i = -(1/n) * advantage_i * d reward_i / d output_i
        #                      = -(1/n) * advantage_i * (-2 * (output_i - target_i))
        #                      = (2/n) * advantage_i * (output_i - target_i)
        for i in range(n):
            grad_output = (2.0 / n) * advantage[i] * (outputs[i] - targets[i])
            self._rollout_backward(task_id, grad_output.astype(np.float32), caches[i])

        info = RolloutInfo(
            rollout_count=n,
            token_count=n * TOKENS_PER_ROLLOUT,
            reward_call_count=n,
        )
        return loss, info


def _mock_reward(output: np.ndarray, target: np.ndarray) -> float:
    """Stand-in reward-model call: negative squared error to a fixed target."""
    return float(-np.sum((output - target) ** 2))


def make_batch(
    task_id: str, *, batch_index: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A pure, deterministic synthetic batch for one task at one batch index.

    Calling this twice with identical arguments always returns bit-identical
    arrays (a fresh local generator, no shared mutable RNG state), which is
    what lets the equivalence harness feed the exact same data to two
    independently constructed model/optimizer pairs.
    """
    task_offset = {"understanding": 1, "generation": 2, "auxiliary": 3}[task_id]
    local_seed = seed * 1_000_003 + batch_index * 97 + task_offset
    rng = np.random.default_rng(local_seed)
    inputs = rng.normal(size=(ROLLOUTS_PER_TASK, CORE_DIM)).astype(np.float32)
    targets = rng.normal(size=(ROLLOUTS_PER_TASK, OUTPUT_DIM)).astype(np.float32)
    return inputs, targets
