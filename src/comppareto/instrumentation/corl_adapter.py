"""Framework-neutral -> CoRL adapter glue.

Everything in :mod:`comppareto.instrumentation.recorder`, ``.blocks``,
``.stats``, ``.params``, and ``.counters`` is framework-neutral: it only
assumes a set of :class:`~comppareto.instrumentation.params.Parameter`
objects grouped into a :class:`~comppareto.instrumentation.blocks.BlockRegistry`
and a model exposing a per-task ``forward_task`` method that computes and
accumulates that task's own gradient contribution. This module is the thin
CoRL-shaped adapter that drives one *combined update step* (sequential
per-task gradient computation at one shared-state hash, then one combined
clip + AdamW step) against any model satisfying :class:`CoRLTaskModel`,
recording every required measurement along the way.

The real CoRL/Janus-Pro model (T710's concern) and this mock stand-in
(:mod:`comppareto.instrumentation.mock_corl`) both satisfy this same
protocol, which is the point: this adapter does not know or care which one
it is driving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .blocks import BlockRegistry
from .counters import StepCounters
from .optim import AdamWOptimizer
from .params import Parameter
from .recorder import (
    ClippingRecord,
    CombinedGradientRecord,
    GradientUpdateRecorder,
    RealizedUpdateRecord,
    canonical_parameters,
    theta_snapshot,
)


class RolloutInfoLike(Protocol):
    rollout_count: int
    token_count: int
    reward_call_count: int


class CoRLTaskModel(Protocol):
    """The minimal shape a CoRL-style multi-task GRPO model must expose."""

    def block_registry(self) -> BlockRegistry: ...

    def forward_task(
        self, task_id: str, batch: object
    ) -> tuple[float, RolloutInfoLike]: ...


@dataclass
class CombinedStepResult:
    """Everything the acceptance gate needs to reconstruct one update step."""

    shared_state_hash: str
    task_ids: tuple[str, ...]
    per_task_statistics: dict[str, dict[str, object]]
    combined: CombinedGradientRecord
    clipping: ClippingRecord
    update: RealizedUpdateRecord
    counters: StepCounters
    theta_before: dict[str, np.ndarray]
    captured_task_gradients: dict[str, dict[str, np.ndarray]]
    losses: dict[str, float]


class CoRLGRPOAdapter:
    """Drives one combined-update step against a :class:`CoRLTaskModel`."""

    def __init__(self, model: CoRLTaskModel, optimizer: AdamWOptimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.registry = model.block_registry()
        self.canonical_parameters = canonical_parameters(self.registry)
        self.recorder = GradientUpdateRecorder(registry=self.registry)

    def shared_parameters(self) -> tuple[Parameter, ...]:
        params: list[Parameter] = []
        for block in self.registry.shared_blocks():
            params.extend(block.parameters)
        return tuple(params)

    def run_combined_step(
        self,
        task_ids: tuple[str, ...],
        batches: dict[str, object],
        *,
        max_grad_norm: float,
    ) -> CombinedStepResult:
        counters = StepCounters()
        shared_params = self.shared_parameters()
        state_hash = self.recorder.begin_batch(shared_params)
        theta_before = theta_snapshot(self.registry)
        moments_before = self.recorder.snapshot_pre_step_moments(self.optimizer)

        losses: dict[str, float] = {}
        for task_id in task_ids:
            self.recorder.check_shared_state(shared_params)
            self.optimizer.zero_grad(set_to_none=True)
            with counters.timed(f"forward_backward::{task_id}"):
                loss, info = self.model.forward_task(task_id, batches[task_id])
            losses[task_id] = loss
            counters.backward_count += 1
            counters.rollout_count += info.rollout_count
            counters.token_count += info.token_count
            counters.reward_call_count += info.reward_call_count
            self.recorder.capture_task_gradient(task_id)

        combined = self.recorder.combine()
        per_task_statistics = {
            task_id: self.recorder.task_statistics(task_id) for task_id in task_ids
        }
        clipping = self.recorder.apply_and_clip(
            self.canonical_parameters, max_norm=max_grad_norm
        )
        update = self.recorder.step_and_record(self.optimizer, theta_before, moments_before)

        return CombinedStepResult(
            shared_state_hash=state_hash,
            task_ids=task_ids,
            per_task_statistics=per_task_statistics,
            combined=combined,
            clipping=clipping,
            update=update,
            counters=counters,
            theta_before=theta_before,
            captured_task_gradients=self.recorder.captured_task_gradients(),
            losses=losses,
        )
