"""Core, framework-neutral per-task gradient / optimizer-update recorder.

:class:`GradientUpdateRecorder` observes (and never mutates on its own
initiative) a set of :class:`~comppareto.instrumentation.params.Parameter`
blocks plus an :class:`~comppareto.instrumentation.optim.AdamWOptimizer`
while a caller drives one *combined update step*: several tasks' gradient
computations done sequentially at one shared parameter state, summed into a
combined gradient, clipped, and applied via a single ``optimizer.step()``.

The recorder never computes a task's gradient or calls ``optimizer.step()``
itself -- that stays under the caller's (adapter's or test's) control, which
is what lets the equivalence harness (:mod:`comppareto.instrumentation.equivalence`)
prove that recording is a read-only observation with no effect on the
numerical result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .blocks import BlockRegistry
from .optim import AdamWOptimizer
from .params import Parameter
from .stats import (
    cosine,
    directional_derivative,
    gram_matrix,
    is_finite,
    l2_norm,
    near_zero_rate,
    norm_ratio,
)


class RecorderProtocolError(RuntimeError):
    """Raised when the recorder's stage protocol is violated by the caller."""


def canonical_parameters(registry: BlockRegistry) -> tuple[Parameter, ...]:
    """The single canonical parameter order used for every flattened vector."""
    ordered: list[Parameter] = []
    for block in registry.blocks:
        ordered.extend(block.parameters)
    return tuple(ordered)


def shared_state_hash(parameters: tuple[Parameter, ...]) -> str:
    """A deterministic content hash of a set of parameter values.

    Used to prove that sequential per-task gradient computations within one
    combined-update step all observed the *identical* shared parameter state
    (stage 2's "sequential task batches at one shared-state hash").
    """
    digest = hashlib.sha256()
    for parameter in parameters:
        array = np.ascontiguousarray(parameter.value)
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class TaskGradientRecord:
    task_id: str
    per_block_norm: dict[str, float]
    shared_scope_vector: np.ndarray
    finite: bool


@dataclass
class CombinedGradientRecord:
    combined_by_block: dict[str, np.ndarray]
    combined_flat: np.ndarray
    pcgrad_vectors: dict[str, np.ndarray]
    gram_matrix: list[list[float]]
    shared_block_ids: tuple[str, ...]

    def combined_flat_shared_only(self) -> np.ndarray:
        parts = [self.combined_by_block[block_id] for block_id in self.shared_block_ids]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


@dataclass
class ClippingRecord:
    pre_clip_norm: float
    post_clip_norm: float
    clip_coefficient: float
    max_norm: float


@dataclass
class MomentSummary:
    initialized: bool
    step: int
    exp_avg_norm: float
    exp_avg_sq_norm: float


@dataclass
class RealizedUpdateRecord:
    delta_theta_by_block: dict[str, np.ndarray]
    delta_theta_flat: np.ndarray
    direction_norm: float
    near_zero_rate: float
    directional_derivative: float
    moments_before: dict[str, MomentSummary]
    moments_after: dict[str, MomentSummary]


@dataclass
class GradientUpdateRecorder:
    """Stateful observer for exactly one combined-update step.

    Call sequence: ``begin_batch`` once, then for each task
    ``check_shared_state`` + ``capture_task_gradient``, then ``combine``,
    ``apply_and_clip``, ``snapshot_pre_step_moments``, and finally
    ``step_and_record`` (which the caller invokes immediately around its own
    ``optimizer.step()`` call).
    """

    registry: BlockRegistry
    near_zero_epsilon: float = 1e-8
    _expected_hash: str | None = field(default=None, init=False, repr=False)
    _task_grads: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, init=False)
    _combined: CombinedGradientRecord | None = field(default=None, init=False, repr=False)
    _clipping: ClippingRecord | None = field(default=None, init=False, repr=False)

    # -- stage 2: sequential task batches at one shared-state hash ---------

    def begin_batch(self, shared_parameters: tuple[Parameter, ...]) -> str:
        self._expected_hash = shared_state_hash(shared_parameters)
        self._task_grads = {}
        self._combined = None
        self._clipping = None
        return self._expected_hash

    def check_shared_state(self, shared_parameters: tuple[Parameter, ...]) -> str:
        if self._expected_hash is None:
            raise RecorderProtocolError("begin_batch must be called before check_shared_state")
        current = shared_state_hash(shared_parameters)
        if current != self._expected_hash:
            raise RecorderProtocolError(
                "shared parameter state changed mid-batch: expected hash "
                f"{self._expected_hash} but observed {current}; a task's "
                "gradient computation must not mutate shared parameters "
                "before the combined optimizer step"
            )
        return current

    # -- per-task gradient capture ------------------------------------------

    def capture_task_gradient(self, task_id: str) -> TaskGradientRecord:
        if self._expected_hash is None:
            raise RecorderProtocolError("begin_batch must be called before capture_task_gradient")
        per_block_norm: dict[str, float] = {}
        per_block_array: dict[str, np.ndarray] = {}
        for block in self.registry.blocks_for_task(task_id):
            grads = []
            for parameter in block.parameters:
                grad = parameter.grad
                grads.append(
                    np.zeros_like(parameter.value) if grad is None else grad.copy()
                )
            flat = (
                np.concatenate([g.reshape(-1) for g in grads])
                if grads
                else np.zeros(0, dtype=np.float32)
            )
            per_block_array[block.block_id] = flat
            per_block_norm[block.block_id] = l2_norm(flat)
        shared_scope_vector = self._zero_padded_shared_vector(per_block_array)
        finite = all(is_finite(array) for array in per_block_array.values())
        self._task_grads[task_id] = per_block_array
        return TaskGradientRecord(
            task_id=task_id,
            per_block_norm=per_block_norm,
            shared_scope_vector=shared_scope_vector,
            finite=finite,
        )

    def _zero_padded_shared_vector(self, per_block_array: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for block in self.registry.shared_blocks():
            if block.block_id in per_block_array:
                parts.append(per_block_array[block.block_id])
            else:
                parts.append(np.zeros(block.param_count, dtype=np.float32))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def task_statistics(self, task_id: str) -> dict[str, object]:
        """Per-task norm/cosine/norm-ratio/finite status against the combined
        shared gradient. Must be called after :meth:`combine`.
        """
        if self._combined is None:
            raise RecorderProtocolError("combine must be called before task_statistics")
        vector = self._zero_padded_shared_vector(self._task_grads[task_id])
        combined = self._combined.combined_flat_shared_only()
        return {
            "task_id": task_id,
            "shared_norm": l2_norm(vector),
            "cosine_vs_combined": cosine(vector, combined),
            "norm_ratio_vs_combined": norm_ratio(vector, combined),
            "finite": all(is_finite(t) for t in self._task_grads[task_id].values()),
            "per_block_norm": {
                block_id: l2_norm(array) for block_id, array in self._task_grads[task_id].items()
            },
        }

    # -- combination, PCGrad vectors, MGDA Gram matrix ----------------------

    def combine(self) -> CombinedGradientRecord:
        combined_by_block: dict[str, np.ndarray] = {}
        for block in self.registry.blocks:
            contributions = [
                self._task_grads[task_id][block.block_id]
                for task_id in block.overlap_tasks
                if task_id in self._task_grads and block.block_id in self._task_grads[task_id]
            ]
            if not contributions:
                combined_by_block[block.block_id] = np.zeros(block.param_count, dtype=np.float32)
            else:
                total = contributions[0].copy()
                for contribution in contributions[1:]:
                    total = total + contribution
                combined_by_block[block.block_id] = total
        combined_flat = np.concatenate(
            [combined_by_block[block.block_id] for block in self.registry.blocks]
        )
        task_ids = self.registry.all_task_ids()
        pcgrad_vectors = {
            task_id: self._zero_padded_shared_vector(self._task_grads.get(task_id, {}))
            for task_id in task_ids
            if task_id in self._task_grads
        }
        vector_list = [pcgrad_vectors[task_id] for task_id in pcgrad_vectors]
        gram = gram_matrix(vector_list)
        record = CombinedGradientRecord(
            combined_by_block=combined_by_block,
            combined_flat=combined_flat,
            pcgrad_vectors=pcgrad_vectors,
            gram_matrix=gram,
            shared_block_ids=tuple(b.block_id for b in self.registry.shared_blocks()),
        )
        self._combined = record
        return record

    # -- clipping -------------------------------------------------------------

    def apply_and_clip(
        self,
        parameters: tuple[Parameter, ...],
        *,
        max_norm: float,
    ) -> ClippingRecord:
        if self._combined is None:
            raise RecorderProtocolError("combine must be called before apply_and_clip")
        offset = 0
        for parameter in parameters:
            n = parameter.numel()
            value = self._combined.combined_flat[offset : offset + n].reshape(parameter.shape)
            parameter.grad = value.copy()
            offset += n
        pre_clip_norm = l2_norm(
            np.concatenate([p.grad.reshape(-1) for p in parameters])
        )
        # Matches torch.nn.utils.clip_grad_norm_'s formula exactly: scale by
        # max_norm / (total_norm + eps), clamped to at most 1.0.
        clip_coefficient = min(max_norm / (pre_clip_norm + 1e-6), 1.0)
        if clip_coefficient < 1.0:
            for parameter in parameters:
                parameter.grad = parameter.grad * clip_coefficient
        post_clip_norm = l2_norm(
            np.concatenate([p.grad.reshape(-1) for p in parameters])
        )
        record = ClippingRecord(
            pre_clip_norm=pre_clip_norm,
            post_clip_norm=post_clip_norm,
            clip_coefficient=clip_coefficient,
            max_norm=max_norm,
        )
        self._clipping = record
        return record

    # -- optimizer moments and realized update -------------------------------

    def snapshot_pre_step_moments(self, optimizer: AdamWOptimizer) -> dict[str, MomentSummary]:
        return self._moment_summaries(optimizer)

    def step_and_record(
        self,
        optimizer: AdamWOptimizer,
        theta_before: dict[str, np.ndarray],
        moments_before: dict[str, MomentSummary],
    ) -> RealizedUpdateRecord:
        if self._clipping is None:
            raise RecorderProtocolError("apply_and_clip must be called before step_and_record")
        optimizer.step()
        delta_by_block: dict[str, np.ndarray] = {}
        for block in self.registry.blocks:
            deltas = []
            for parameter in block.parameters:
                before = theta_before[block.block_id + "::" + str(id(parameter))]
                after = parameter.value.copy()
                deltas.append((after - before).reshape(-1))
            delta_by_block[block.block_id] = (
                np.concatenate(deltas) if deltas else np.zeros(0, dtype=np.float32)
            )
        delta_flat = np.concatenate([delta_by_block[b.block_id] for b in self.registry.blocks])
        moments_after = self._moment_summaries(optimizer)
        record = RealizedUpdateRecord(
            delta_theta_by_block=delta_by_block,
            delta_theta_flat=delta_flat,
            direction_norm=l2_norm(delta_flat),
            near_zero_rate=near_zero_rate(delta_flat, epsilon=self.near_zero_epsilon),
            directional_derivative=directional_derivative(self._combined.combined_flat, delta_flat),
            moments_before=moments_before,
            moments_after=moments_after,
        )
        return record

    def _moment_summaries(self, optimizer: AdamWOptimizer) -> dict[str, MomentSummary]:
        summaries: dict[str, MomentSummary] = {}
        for block in self.registry.blocks:
            exp_avgs = []
            exp_avg_sqs = []
            step = 0
            initialized = True
            for parameter in block.parameters:
                state = optimizer.moment_state(parameter)
                if "exp_avg" not in state:
                    initialized = False
                    continue
                exp_avgs.append(state["exp_avg"].reshape(-1))
                exp_avg_sqs.append(state["exp_avg_sq"].reshape(-1))
                step = max(step, int(state.get("step", 0)))
            exp_avg_flat = np.concatenate(exp_avgs) if exp_avgs else np.zeros(0, dtype=np.float32)
            exp_avg_sq_flat = (
                np.concatenate(exp_avg_sqs) if exp_avg_sqs else np.zeros(0, dtype=np.float32)
            )
            summaries[block.block_id] = MomentSummary(
                initialized=initialized and bool(exp_avgs),
                step=step,
                exp_avg_norm=l2_norm(exp_avg_flat),
                exp_avg_sq_norm=l2_norm(exp_avg_sq_flat),
            )
        return summaries

    def captured_task_gradients(self) -> dict[str, dict[str, np.ndarray]]:
        """The raw per-task per-block gradient arrays captured this batch.

        Exposed so log-reconstruction (:mod:`comppareto.instrumentation.reconstruction`)
        can independently recompute the combined gradient purely from what
        was recorded, without reaching into private state.
        """
        return {task_id: dict(blocks) for task_id, blocks in self._task_grads.items()}


def theta_snapshot(registry: BlockRegistry) -> dict[str, np.ndarray]:
    """Snapshot every block's parameters, keyed for :meth:`step_and_record`."""
    snapshot: dict[str, np.ndarray] = {}
    for block in registry.blocks:
        for parameter in block.parameters:
            snapshot[block.block_id + "::" + str(id(parameter))] = parameter.value.copy()
    return snapshot
