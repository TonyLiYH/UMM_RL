"""Log-reconstruction checks (pass/fail gate: "logs must reconstruct combined
gradient and realized update").

Given the raw per-task captured gradients and the pre-step parameter
snapshot that :class:`~comppareto.instrumentation.corl_adapter.CombinedStepResult`
already carries, these functions independently recompute (a) the combined
gradient, purely as a sum of per-task per-block arrays, and (b) the realized
parameter update, purely as post-step-parameter minus the pre-step snapshot,
and compare each against what the recorder itself logged. A pass here proves
the logged records are sufficient to reconstruct both quantities from
scratch -- i.e. the instrumentation is not silently dropping information the
gate requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blocks import BlockRegistry
from .corl_adapter import CombinedStepResult


@dataclass
class ReconstructionReport:
    combined_gradient_max_abs_error: float
    combined_gradient_pass: bool
    realized_update_max_abs_error: float
    realized_update_pass: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "combined_gradient_max_abs_error": self.combined_gradient_max_abs_error,
            "combined_gradient_pass": self.combined_gradient_pass,
            "realized_update_max_abs_error": self.realized_update_max_abs_error,
            "realized_update_pass": self.realized_update_pass,
        }


def reconstruct_combined_gradient(
    registry: BlockRegistry,
    captured_task_gradients: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Recompute each block's combined gradient purely from the logged
    per-task per-block arrays (a sum over that block's overlap tasks).
    """
    reconstructed: dict[str, np.ndarray] = {}
    for block in registry.blocks:
        contributions = [
            captured_task_gradients[task_id][block.block_id]
            for task_id in block.overlap_tasks
            if task_id in captured_task_gradients
            and block.block_id in captured_task_gradients[task_id]
        ]
        if contributions:
            total = contributions[0].copy()
            for contribution in contributions[1:]:
                total = total + contribution
            reconstructed[block.block_id] = total
        else:
            reconstructed[block.block_id] = np.zeros(block.param_count, dtype=np.float32)
    return reconstructed


def reconstruct_realized_update(
    registry: BlockRegistry,
    theta_before: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Recompute each block's realized update purely as the current
    (post-step) parameter value minus the logged pre-step snapshot.
    """
    reconstructed: dict[str, np.ndarray] = {}
    for block in registry.blocks:
        deltas = []
        for parameter in block.parameters:
            before = theta_before[block.block_id + "::" + str(id(parameter))]
            deltas.append((parameter.value - before).reshape(-1))
        reconstructed[block.block_id] = (
            np.concatenate(deltas) if deltas else np.zeros(0, dtype=np.float32)
        )
    return reconstructed


def check_reconstruction(
    registry: BlockRegistry,
    result: CombinedStepResult,
    *,
    tolerance: float = 1e-6,
) -> ReconstructionReport:
    reconstructed_gradient = reconstruct_combined_gradient(
        registry, result.captured_task_gradients
    )
    gradient_errors = [
        float(
            np.abs(
                reconstructed_gradient[block_id] - result.combined.combined_by_block[block_id]
            ).max()
        )
        for block_id in reconstructed_gradient
    ]
    gradient_max_error = max(gradient_errors, default=0.0)

    reconstructed_update = reconstruct_realized_update(registry, result.theta_before)
    update_errors = [
        float(
            np.abs(
                reconstructed_update[block_id] - result.update.delta_theta_by_block[block_id]
            ).max()
        )
        for block_id in reconstructed_update
    ]
    update_max_error = max(update_errors, default=0.0)

    return ReconstructionReport(
        combined_gradient_max_abs_error=gradient_max_error,
        combined_gradient_pass=gradient_max_error <= tolerance,
        realized_update_max_abs_error=update_max_error,
        realized_update_pass=update_max_error <= tolerance,
    )
