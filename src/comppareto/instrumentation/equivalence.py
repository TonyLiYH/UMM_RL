"""Instrumented-vs-uninstrumented equivalence proof (stage 3).

The pass/fail gate requires proving that adding instrumentation does not
change the numerical result of a combined update step. This module builds
two independently constructed ``(model, optimizer)`` pairs from an identical
initial state, drives them with bit-identical synthetic batches
(:func:`comppareto.instrumentation.mock_corl.make_batch` is a pure function
of its arguments, so calling it twice yields identical arrays), and
compares:

- the **baseline / uninstrumented** path: one ``zero_grad`` followed by each
  task's gradient computation accumulating straight into the shared
  parameters' ``.grad`` (exactly what an uninstrumented training loop
  summing a joint multi-task loss and calling one ``backward()`` would
  produce), then one ``clip`` and one ``optimizer.step()``; against
- the **instrumented** path:
  :meth:`~comppareto.instrumentation.corl_adapter.CoRLGRPOAdapter.run_combined_step`,
  which computes each task's gradient *separately* (zeroing in between),
  clones and captures it, manually sums the captured per-task gradients into
  the combined gradient, and only then clips and steps.

These two paths are mathematically equivalent (gradients are linear in the
sum of per-task losses), and this module measures the discrepancy directly
rather than assuming it. See ``runs/instrumentation-corl-v1/notes.md`` for
what the measured value turned out to be and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corl_adapter import CoRLGRPOAdapter
from .mock_corl import TASK_IDS, MockCoRLModel, make_batch
from .optim import AdamWOptimizer
from .recorder import canonical_parameters

ADAMW_LR = 1e-2
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0


def build_model_and_optimizer(seed: int) -> tuple[MockCoRLModel, AdamWOptimizer]:
    model = MockCoRLModel(seed=seed)
    optimizer = AdamWOptimizer(
        parameters=model.parameters(),
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
    )
    return model, optimizer


def _batches_for(batch_index: int, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        task_id: make_batch(task_id, batch_index=batch_index, seed=seed)
        for task_id in TASK_IDS
    }


def run_baseline_step(
    model: MockCoRLModel,
    optimizer: AdamWOptimizer,
    task_ids: tuple[str, ...],
    batches: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    max_grad_norm: float,
) -> np.ndarray:
    """One uninstrumented combined update step. Returns the pre-clip combined
    gradient, flattened in the model's canonical parameter order, for
    comparison against the instrumented path's
    :attr:`CombinedGradientRecord.combined_flat`.
    """
    registry = model.block_registry()
    parameters = canonical_parameters(registry)
    optimizer.zero_grad(set_to_none=True)
    for task_id in task_ids:
        model.forward_task(task_id, batches[task_id])
    pre_clip_flat = np.concatenate(
        [(p.grad if p.grad is not None else np.zeros_like(p.value)).reshape(-1) for p in parameters]
    )
    pre_clip_norm = float(np.linalg.norm(pre_clip_flat))
    clip_coefficient = min(max_grad_norm / (pre_clip_norm + 1e-6), 1.0)
    if clip_coefficient < 1.0:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * clip_coefficient
    optimizer.step()
    return pre_clip_flat


@dataclass
class EquivalenceStepResult:
    batch_index: int
    gradient_max_abs_error: float
    parameter_update_max_abs_error: float


@dataclass
class EquivalenceReport:
    steps: list[EquivalenceStepResult] = field(default_factory=list)

    @property
    def gradient_max_abs_error(self) -> float:
        return max((s.gradient_max_abs_error for s in self.steps), default=0.0)

    @property
    def parameter_update_max_abs_error(self) -> float:
        return max((s.parameter_update_max_abs_error for s in self.steps), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "gradient_max_abs_error": self.gradient_max_abs_error,
            "parameter_update_max_abs_error": self.parameter_update_max_abs_error,
            "steps": [
                {
                    "batch_index": s.batch_index,
                    "gradient_max_abs_error": s.gradient_max_abs_error,
                    "parameter_update_max_abs_error": s.parameter_update_max_abs_error,
                }
                for s in self.steps
            ],
        }


def run_equivalence_check(
    *, seed: int, num_batches: int, max_grad_norm: float = MAX_GRAD_NORM
) -> EquivalenceReport:
    """Run ``num_batches`` sequential combined update steps through both the
    baseline and instrumented paths, from identical initial weights, and
    report the per-step and worst-case discrepancy.
    """
    baseline_model, baseline_optimizer = build_model_and_optimizer(seed)
    instrumented_model, instrumented_optimizer = build_model_and_optimizer(seed)
    # Constructing two models from the same seed already yields identical
    # initial weights (mock_corl uses a local Generator, not global RNG
    # state), but assert it defensively so a future refactor cannot silently
    # break the equivalence proof's premise.
    for param_a, param_b in zip(baseline_model.parameters(), instrumented_model.parameters()):
        assert np.array_equal(param_a.value, param_b.value), "initial weights diverge"

    adapter = CoRLGRPOAdapter(instrumented_model, instrumented_optimizer)
    report = EquivalenceReport()

    for batch_index in range(num_batches):
        batches = _batches_for(batch_index, seed)
        baseline_pre_clip = run_baseline_step(
            baseline_model,
            baseline_optimizer,
            TASK_IDS,
            batches,
            max_grad_norm=max_grad_norm,
        )
        result = adapter.run_combined_step(TASK_IDS, batches, max_grad_norm=max_grad_norm)

        gradient_error = float(np.abs(baseline_pre_clip - result.combined.combined_flat).max())

        baseline_flat = np.concatenate(
            [p.value.reshape(-1) for p in canonical_parameters(baseline_model.block_registry())]
        )
        instrumented_flat = np.concatenate(
            [p.value.reshape(-1) for p in canonical_parameters(instrumented_model.block_registry())]
        )
        parameter_error = float(np.abs(baseline_flat - instrumented_flat).max())

        report.steps.append(
            EquivalenceStepResult(
                batch_index=batch_index,
                gradient_max_abs_error=gradient_error,
                parameter_update_max_abs_error=parameter_error,
            )
        )

    return report
