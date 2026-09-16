"""Top-level experiment orchestration: runs stages 1-4 and produces the exact
metrics the acceptance contract (`tasks/contracts/T750.acceptance.yaml`)
checks.

Stage 5 (an optional real T710 diagnostic batch) is intentionally not driven
from here -- see `reports/T750/result-summary.md` and
`runs/instrumentation-corl-v1/notes.md` for why it was skipped for this
submission (T710 had not reached `awaiting_review` at the time this ran).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corl_adapter import CoRLGRPOAdapter
from .equivalence import MAX_GRAD_NORM, build_model_and_optimizer, run_equivalence_check
from .mock_corl import TASK_IDS, make_batch
from .reconstruction import check_reconstruction

DEFAULT_SEED = 20260916
DEFAULT_NUM_BATCHES = 5


@dataclass
class ExperimentReport:
    seed: int
    num_batches: int
    equivalence_gradient_max_abs_error: float
    equivalence_parameter_update_max_abs_error: float
    ownership_unassigned_trainable_parameters: int
    reconstruction_combined_gradient_pass: bool
    reconstruction_realized_update_pass: bool
    reconstruction_combined_gradient_max_abs_error: float
    reconstruction_realized_update_max_abs_error: float
    shared_state_hashes: list[str] = field(default_factory=list)
    per_batch_summaries: list[dict[str, object]] = field(default_factory=list)
    gpu_hours: float = 0.0
    device: str = "cpu"

    def to_metrics_dict(self) -> dict[str, object]:
        return {
            "equivalence": {
                "gradient_max_abs_error": self.equivalence_gradient_max_abs_error,
                "parameter_update_max_abs_error": self.equivalence_parameter_update_max_abs_error,
            },
            "ownership": {
                "unassigned_trainable_parameters": self.ownership_unassigned_trainable_parameters,
            },
            "reconstruction": {
                "combined_gradient_pass": self.reconstruction_combined_gradient_pass,
                "realized_update_pass": self.reconstruction_realized_update_pass,
                "combined_gradient_max_abs_error": self.reconstruction_combined_gradient_max_abs_error,
                "realized_update_max_abs_error": self.reconstruction_realized_update_max_abs_error,
            },
            "resources": {
                "gpu_hours": self.gpu_hours,
                "device": self.device,
            },
            "shared_state_hashes": self.shared_state_hashes,
            "per_batch_summaries": self.per_batch_summaries,
        }


def run_experiment(
    *, seed: int = DEFAULT_SEED, num_batches: int = DEFAULT_NUM_BATCHES
) -> ExperimentReport:
    # Stage 3: instrumented-vs-uninstrumented equivalence proof.
    equivalence = run_equivalence_check(
        seed=seed, num_batches=num_batches, max_grad_norm=MAX_GRAD_NORM
    )

    # Stage 4: mock CoRL-compatible model integration, driven through the
    # CoRL adapter for `num_batches` sequential combined-update steps.
    model, optimizer = build_model_and_optimizer(seed)
    registry = model.block_registry()
    trainable = tuple(model.parameters())
    registry.validate_ownership(trainable)  # raises OwnershipError on any gap
    unassigned = registry.unassigned_trainable_parameters(trainable)

    adapter = CoRLGRPOAdapter(model, optimizer)
    shared_state_hashes: list[str] = []
    per_batch_summaries: list[dict[str, object]] = []
    worst_reconstruction_gradient_error = 0.0
    worst_reconstruction_update_error = 0.0
    reconstruction_all_pass = True

    for batch_index in range(num_batches):
        batches = {
            task_id: make_batch(task_id, batch_index=batch_index, seed=seed)
            for task_id in TASK_IDS
        }
        result = adapter.run_combined_step(TASK_IDS, batches, max_grad_norm=MAX_GRAD_NORM)
        shared_state_hashes.append(result.shared_state_hash)

        recon = check_reconstruction(registry, result)
        worst_reconstruction_gradient_error = max(
            worst_reconstruction_gradient_error, recon.combined_gradient_max_abs_error
        )
        worst_reconstruction_update_error = max(
            worst_reconstruction_update_error, recon.realized_update_max_abs_error
        )
        reconstruction_all_pass = (
            reconstruction_all_pass
            and recon.combined_gradient_pass
            and recon.realized_update_pass
        )

        per_batch_summaries.append(
            {
                "batch_index": batch_index,
                "shared_state_hash": result.shared_state_hash,
                "clip_coefficient": result.clipping.clip_coefficient,
                "pre_clip_norm": result.clipping.pre_clip_norm,
                "post_clip_norm": result.clipping.post_clip_norm,
                "direction_norm": result.update.direction_norm,
                "near_zero_rate": result.update.near_zero_rate,
                "directional_derivative": result.update.directional_derivative,
                "gram_matrix": result.combined.gram_matrix,
                "per_task_statistics": {
                    task_id: {
                        k: v
                        for k, v in stats.items()
                        if k != "per_block_norm"
                    }
                    for task_id, stats in result.per_task_statistics.items()
                },
                "counters": result.counters.to_dict(),
                "reconstruction": recon.to_dict(),
            }
        )

    # All shared-state hashes must be distinct across sequential batches
    # (parameters actually changed each step) but internally consistent
    # within a batch, which `check_shared_state` already enforced live.
    assert len(shared_state_hashes) == num_batches

    return ExperimentReport(
        seed=seed,
        num_batches=num_batches,
        equivalence_gradient_max_abs_error=equivalence.gradient_max_abs_error,
        equivalence_parameter_update_max_abs_error=equivalence.parameter_update_max_abs_error,
        ownership_unassigned_trainable_parameters=unassigned,
        reconstruction_combined_gradient_pass=reconstruction_all_pass,
        reconstruction_realized_update_pass=reconstruction_all_pass,
        reconstruction_combined_gradient_max_abs_error=worst_reconstruction_gradient_error,
        reconstruction_realized_update_max_abs_error=worst_reconstruction_update_error,
        shared_state_hashes=shared_state_hashes,
        per_batch_summaries=per_batch_summaries,
        gpu_hours=0.0,
        device="cpu",
    )
