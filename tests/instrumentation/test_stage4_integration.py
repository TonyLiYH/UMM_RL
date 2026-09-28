"""Stage 4: mock CoRL-compatible model integration via the adapter and the
top-level experiment orchestrator, plus log-reconstruction correctness.
"""

from __future__ import annotations

import numpy as np

from comppareto.instrumentation.corl_adapter import CoRLGRPOAdapter
from comppareto.instrumentation.equivalence import build_model_and_optimizer
from comppareto.instrumentation.experiment import run_experiment
from comppareto.instrumentation.mock_corl import TASK_IDS, make_batch
from comppareto.instrumentation.reconstruction import check_reconstruction

GRADIENT_TOLERANCE = 1.0e-6
PARAMETER_UPDATE_TOLERANCE = 1.0e-6


def test_adapter_run_combined_step_reconstructs_cleanly() -> None:
    model, optimizer = build_model_and_optimizer(seed=555)
    registry = model.block_registry()
    adapter = CoRLGRPOAdapter(model, optimizer)
    batches = {
        task_id: make_batch(task_id, batch_index=0, seed=555) for task_id in TASK_IDS
    }
    result = adapter.run_combined_step(TASK_IDS, batches, max_grad_norm=1.0)

    assert result.task_ids == TASK_IDS
    assert set(result.losses) == set(TASK_IDS)
    for loss in result.losses.values():
        assert np.isfinite(loss)

    # Ownership: every trainable parameter belongs to exactly one block.
    registry.validate_ownership(model.parameters())
    assert registry.unassigned_trainable_parameters(model.parameters()) == 0

    recon = check_reconstruction(registry, result)
    assert recon.combined_gradient_pass
    assert recon.realized_update_pass
    assert recon.combined_gradient_max_abs_error <= GRADIENT_TOLERANCE
    assert recon.realized_update_max_abs_error <= PARAMETER_UPDATE_TOLERANCE


def test_adapter_sequential_batches_change_shared_state_each_step() -> None:
    model, optimizer = build_model_and_optimizer(seed=777)
    adapter = CoRLGRPOAdapter(model, optimizer)
    hashes = []
    for batch_index in range(3):
        batches = {
            task_id: make_batch(task_id, batch_index=batch_index, seed=777)
            for task_id in TASK_IDS
        }
        result = adapter.run_combined_step(TASK_IDS, batches, max_grad_norm=1.0)
        hashes.append(result.shared_state_hash)
    # Shared parameters genuinely change after every combined AdamW step.
    assert len(set(hashes)) == len(hashes)


def test_run_experiment_end_to_end_metrics() -> None:
    report = run_experiment(seed=20260916, num_batches=5)
    metrics = report.to_metrics_dict()

    assert metrics["equivalence"]["gradient_max_abs_error"] <= GRADIENT_TOLERANCE
    assert metrics["equivalence"]["parameter_update_max_abs_error"] <= PARAMETER_UPDATE_TOLERANCE
    assert metrics["ownership"]["unassigned_trainable_parameters"] == 0
    assert metrics["reconstruction"]["combined_gradient_pass"] is True
    assert metrics["reconstruction"]["realized_update_pass"] is True
    assert metrics["resources"]["gpu_hours"] <= 1
    assert metrics["resources"]["device"] == "cpu"
    assert len(metrics["shared_state_hashes"]) == 5
    assert len(set(metrics["shared_state_hashes"])) == 5
    assert len(metrics["per_batch_summaries"]) == 5


def test_run_experiment_is_reproducible() -> None:
    report_a = run_experiment(seed=42, num_batches=2)
    report_b = run_experiment(seed=42, num_batches=2)
    assert report_a.shared_state_hashes == report_b.shared_state_hashes
    assert (
        report_a.equivalence_gradient_max_abs_error
        == report_b.equivalence_gradient_max_abs_error
    )
