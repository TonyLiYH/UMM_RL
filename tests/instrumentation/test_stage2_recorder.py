"""Stage 2: sequential task batches at one shared-state hash, plus the
recorder's block-local statistics, overlap IDs, PCGrad vectors, MGDA Gram
matrix, clipping math, and AdamW moment summaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from comppareto.instrumentation.mock_corl import TASK_IDS, MockCoRLModel, make_batch
from comppareto.instrumentation.optim import AdamWOptimizer
from comppareto.instrumentation.recorder import (
    GradientUpdateRecorder,
    RecorderProtocolError,
    canonical_parameters,
    shared_state_hash,
    theta_snapshot,
)


def _build() -> tuple[MockCoRLModel, AdamWOptimizer, GradientUpdateRecorder]:
    model = MockCoRLModel(seed=99)
    registry = model.block_registry()
    optimizer = AdamWOptimizer(parameters=model.parameters(), lr=0.05)
    recorder = GradientUpdateRecorder(registry=registry)
    return model, optimizer, recorder


def test_shared_state_hash_is_stable_and_content_sensitive() -> None:
    model = MockCoRLModel(seed=1)
    params = canonical_parameters(model.block_registry())
    hash_a = shared_state_hash(params)
    hash_b = shared_state_hash(params)
    assert hash_a == hash_b
    params[0].value = params[0].value + 1.0
    hash_c = shared_state_hash(params)
    assert hash_c != hash_a


def test_sequential_task_batches_observe_one_shared_state_hash() -> None:
    model, optimizer, recorder = _build()
    registry = model.block_registry()
    shared_params = tuple(p for b in registry.shared_blocks() for p in b.parameters)
    expected_hash = recorder.begin_batch(shared_params)

    for task_id in TASK_IDS:
        # Every task in this batch must observe the identical shared state.
        observed = recorder.check_shared_state(shared_params)
        assert observed == expected_hash
        optimizer.zero_grad(set_to_none=True)
        batch = make_batch(task_id, batch_index=0, seed=99)
        model.forward_task(task_id, batch)
        recorder.capture_task_gradient(task_id)

    # Shared parameters must not have moved during the whole sequential pass.
    assert recorder.check_shared_state(shared_params) == expected_hash


def test_check_shared_state_detects_mutation_mid_batch() -> None:
    model, optimizer, recorder = _build()
    registry = model.block_registry()
    shared_params = tuple(p for b in registry.shared_blocks() for p in b.parameters)
    recorder.begin_batch(shared_params)
    # Mutate a shared parameter mid-batch (simulating a protocol violation).
    shared_params[0].value = shared_params[0].value + 1.0
    with pytest.raises(RecorderProtocolError):
        recorder.check_shared_state(shared_params)


def test_capture_before_begin_batch_raises() -> None:
    _model, _optimizer, recorder = _build()
    with pytest.raises(RecorderProtocolError):
        recorder.capture_task_gradient("understanding")


def test_recorder_block_local_stats_and_overlap_ids() -> None:
    model, optimizer, recorder = _build()
    registry = model.block_registry()
    shared_params = tuple(p for b in registry.shared_blocks() for p in b.parameters)
    recorder.begin_batch(shared_params)

    for task_id in TASK_IDS:
        optimizer.zero_grad(set_to_none=True)
        batch = make_batch(task_id, batch_index=0, seed=99)
        model.forward_task(task_id, batch)
        record = recorder.capture_task_gradient(task_id)
        assert record.finite
        # A task only has per-block norms for the blocks it actually touches.
        expected_blocks = {b.block_id for b in registry.blocks_for_task(task_id)}
        assert set(record.per_block_norm) == expected_blocks

    combined = recorder.combine()
    assert combined.shared_block_ids == ("core_shared", "ug_shared")
    # Overlap IDs: core_shared is touched by all three tasks, ug_shared only
    # by understanding+generation.
    core_block = next(b for b in registry.blocks if b.block_id == "core_shared")
    ug_block = next(b for b in registry.blocks if b.block_id == "ug_shared")
    assert core_block.overlap_id == "understanding+generation+auxiliary"
    assert ug_block.overlap_id == "understanding+generation"

    stats = recorder.task_statistics("understanding")
    assert stats["finite"] is True
    assert stats["shared_norm"] >= 0.0

    # PCGrad-ready vectors: one zero-padded shared-scope vector per task,
    # all the same length (the total shared-block parameter count).
    lengths = {len(v) for v in combined.pcgrad_vectors.values()}
    assert len(lengths) == 1
    assert len(combined.gram_matrix) == len(TASK_IDS)
    for row in combined.gram_matrix:
        assert len(row) == len(TASK_IDS)
    # Gram matrix is symmetric and its diagonal is each vector's squared norm.
    vectors = list(combined.pcgrad_vectors.values())
    for i, vector in enumerate(vectors):
        assert combined.gram_matrix[i][i] == pytest.approx(float(np.dot(vector, vector)), rel=1e-4)


def test_recorder_clipping_math() -> None:
    model, optimizer, recorder = _build()
    registry = model.block_registry()
    parameters = canonical_parameters(registry)
    shared_params = tuple(p for b in registry.shared_blocks() for p in b.parameters)
    recorder.begin_batch(shared_params)
    for task_id in TASK_IDS:
        optimizer.zero_grad(set_to_none=True)
        batch = make_batch(task_id, batch_index=0, seed=99)
        model.forward_task(task_id, batch)
        recorder.capture_task_gradient(task_id)
    recorder.combine()

    # A tiny max_norm forces clipping; verify pre/post norms and coefficient.
    clip_record = recorder.apply_and_clip(parameters, max_norm=1e-3)
    assert clip_record.pre_clip_norm > clip_record.post_clip_norm
    assert clip_record.clip_coefficient < 1.0
    assert clip_record.post_clip_norm == pytest.approx(
        clip_record.pre_clip_norm * clip_record.clip_coefficient, rel=1e-4
    )


def test_recorder_moments_and_realized_update() -> None:
    model, optimizer, recorder = _build()
    registry = model.block_registry()
    parameters = canonical_parameters(registry)
    shared_params = tuple(p for b in registry.shared_blocks() for p in b.parameters)
    theta_before = theta_snapshot(registry)
    moments_before = recorder.snapshot_pre_step_moments(optimizer)
    # Before any step, AdamW moments are uninitialized for every block.
    assert all(not m.initialized for m in moments_before.values())

    recorder.begin_batch(shared_params)
    for task_id in TASK_IDS:
        optimizer.zero_grad(set_to_none=True)
        batch = make_batch(task_id, batch_index=0, seed=99)
        model.forward_task(task_id, batch)
        recorder.capture_task_gradient(task_id)
    recorder.combine()
    recorder.apply_and_clip(parameters, max_norm=1.0)
    update = recorder.step_and_record(optimizer, theta_before, moments_before)

    assert update.direction_norm >= 0.0
    assert 0.0 <= update.near_zero_rate <= 1.0
    # Every block that had a nonzero combined gradient should now show step=1.
    touched_blocks = {b.block_id for b in registry.blocks if b.overlap_tasks}
    for block_id in touched_blocks:
        assert update.moments_after[block_id].initialized
        assert update.moments_after[block_id].step == 1
