"""Stage 1 (continued): counters, the mock CoRL model's determinism, and the
AdamW optimizer's basic correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from comppareto.instrumentation.counters import StepCounters
from comppareto.instrumentation.mock_corl import MockCoRLModel, TASK_IDS, make_batch
from comppareto.instrumentation.optim import AdamWOptimizer
from comppareto.instrumentation.params import Parameter


def test_counters_accumulate() -> None:
    counters = StepCounters()
    counters.add_rollout(tokens=16)
    counters.add_rollout(tokens=16)
    counters.add_reward_call()
    counters.add_backward()
    assert counters.rollout_count == 2
    assert counters.token_count == 32
    assert counters.reward_call_count == 1
    assert counters.backward_count == 1
    with counters.timed("phase_a"):
        pass
    assert counters.wall_time_seconds["phase_a"] >= 0.0
    assert counters.total_wall_time_seconds() >= 0.0
    as_dict = counters.to_dict()
    assert as_dict["rollout_count"] == 2


def test_counters_rejects_negative_tokens() -> None:
    counters = StepCounters()
    with pytest.raises(ValueError):
        counters.add_rollout(tokens=-1)


def test_make_batch_is_deterministic_and_pure() -> None:
    a = make_batch("understanding", batch_index=0, seed=123)
    b = make_batch("understanding", batch_index=0, seed=123)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])

    other_task = make_batch("generation", batch_index=0, seed=123)
    assert not np.array_equal(a[0], other_task[0])

    other_batch = make_batch("understanding", batch_index=1, seed=123)
    assert not np.array_equal(a[0], other_batch[0])


def test_mock_model_construction_is_deterministic() -> None:
    model_a = MockCoRLModel(seed=42)
    model_b = MockCoRLModel(seed=42)
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        np.testing.assert_array_equal(param_a.value, param_b.value)

    model_c = MockCoRLModel(seed=43)
    assert not np.array_equal(model_a.core_shared_w.value, model_c.core_shared_w.value)


def test_mock_model_block_registry_shapes() -> None:
    model = MockCoRLModel(seed=42)
    registry = model.block_registry()
    assert registry.block_ids() == (
        "core_shared",
        "ug_shared",
        "private_understanding",
        "private_generation",
        "private_auxiliary",
    )
    assert set(registry.all_task_ids()) == set(TASK_IDS)
    shared_ids = {b.block_id for b in registry.shared_blocks()}
    assert shared_ids == {"core_shared", "ug_shared"}
    registry.validate_ownership(model.parameters())


def test_mock_model_forward_task_produces_finite_gradients() -> None:
    model = MockCoRLModel(seed=7)
    batch = make_batch("understanding", batch_index=0, seed=7)
    loss, info = model.forward_task("understanding", batch)
    assert np.isfinite(loss)
    assert info.rollout_count == 4
    assert info.reward_call_count == 4
    assert info.token_count == 64
    # This task's blocks got a gradient; the auxiliary-only block did not.
    assert model.core_shared_w.grad is not None
    assert model.ug_shared_w.grad is not None
    assert model.private_understanding_w.grad is not None
    assert model.private_generation_w.grad is None
    assert model.private_auxiliary_w.grad is None
    assert np.isfinite(model.core_shared_w.grad).all()


def test_adamw_step_changes_parameters_and_tracks_state() -> None:
    parameter = Parameter(np.zeros(3, dtype=np.float32))
    optimizer = AdamWOptimizer(parameters=(parameter,), lr=0.1)
    parameter.grad = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    before = parameter.value.copy()
    optimizer.step()
    assert not np.array_equal(before, parameter.value)
    state = optimizer.moment_state(parameter)
    assert state["step"] == 1
    assert state["exp_avg"] is not None


def test_adamw_zero_grad_set_to_none() -> None:
    parameter = Parameter(np.zeros(2, dtype=np.float32))
    optimizer = AdamWOptimizer(parameters=(parameter,))
    parameter.grad = np.ones(2, dtype=np.float32)
    optimizer.zero_grad(set_to_none=True)
    assert parameter.grad is None
