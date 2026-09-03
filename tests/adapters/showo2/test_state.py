"""Unit tests for ``comppareto.adapters.showo2.state`` (T215).

Pure PyTorch, small synthetic ``nn.Module``s only -- no GPU, no real Show-o2
model. Exercises the reversible snapshot/restore contract that the T215
diagnostic depends on: after any mutation to a module's declared subspace
parameters/grads/AdamW optimizer state/RNG streams, restoring from a
snapshot must reassert bitwise-identical (within dtype tolerance for
floats, exactly for counters/RNG) state.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from comppareto.adapters.showo2 import state


class _TinySubspaceModule(nn.Module):
    """A tiny synthetic module with 3 named parameter tensors, standing in
    for the real diagnostic's (fusion_proj, und_trans.layers[0],
    diffusion_head_a[0]) subspace.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.private_a = nn.Linear(4, 3)
        self.private_b = nn.Linear(3, 2)
        # An extra parameter NOT in the declared diagnostic subspace, to
        # confirm snapshot/restore only touches the named subset.
        self.untouched = nn.Linear(2, 2)


SUBSPACE_NAMES = [
    "shared.weight",
    "shared.bias",
    "private_a.weight",
    "private_b.weight",
]


def _make_module_and_optimizer() -> tuple[_TinySubspaceModule, AdamW]:
    torch.manual_seed(0)
    module = _TinySubspaceModule()
    params = [state._param_by_name(module, n) for n in SUBSPACE_NAMES]
    optimizer = AdamW(params, lr=1e-2)
    return module, optimizer


def _run_dummy_step(module: _TinySubspaceModule, optimizer: AdamW) -> None:
    """One forward/backward/optimizer-step over the declared subspace,
    to populate grads and AdamW moment state before snapshotting.
    """

    x = torch.randn(5, 4)
    y = module.private_b(module.private_a(module.shared(x)))
    loss = y.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_param_by_name_finds_declared_and_raises_on_unknown() -> None:
    module, _ = _make_module_and_optimizer()
    p = state._param_by_name(module, "shared.weight")
    assert p is module.shared.weight
    with pytest.raises(KeyError):
        state._param_by_name(module, "does.not.exist")


def test_snapshot_and_restore_params_roundtrip_after_optimizer_step() -> None:
    module, optimizer = _make_module_and_optimizer()
    _run_dummy_step(module, optimizer)

    snapshot = state.snapshot_params(module, SUBSPACE_NAMES, optimizer)

    # Mutate everything in the declared subspace: another optimizer step
    # plus a manual perturbation.
    _run_dummy_step(module, optimizer)
    with torch.no_grad():
        state._param_by_name(module, "shared.weight").add_(1.0)

    restored_ok, per_param = state.verify_rollback(module, state.SubspaceSnapshot(
        params=snapshot, rng=state.snapshot_rng(), data_order=state.DataOrderSnapshot()
    ), optimizer)
    # Before restoring, rollback verification must fail (state has moved).
    assert not restored_ok
    assert not all(r.all_matches for r in per_param)

    state.restore_params(module, snapshot, optimizer)

    for name in SUBSPACE_NAMES:
        comparison = state.compare_param_snapshot(
            module, next(s for s in snapshot if s.name == name), optimizer
        )
        assert comparison.all_matches, comparison


def test_snapshot_restore_leaves_untouched_parameter_alone() -> None:
    module, optimizer = _make_module_and_optimizer()
    _run_dummy_step(module, optimizer)
    snapshot = state.snapshot_params(module, SUBSPACE_NAMES, optimizer)

    before = module.untouched.weight.detach().clone()
    with torch.no_grad():
        state._param_by_name(module, "shared.weight").add_(5.0)
    state.restore_params(module, snapshot, optimizer)

    # restore_params never touches parameters outside the declared subspace.
    assert torch.equal(module.untouched.weight.detach(), before)


def test_grad_snapshot_restore_handles_none_grad() -> None:
    module, optimizer = _make_module_and_optimizer()
    # No backward pass yet: grads are None.
    snapshot = state.snapshot_params(module, SUBSPACE_NAMES, optimizer)
    for snap in snapshot:
        assert snap.grad is None

    _run_dummy_step(module, optimizer)
    for name in SUBSPACE_NAMES:
        assert state._param_by_name(module, name).grad is not None

    state.restore_params(module, snapshot, optimizer)
    for name in SUBSPACE_NAMES:
        # After restoring a None-grad snapshot, grad must be reset to None.
        assert state._param_by_name(module, name).grad is None


def test_opt_state_snapshot_restore_matches_exactly_for_step_counter() -> None:
    module, optimizer = _make_module_and_optimizer()
    _run_dummy_step(module, optimizer)
    _run_dummy_step(module, optimizer)
    snapshot = state.snapshot_params(module, SUBSPACE_NAMES, optimizer)

    for snap in snapshot:
        assert snap.opt_step is not None
        assert int(snap.opt_step.item()) == 2

    _run_dummy_step(module, optimizer)
    for name in SUBSPACE_NAMES:
        p = state._param_by_name(module, name)
        assert int(optimizer.state[p]["step"].item()) == 3

    state.restore_params(module, snapshot, optimizer)
    for name in SUBSPACE_NAMES:
        p = state._param_by_name(module, name)
        assert int(optimizer.state[p]["step"].item()) == 2


def test_rng_snapshot_restore_exact_torch_and_python_random() -> None:
    torch.manual_seed(123)
    random.seed(456)
    snap = state.snapshot_rng()

    # Advance both RNG streams.
    _ = torch.randn(10)
    _ = [random.random() for _ in range(5)]

    current = state.snapshot_rng()
    assert not state.compare_rng_snapshot(current, snap)

    state.restore_rng(snap)
    restored = state.snapshot_rng()
    assert state.compare_rng_snapshot(restored, snap)

    # And the actual next-drawn values match a fresh draw from the restored
    # state (not just the state object itself).
    state.restore_rng(snap)
    first_after_restore = torch.randn(3).clone()
    state.restore_rng(snap)
    second_after_restore = torch.randn(3).clone()
    assert torch.equal(first_after_restore, second_after_restore)


def test_data_order_snapshot_restore_and_compare() -> None:
    indices = {"adapt": [3, 1, 4], "meta": [1, 5, 9]}
    snap = state.snapshot_data_order(indices)
    same = state.snapshot_data_order({"adapt": [3, 1, 4], "meta": [1, 5, 9]})
    different = state.snapshot_data_order({"adapt": [3, 1, 4], "meta": [1, 5, 8]})

    assert state.compare_data_order_snapshot(same, snap)
    assert not state.compare_data_order_snapshot(different, snap)


def test_full_subspace_snapshot_and_verify_rollback_end_to_end() -> None:
    module, optimizer = _make_module_and_optimizer()
    _run_dummy_step(module, optimizer)

    data_indices = {"adapt": [0, 1, 2], "meta": [3, 4]}
    full_snapshot = state.snapshot_subspace(module, SUBSPACE_NAMES, optimizer, data_indices)

    # Verify immediately: nothing has moved, rollback check must pass.
    ok, per_param = state.verify_rollback(module, full_snapshot, optimizer, data_indices)
    assert ok
    assert all(r.all_matches for r in per_param)

    # Perform a diagnostic-like transition: mutate params/grads/optimizer
    # state and RNG, then restore and re-verify.
    _run_dummy_step(module, optimizer)
    torch.manual_seed(999)
    _ = torch.randn(4)

    ok_before_restore, _ = state.verify_rollback(module, full_snapshot, optimizer, data_indices)
    assert not ok_before_restore

    state.restore_subspace(module, full_snapshot, optimizer)
    ok_after_restore, per_param_after = state.verify_rollback(
        module, full_snapshot, optimizer, data_indices
    )
    assert ok_after_restore
    assert all(r.all_matches for r in per_param_after)


def test_dtype_tolerance_bounds() -> None:
    assert state._dtype_tolerance(torch.float64) == 0.0
    assert state._dtype_tolerance(torch.float32) == pytest.approx(1e-12)
    assert state._dtype_tolerance(torch.float16) == pytest.approx(1e-3)
    assert state._dtype_tolerance(torch.bfloat16) == pytest.approx(1e-3)
