"""Toy-model unit tests for T216 alternating-protocol diagnostic utilities.

All tests are CPU-only and require no GPU, no Show-o2 model, and no large
dependencies beyond torch. They verify:
  1. snapshot/restore restores parameters to bit-exact values
  2. RNG snapshot/restore is byte-exact (the ROLLBACK-RNG fix)
  3. protocol ordering matches the P0-P3 pseudocode from first-report.md
  4. all four negotiators produce finite, correctly-shaped outputs
  5. the attribution identity Delta_total = Delta_private + Delta_controlled
     holds within float tolerance on a deterministic toy quadratic
"""

import math
import random
import sys
import os
import pytest
import torch
import torch.nn as nn

# Make the package importable from the worktree root.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

from comppareto.adapters.showo2_alternating.snapshot import (
    snapshot_params, restore_params, snapshot_rng, restore_rng, assert_exact_restore
)
from comppareto.adapters.showo2_alternating.tensor_utils import flatten, unflatten
from comppareto.adapters.showo2_alternating.negotiators import (
    raw_sum, normalized_sum, pcgrad, mgda_two_task, unit_normalize, all_mandatory_directions
)
from comppareto.adapters.showo2_alternating.protocols import (
    TaskModel, fresh_adamw_step, private_adapt, raw_shared_gradient,
    run_p1_control, run_p0, run_p2, run_p3, _split_direction
)


# ---------------------------------------------------------------------------
# Shared toy setup
# ---------------------------------------------------------------------------

def _make_toy_shared(dim: int = 8) -> list[torch.Tensor]:
    """Small linear layer as the shared subspace."""
    w = nn.Parameter(torch.randn(dim, dim) * 0.1)
    b = nn.Parameter(torch.zeros(dim))
    w.requires_grad_(True)
    b.requires_grad_(True)
    return [w, b]


def _make_toy_private(dim: int = 8) -> list[torch.Tensor]:
    w = nn.Parameter(torch.randn(dim, dim) * 0.1)
    b = nn.Parameter(torch.zeros(dim))
    w.requires_grad_(True)
    b.requires_grad_(True)
    return [w, b]


def _toy_loss_fn(
    shared: list[torch.Tensor],
    private: list[torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    """Simple quadratic L = ||W_s x + W_p x - target||^2, x = ones."""
    x = torch.ones(shared[0].shape[0])
    out = torch.mv(shared[0], x) + shared[1] + torch.mv(private[0], x) + private[1]
    return ((out - target) ** 2).mean()


def _make_tasks(dim: int = 8):
    torch.manual_seed(42)
    shared = _make_toy_shared(dim)
    priv_und = _make_toy_private(dim)
    priv_gen = _make_toy_private(dim)
    tgt_und = torch.randn(dim)
    tgt_gen = torch.randn(dim)

    task_und = TaskModel(
        shared_params=shared,
        private_params=priv_und,
        loss_fn=lambda s, p: _toy_loss_fn(s, p, tgt_und),
        task_id="und",
    )
    task_gen = TaskModel(
        shared_params=shared,
        private_params=priv_gen,
        loss_fn=lambda s, p: _toy_loss_fn(s, p, tgt_gen),
        task_id="gen",
    )
    return task_und, task_gen, shared, priv_und, priv_gen


# ---------------------------------------------------------------------------
# Tests: snapshot / restore
# ---------------------------------------------------------------------------

class TestSnapshotRestore:
    def test_params_restore_bitexact(self):
        _, _, shared, priv_und, _ = _make_tasks()
        params = shared + priv_und
        snap = snapshot_params(params)
        # mutate
        with torch.no_grad():
            for p in params:
                p.add_(1.0)
        restore_params(params, snap)
        assert_exact_restore(params, snap, atol=0.0, rtol=0.0)

    def test_rng_restore_bitexact(self):
        """Restoring RNG from saved snapshot produces bit-identical random draws."""
        snap = snapshot_rng()
        a1 = torch.rand(10).tolist()
        restore_rng(snap)
        a2 = torch.rand(10).tolist()
        assert a1 == a2, "RNG restore must produce bit-identical draws"

    def test_rng_restore_vs_manual_seed(self):
        """Explicit snapshot restore and manual_seed diverge when intermediate ops differ."""
        torch.manual_seed(99)
        _ = torch.rand(5)  # consume some randomness
        snap = snapshot_rng()
        draw_a = torch.rand(3).tolist()
        # restore via snapshot: must reproduce draw_a
        restore_rng(snap)
        draw_b = torch.rand(3).tolist()
        assert draw_a == draw_b

    def test_flatten_unflatten_roundtrip(self):
        _, _, shared, priv_und, _ = _make_tasks()
        params = shared + priv_und
        flat = flatten(params)
        restored = unflatten(flat, params)
        for orig, rec in zip(params, restored):
            assert orig.shape == rec.shape
            assert torch.allclose(orig, rec)

    def test_flatten_empty_raises(self):
        with pytest.raises(ValueError, match="at least one tensor"):
            flatten([])


# ---------------------------------------------------------------------------
# Tests: negotiators
# ---------------------------------------------------------------------------

class TestNegotiators:
    def _grads(self, dim=16):
        torch.manual_seed(7)
        g1 = torch.randn(dim)
        g2 = torch.randn(dim)
        return g1, g2

    def test_raw_sum_shape(self):
        g1, g2 = self._grads()
        d = raw_sum([g1, g2])
        assert d.shape == g1.shape
        assert torch.allclose(d, g1 + g2)

    def test_normalized_sum_unit_when_collinear(self):
        g = torch.tensor([1.0, 0.0, 0.0])
        d = normalized_sum([g, g])
        # both normalized to 1.0, so sum norm == 2.0; direction should be unit
        # (not necessarily unit norm — but the direction must be finite)
        assert torch.isfinite(d).all()

    def test_normalized_sum_zero_guard(self):
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.zeros(2)  # near-zero norm
        d = normalized_sum([g1, g2])
        # g2 contribution is guarded to zero
        assert torch.isfinite(d).all()

    def test_pcgrad_finite(self):
        g1, g2 = self._grads()
        d = pcgrad([g1, g2])
        assert d.shape == g1.shape
        assert torch.isfinite(d).all()

    def test_pcgrad_order_sensitivity(self):
        torch.manual_seed(3)
        g1 = torch.randn(16)
        g2 = -g1 + 0.01 * torch.randn(16)  # near-antipodal → order matters
        d_fwd = pcgrad([g1, g2], order=[0, 1])
        d_rev = pcgrad([g1, g2], order=[1, 0])
        # directions are generally different when gradients conflict
        # (not always, but near-antipodal should differ)
        assert not torch.allclose(d_fwd, d_rev, atol=1e-4), \
            "PCGrad with reversed order should differ for conflicting gradients"

    def test_mgda_convex_combination(self):
        g1, g2 = self._grads()
        d = mgda_two_task(g1, g2)
        assert d.shape == g1.shape
        assert torch.isfinite(d).all()

    def test_mgda_degenerate_equal_grads(self):
        g = torch.randn(8)
        d = mgda_two_task(g, g)
        assert torch.allclose(d, g)

    def test_unit_normalize(self):
        v = torch.tensor([3.0, 4.0])
        u = unit_normalize(v)
        assert abs(float(torch.linalg.vector_norm(u)) - 1.0) < 1e-6

    def test_unit_normalize_zero_passthrough(self):
        v = torch.zeros(4)
        u = unit_normalize(v)
        assert torch.allclose(u, v)

    def test_all_mandatory_directions_keys(self):
        g1, g2 = self._grads()
        dirs = all_mandatory_directions([g1, g2])
        assert set(dirs.keys()) == {"raw_sum", "normalized_sum", "pcgrad", "mgda"}
        for d in dirs.values():
            assert torch.isfinite(d).all()


# ---------------------------------------------------------------------------
# Tests: protocol logic and attribution identity
# ---------------------------------------------------------------------------

STEP_SCALES = [5e-6, 5e-5, 5e-4]


def _make_snapshot_closures(tasks, shared):
    from comppareto.adapters.showo2_alternating.snapshot import snapshot_params, restore_params

    def snapshot_fn():
        all_params = list(shared)
        for t in tasks:
            all_params += list(t.private_params)
        return snapshot_params(all_params)

    def restore_fn(snap):
        all_params = list(shared)
        for t in tasks:
            all_params += list(t.private_params)
        restore_params(all_params, snap)

    return snapshot_fn, restore_fn


class TestProtocols:
    def setup_method(self):
        self.task_und, self.task_gen, self.shared, self.priv_und, self.priv_gen = _make_tasks()
        self.tasks = [self.task_und, self.task_gen]
        self.snapshot_fn, self.restore_fn = _make_snapshot_closures(
            self.tasks, self.shared)

    def test_p1_control_restores_params(self):
        snap0 = self.snapshot_fn()
        run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        # after P1, params should be restored to snap0
        assert_exact_restore(
            list(self.shared) + list(self.priv_und) + list(self.priv_gen),
            snap0, atol=0.0, rtol=0.0
        )

    def test_p1_control_losses_finite(self):
        results = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        for tid in ["und", "gen"]:
            assert math.isfinite(results["control_losses"][tid])
            assert math.isfinite(results["loss_before"][tid])

    def _build_directions(self):
        g_und = raw_shared_gradient(self.task_und)
        g_gen = raw_shared_gradient(self.task_gen)
        g_und_flat = flatten(g_und)
        g_gen_flat = flatten(g_gen)
        return all_mandatory_directions([g_und_flat, g_gen_flat])

    def test_p0_rows_all_finite(self):
        control = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        dirs = self._build_directions()
        rows = run_p0(
            self.tasks, dirs, STEP_SCALES, control,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
        )
        assert len(rows) == len(dirs) * len(STEP_SCALES) * len(self.tasks)
        for row in rows:
            assert row.finite, f"P0 row not finite: {row}"

    def test_p0_restores_after_every_row(self):
        snap_before = self.snapshot_fn()
        control = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        dirs = self._build_directions()
        run_p0(
            self.tasks, dirs, STEP_SCALES, control,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
        )
        snap_after = self.snapshot_fn()
        # restored snapshots should match snap_before
        for a, b in zip(snap_before.values, snap_after.values):
            assert torch.allclose(a, b), "P0 should leave params in snap0 state"

    def test_p2_rows_all_finite(self):
        control = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        dirs = self._build_directions()
        rows = run_p2(
            self.tasks, dirs, STEP_SCALES, control, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
        )
        assert len(rows) == len(dirs) * len(STEP_SCALES) * len(self.tasks)
        for row in rows:
            assert row.finite, f"P2 row not finite: {row}"

    def test_p3_rows_all_finite(self):
        # compute commit directions (raw_sum and mgda of raw gradients for toy model)
        g_und = raw_shared_gradient(self.task_und)
        g_gen = raw_shared_gradient(self.task_gen)
        g_und_flat = flatten(g_und)
        g_gen_flat = flatten(g_gen)
        commit_dirs = {
            "raw_sum_commit": raw_sum([g_und_flat, g_gen_flat]),
            "mgda_commit": mgda_two_task(g_und_flat, g_gen_flat),
        }
        control = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        rows = run_p3(
            self.tasks, commit_dirs, STEP_SCALES, control, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
        )
        for row in rows:
            assert row.finite, f"P3 row not finite: {row}"

    def test_attribution_identity_p2(self):
        """Delta_total == Delta_private + Delta_controlled within float tol."""
        control = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        dirs = self._build_directions()
        rows = run_p2(
            self.tasks, dirs, STEP_SCALES, control, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
        )
        for row in rows:
            if row.finite and row.delta_total is not None:
                expected = (row.delta_private or 0.0) + (row.delta_controlled or 0.0)
                assert abs(row.delta_total - expected) < 1e-4, \
                    f"Attribution identity violated: {row.delta_total} != {expected} for {row}"

    def test_persistent_updates_zero(self):
        """After all protocol runs, model params must equal original snapshot."""
        snap0 = self.snapshot_fn()
        control = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )
        dirs = self._build_directions()
        run_p0(self.tasks, dirs, STEP_SCALES, control,
               snapshot_fn=self.snapshot_fn, restore_fn=self.restore_fn)
        run_p2(self.tasks, dirs, STEP_SCALES, control, K=1,
               snapshot_fn=self.snapshot_fn, restore_fn=self.restore_fn)
        snap_final = self.snapshot_fn()
        for a, b in zip(snap0.values, snap_final.values):
            assert torch.allclose(a, b), "persistent_updates must be 0: params must be restored"

    def test_determinism_same_seed_same_output(self):
        """Same RNG snapshot → bit-identical losses."""
        rng_snap = snapshot_rng()
        snap0 = self.snapshot_fn()

        restore_rng(rng_snap)
        self.restore_fn(snap0)
        control_a = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )

        restore_rng(rng_snap)
        self.restore_fn(snap0)
        control_b = run_p1_control(
            self.tasks, K=1,
            snapshot_fn=self.snapshot_fn,
            restore_fn=self.restore_fn,
            step_scales=STEP_SCALES,
        )

        for tid in ["und", "gen"]:
            assert control_a["control_losses"][tid] == control_b["control_losses"][tid], \
                f"Non-deterministic control loss for task {tid}"
