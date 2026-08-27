from __future__ import annotations

import numpy as np
import pytest

from comppareto.oracle.generation import generate_coupling, generate_curvature, generate_tasks
from comppareto.oracle.selectors import BlockLayout, build_incidence


def test_generate_curvature_condition_number_and_symmetry(rng: np.random.Generator) -> None:
    h = generate_curvature(rng, 6, 100.0)
    assert np.allclose(h, h.T)
    eigs = np.linalg.eigvalsh(h)
    assert np.all(eigs > 0)
    assert eigs.max() / eigs.min() == pytest.approx(100.0, rel=1e-9)


def test_generate_coupling_rank(rng: np.random.Generator) -> None:
    h_xphi = generate_coupling(rng, 6, 8, 3)
    assert np.linalg.matrix_rank(h_xphi) == 3


@pytest.mark.parametrize("cosine_target,scale_target", [(-0.8, 1.0), (0.0, 10.0), (0.8, 100.0)])
def test_generate_tasks_realizes_designated_pair_cosine_scale(
    rng: np.random.Generator, cosine_target: float, scale_target: float
) -> None:
    layout = BlockLayout((2,) * 8)
    inc = build_incidence("full_overlap", num_tasks=3, num_blocks=8, rng=rng)
    tasks, diagnostics = generate_tasks(
        layout,
        inc,
        rng,
        private_dims=(4, 4, 4),
        condition_number=10.0,
        coupling_rank=2,
        mu=0.5,
        gradient_cosine_target=cosine_target,
        gradient_scale_target=scale_target,
    )
    assert diagnostics.designated_pair is not None
    assert diagnostics.gradient_cosine_realized == pytest.approx(cosine_target, abs=1e-9)
    assert diagnostics.gradient_scale_realized == pytest.approx(scale_target, rel=1e-9)


def test_generate_tasks_disjoint_has_no_designated_pair(rng: np.random.Generator) -> None:
    layout = BlockLayout((2,) * 8)
    inc = build_incidence("disjoint", num_tasks=4, num_blocks=8, rng=rng)
    tasks, diagnostics = generate_tasks(
        layout, inc, rng, private_dims=(4, 4, 4, 4), condition_number=10.0, coupling_rank=2, mu=0.5,
    )
    assert diagnostics.designated_pair is None
    assert diagnostics.gradient_cosine_realized is None
    assert len(tasks) == 4
