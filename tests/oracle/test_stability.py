from __future__ import annotations

import numpy as np
import pytest

from comppareto.oracle import stability as st
from comppareto.oracle.momentum import momentum_transition_jacobian
from comppareto.oracle.sgd import sgd_state_jacobian
from _helpers import make_task


def test_sgd_stable_regime_spectral_radius_below_one(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    eigs_c = np.linalg.eigvalsh(task.private_curvature)
    eta, realized = st.sgd_eta_for_regime(eigs_c, "stable", rng)
    actual = st.spectral_radius(sgd_state_jacobian(task, eta))
    assert realized < 1.0
    assert actual == pytest.approx(realized, rel=1e-9)


def test_sgd_unstable_regime_hits_target_range(rng: np.random.Generator) -> None:
    task = make_task(rng, 4, 6)
    eigs_c = np.linalg.eigvalsh(task.private_curvature)
    eta, realized = st.sgd_eta_for_regime(eigs_c, "unstable", rng)
    actual = st.spectral_radius(sgd_state_jacobian(task, eta))
    assert st.UNSTABLE_RHO_RANGE[0] <= realized <= st.UNSTABLE_RHO_RANGE[1]
    assert actual == pytest.approx(realized, rel=1e-9)


@pytest.mark.parametrize("beta", [0.5, 0.9])
def test_momentum_stable_regime_spectral_radius_below_one(rng: np.random.Generator, beta: float) -> None:
    task = make_task(rng, 4, 6)
    eta, realized = st.momentum_eta_for_regime(task, beta, "stable", rng)
    actual = st.spectral_radius(momentum_transition_jacobian(task, eta, beta))
    assert realized < 1.0
    assert actual == pytest.approx(realized, rel=1e-6)


@pytest.mark.parametrize("beta", [0.5, 0.9])
def test_momentum_unstable_regime_hits_target_range(rng: np.random.Generator, beta: float) -> None:
    task = make_task(rng, 4, 6)
    eta, realized = st.momentum_eta_for_regime(task, beta, "unstable", rng)
    actual = st.spectral_radius(momentum_transition_jacobian(task, eta, beta))
    assert st.UNSTABLE_RHO_RANGE[0] <= realized <= st.UNSTABLE_RHO_RANGE[1]
    assert actual == pytest.approx(realized, rel=1e-6)
