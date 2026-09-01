"""Shared helpers for the oracle test suite (not a conftest fixture, plain import)."""

from __future__ import annotations

import numpy as np

from comppareto.oracle.tasks import OracleTask


def make_task(
    rng: np.random.Generator,
    p_i: int,
    d_i: int,
    *,
    global_dim: int | None = None,
    mu: float = 0.5,
) -> OracleTask:
    global_dim = global_dim if global_dim is not None else p_i
    selector = np.zeros((p_i, global_dim))
    for row in range(p_i):
        selector[row, row] = 1.0
    a = rng.standard_normal((p_i, p_i))
    h_xx = a @ a.T + p_i * np.eye(p_i)
    h_xphi = rng.standard_normal((p_i, d_i))
    b = rng.standard_normal((d_i, d_i))
    h_phiphi = b @ b.T
    phi0 = rng.standard_normal(d_i)
    a_train = rng.standard_normal(p_i)
    b_train = rng.standard_normal(d_i)
    a_meta = rng.standard_normal(p_i)
    b_meta = rng.standard_normal(d_i)
    return OracleTask(h_xx, h_xphi, h_phiphi, mu, phi0, a_train, b_train, a_meta, b_meta, selector)
