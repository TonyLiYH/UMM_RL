"""Controllable random task generation: curvature spectrum, coupling rank,
gradient cosine/scale, matching ``docs/theory/oracle-spec.md`` section 8.

Every random draw goes through the caller-supplied ``rng``
(``numpy.random.Generator``), so a case is exactly reproducible from its
seed (see :mod:`comppareto.oracle.seeds`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from comppareto.oracle.noise import NoiseModel
from comppareto.oracle.selectors import BlockLayout, BoolArray, build_task_selectors
from comppareto.oracle.tasks import OracleTask

FloatArray = NDArray[np.float64]


def generate_curvature(rng: np.random.Generator, dim: int, condition_number: float) -> FloatArray:
    """Symmetric PD matrix with log-spaced eigenvalues in ``[1, condition_number]``."""

    if dim == 0:
        return np.zeros((0, 0))
    if dim == 1:
        return np.array([[condition_number]])
    a = rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(a)
    eigs = np.geomspace(1.0, condition_number, dim)
    return q @ np.diag(eigs) @ q.T


def generate_coupling(rng: np.random.Generator, p_i: int, d_i: int, rank: int) -> FloatArray:
    """``H_xphi`` of prescribed rank, via a random orthonormal-factor product."""

    rank = max(1, min(rank, p_i, d_i)) if p_i and d_i else 0
    if rank == 0:
        return np.zeros((p_i, d_i))
    u, _ = np.linalg.qr(rng.standard_normal((p_i, rank)))
    v, _ = np.linalg.qr(rng.standard_normal((d_i, rank)))
    s = rng.uniform(0.5, 1.5, size=rank)
    return u @ np.diag(s) @ v.T


def _find_designated_pair(incidence: BoolArray) -> tuple[int, int] | None:
    m = incidence.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            if np.any(incidence[i] & incidence[j]):
                return i, j
    return None


def _shared_global_coords(layout: BlockLayout, incidence: BoolArray, i: int, j: int) -> NDArray[np.intp]:
    shared_blocks = np.nonzero(incidence[i] & incidence[j])[0]
    coords: list[int] = []
    for b in shared_blocks:
        block_slice = layout.block_slice(int(b))
        coords.extend(range(block_slice.start, block_slice.stop))
    return np.array(coords, dtype=np.intp)


def _apply_cosine_scale(
    rng: np.random.Generator,
    a_i_global: FloatArray,
    a_j_global: FloatArray,
    shared_coords: NDArray[np.intp],
    cosine: float,
    scale_i: float,
    scale_j: float,
) -> None:
    """Overwrite ``a_i``/``a_j`` on ``shared_coords`` to realize the target cosine/scale.

    Exact by construction when ``len(shared_coords) >= 2``: the two vectors
    are built from an orthonormal pair ``(u, w)`` on the shared subspace. With
    exactly one shared coordinate only ``cosine = +-1`` is achievable; the
    sign of the target is used and the shortfall is left for the caller to
    measure and record (per §8's "targets... realized" scoping).
    """

    n = shared_coords.size
    if n == 0:
        return
    if n == 1:
        sign = 1.0 if cosine >= 0 else -1.0
        unit = np.array([1.0])
        a_i_global[shared_coords] = scale_i * unit
        a_j_global[shared_coords] = scale_j * sign * unit
        return
    u = rng.standard_normal(n)
    u /= np.linalg.norm(u)
    w = rng.standard_normal(n)
    w = w - (w @ u) * u
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-12:
        w = rng.standard_normal(n)
        w = w - (w @ u) * u
        w_norm = np.linalg.norm(w)
    w /= w_norm
    a_i_global[shared_coords] = scale_i * u
    a_j_global[shared_coords] = scale_j * (cosine * u + np.sqrt(max(0.0, 1.0 - cosine**2)) * w)


@dataclass(frozen=True)
class GenerationDiagnostics:
    """Realized generation targets, recorded verbatim in the case manifest."""

    condition_number_target: float
    condition_number_realized: tuple[float, ...]
    coupling_rank_target: int
    coupling_rank_realized: tuple[int, ...]
    designated_pair: tuple[int, int] | None
    gradient_cosine_target: float | None
    gradient_cosine_realized: float | None
    gradient_scale_target: float | None
    gradient_scale_realized: float | None


def generate_tasks(
    layout: BlockLayout,
    incidence: BoolArray,
    rng: np.random.Generator,
    *,
    private_dims: tuple[int, ...],
    condition_number: float,
    coupling_rank: int,
    mu: float = 0.5,
    gradient_scale_baseline: float = 1.0,
    gradient_cosine_target: float | None = None,
    gradient_scale_target: float | None = None,
) -> tuple[tuple[OracleTask, ...], GenerationDiagnostics]:
    m = incidence.shape[0]
    if len(private_dims) != m:
        raise ValueError("private_dims must have one entry per task")

    selectors = build_task_selectors(layout, incidence)
    p_total = layout.total_dim

    a_train_global = [gradient_scale_baseline * rng.standard_normal(p_total) for _ in range(m)]
    a_meta_global = [gradient_scale_baseline * rng.standard_normal(p_total) for _ in range(m)]

    designated = _find_designated_pair(incidence)
    realized_cosine = None
    realized_scale = None
    if designated is not None and (gradient_cosine_target is not None or gradient_scale_target is not None):
        i0, i1 = designated
        shared = _shared_global_coords(layout, incidence, i0, i1)
        cosine = gradient_cosine_target if gradient_cosine_target is not None else 0.0
        ratio = gradient_scale_target if gradient_scale_target is not None else 1.0
        _apply_cosine_scale(
            rng, a_train_global[i0], a_train_global[i1], shared, cosine, gradient_scale_baseline,
            gradient_scale_baseline * ratio,
        )
        v0 = a_train_global[i0][shared]
        v1 = a_train_global[i1][shared]
        realized_cosine = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        realized_scale = float(np.linalg.norm(v1) / np.linalg.norm(v0))

    tasks = []
    cond_realized = []
    rank_realized = []
    for i in range(m):
        selector = selectors[i]
        p_i = selector.shape[0]
        d_i = private_dims[i]

        h_xx = generate_curvature(rng, p_i, condition_number)
        h_phiphi = generate_curvature(rng, d_i, condition_number)
        h_xphi = generate_coupling(rng, p_i, d_i, coupling_rank)
        phi0 = rng.standard_normal(d_i)
        b_train = gradient_scale_baseline * rng.standard_normal(d_i)
        b_meta = gradient_scale_baseline * rng.standard_normal(d_i)
        a_train = selector @ a_train_global[i]
        a_meta = selector @ a_meta_global[i]

        tasks.append(
            OracleTask(h_xx, h_xphi, h_phiphi, mu, phi0, a_train, b_train, a_meta, b_meta, selector)
        )
        cond_realized.append(
            float(np.linalg.cond(h_xx)) if p_i > 1 else 1.0
        )
        rank_realized.append(int(np.linalg.matrix_rank(h_xphi)) if p_i and d_i else 0)

    diagnostics = GenerationDiagnostics(
        condition_number_target=condition_number,
        condition_number_realized=tuple(cond_realized),
        coupling_rank_target=coupling_rank,
        coupling_rank_realized=tuple(rank_realized),
        designated_pair=designated,
        gradient_cosine_target=gradient_cosine_target,
        gradient_cosine_realized=realized_cosine,
        gradient_scale_target=gradient_scale_target,
        gradient_scale_realized=realized_scale,
    )
    return tuple(tasks), diagnostics
