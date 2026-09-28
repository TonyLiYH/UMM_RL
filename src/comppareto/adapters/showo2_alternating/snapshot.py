"""Exact snapshot/restore utilities for parameters and RNG state.

Deliberately restores RNG state from one explicitly-saved snapshot rather
than re-seeding via ``torch.manual_seed()`` on every row, which is the fix
for T215's documented ``ROLLBACK-RNG`` failure (branches consuming different
amounts of randomness before a re-seed call causing byte-inexact restores;
see ``reports/T216/first-report.md`` section 9 and the (read-only, not
imported) T215 failure ledger it cites).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch


@dataclass
class ParamSnapshot:
    """Cloned, detached tensor values for one named group of parameters."""

    values: List[torch.Tensor]

    def restore_into(self, params: Iterable[torch.Tensor]) -> None:
        for p, saved in zip(params, self.values):
            with torch.no_grad():
                p.copy_(saved)


@dataclass
class RngSnapshot:
    torch_cpu: torch.Tensor
    torch_cuda: Any  # list[Tensor] or None
    python_random: tuple
    numpy_random: Any  # optional numpy state tuple, or None


def snapshot_params(params: Iterable[torch.Tensor]) -> ParamSnapshot:
    return ParamSnapshot(values=[p.detach().clone() for p in params])


def restore_params(params: Iterable[torch.Tensor], snap: ParamSnapshot) -> None:
    snap.restore_into(params)


def snapshot_rng() -> RngSnapshot:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        import numpy as _np  # local, optional dependency

        numpy_state = _np.random.get_state()
    except Exception:
        numpy_state = None
    return RngSnapshot(
        torch_cpu=torch.get_rng_state(),
        torch_cuda=cuda_state,
        python_random=random.getstate(),
        numpy_random=numpy_state,
    )


def restore_rng(snap: RngSnapshot) -> None:
    torch.set_rng_state(snap.torch_cpu)
    if snap.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap.torch_cuda)
    random.setstate(snap.python_random)
    if snap.numpy_random is not None:
        try:
            import numpy as _np

            _np.random.set_state(snap.numpy_random)
        except Exception:
            pass


def assert_exact_restore(
    params: Iterable[torch.Tensor], snap: ParamSnapshot, *, atol: float = 0.0, rtol: float = 0.0
) -> None:
    """Raise AssertionError unless every tensor matches the snapshot within tolerance.

    Default tolerance is exact (0, 0): a plain ``copy_`` restore should be
    bit-exact. A nonzero ``atol``/``rtol`` may be declared explicitly by a
    caller that needs a dtype-driven tolerance (per the task's pass/fail gate:
    "parameter tensors restore within declared dtype tolerance").
    """

    for p, saved in zip(params, snap.values):
        if not torch.allclose(p.detach(), saved, atol=atol, rtol=rtol):
            raise AssertionError(f"parameter restore mismatch: shape={tuple(p.shape)}")
