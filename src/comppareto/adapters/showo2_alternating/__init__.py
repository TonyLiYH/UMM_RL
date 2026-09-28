"""T216 alternating-protocol diagnostic utilities.

Small, self-contained utilities for the compute-matched P0-P3 alternating
shared/private update diagnostic (see
``tasks/T216-showo2-alternating-protocol-diagnostic.md`` and
``docs/math/01-alternating-shared-private-update.md``).

This package intentionally does not import anything from
``src/comppareto/adapters/showo2/`` (T210/T215's adapter) or from any T215
branch. It depends only on ``torch`` and, for the real Show-o2 execution
path (``run_k1.py``), on the official upstream Show-o2 library used as a
plain read-only dependency, exactly as T210's admission audited it.
"""

from .tensor_utils import flatten, unflatten
from .snapshot import ParamSnapshot, RngSnapshot, snapshot_params, restore_params, snapshot_rng, restore_rng
from .negotiators import raw_sum, normalized_sum, pcgrad, mgda_two_task, unit_normalize
from .protocols import (
    TaskModel,
    fresh_adamw_step,
    private_adapt,
    raw_shared_gradient,
    run_p1_control,
    run_p0,
    run_p2,
    run_p3,
)

__all__ = [
    "flatten",
    "unflatten",
    "ParamSnapshot",
    "RngSnapshot",
    "snapshot_params",
    "restore_params",
    "snapshot_rng",
    "restore_rng",
    "raw_sum",
    "normalized_sum",
    "pcgrad",
    "mgda_two_task",
    "unit_normalize",
    "TaskModel",
    "fresh_adamw_step",
    "private_adapt",
    "raw_shared_gradient",
    "run_p1_control",
    "run_p0",
    "run_p2",
    "run_p3",
]
