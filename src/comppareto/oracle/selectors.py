"""Global block layout, per-task selectors, and graph-family incidence.

Notation matches ``docs/theory/oracle-spec.md`` sections 1 and 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

GraphFamily = Literal[
    "disjoint", "full_overlap", "star", "chain", "partial", "random_sparse"
]

GRAPH_FAMILIES: tuple[GraphFamily, ...] = (
    "disjoint",
    "full_overlap",
    "star",
    "chain",
    "partial",
    "random_sparse",
)


class SelectorError(ValueError):
    """Raised when a selector or incidence matrix violates its contract."""


@dataclass(frozen=True)
class BlockLayout:
    """Global parameter blocks of prescribed widths."""

    block_widths: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.block_widths) == 0:
            raise SelectorError("at least one block is required")
        if any(w <= 0 for w in self.block_widths):
            raise SelectorError("block widths must be positive")

    @property
    def num_blocks(self) -> int:
        return len(self.block_widths)

    @property
    def total_dim(self) -> int:
        return sum(self.block_widths)

    def block_slice(self, block: int) -> slice:
        start = sum(self.block_widths[:block])
        return slice(start, start + self.block_widths[block])


def build_selector(layout: BlockLayout, blocks_touched: Sequence[int]) -> FloatArray:
    """Build a task selector ``P_i`` that lifts the touched blocks, in block order.

    Each row selects exactly one global coordinate and no coordinate is
    selected twice within this selector, matching ``QuadraticTask.selector``'s
    contract in ``src/comppareto/quadratic.py`` (validated independently here).
    """

    ordered = sorted(set(blocks_touched))
    if any(b < 0 or b >= layout.num_blocks for b in ordered):
        raise SelectorError("blocks_touched references a block outside the layout")
    p_i = sum(layout.block_widths[b] for b in ordered)
    selector = np.zeros((p_i, layout.total_dim), dtype=np.float64)
    row = 0
    for b in ordered:
        block_slice = layout.block_slice(b)
        for col in range(block_slice.start, block_slice.stop):
            selector[row, col] = 1.0
            row += 1
    validate_selector(selector)
    return selector


def validate_selector(selector: FloatArray) -> None:
    if selector.ndim != 2:
        raise SelectorError("selector must be two-dimensional")
    if not np.all((selector == 0.0) | (selector == 1.0)):
        raise SelectorError("selector must be binary")
    if not np.all(np.sum(selector, axis=1) == 1.0):
        raise SelectorError("each selector row must select exactly one global coordinate")
    if not np.all(np.sum(selector, axis=0) <= 1.0):
        raise SelectorError("selector must not select a global coordinate more than once")


def build_incidence(
    family: GraphFamily,
    num_tasks: int,
    num_blocks: int,
    rng: np.random.Generator,
    *,
    overlap_probability: float = 0.4,
    sparsity_probability: float = 0.12,
) -> BoolArray:
    """Build the task-by-block incidence matrix for a graph family.

    Rows are tasks, columns are blocks. ``P_i`` follows deterministically as
    the set of blocks where row ``i`` is ``True`` (see ``build_selector``).
    """

    if num_tasks < 2 or num_tasks > 8:
        raise SelectorError("num_tasks must be within [2, 8]")
    if num_blocks < 4 or num_blocks > 64:
        raise SelectorError("num_blocks must be within [4, 64]")

    if family == "disjoint":
        inc = _disjoint(num_tasks, num_blocks)
    elif family == "full_overlap":
        inc = np.ones((num_tasks, num_blocks), dtype=bool)
    elif family == "star":
        inc = _star(num_tasks, num_blocks)
    elif family == "chain":
        inc = _chain(num_tasks, num_blocks)
    elif family == "partial":
        inc = _partial(num_tasks, num_blocks, rng, overlap_probability)
    elif family == "random_sparse":
        inc = _random_sparse(num_tasks, num_blocks, rng, sparsity_probability)
    else:  # pragma: no cover - exhaustive Literal
        raise SelectorError(f"unknown graph family {family!r}")

    _validate_family(family, inc)
    return inc


def _disjoint(num_tasks: int, num_blocks: int) -> BoolArray:
    inc = np.zeros((num_tasks, num_blocks), dtype=bool)
    groups = np.array_split(np.arange(num_blocks), num_tasks)
    for i, group in enumerate(groups):
        inc[i, group] = True
    return inc


def _star(num_tasks: int, num_blocks: int) -> BoolArray:
    inc = np.zeros((num_tasks, num_blocks), dtype=bool)
    hub_size = max(1, -(-num_blocks // 8))  # ceil(num_blocks / 8)
    inc[:, :hub_size] = True
    spokes = np.array_split(np.arange(hub_size, num_blocks), num_tasks)
    for i, spoke in enumerate(spokes):
        inc[i, spoke] = True
    return inc


def _chain(num_tasks: int, num_blocks: int) -> BoolArray:
    inc = np.zeros((num_tasks, num_blocks), dtype=bool)
    # Sliding windows over the block line, one block of overlap between
    # consecutive tasks, covering the full range of blocks.
    boundaries = np.linspace(0, num_blocks, num_tasks + 1)
    starts = [int(round(b)) for b in boundaries[:-1]]
    ends = [int(round(b)) for b in boundaries[1:]]
    for i in range(num_tasks):
        lo = max(0, starts[i] - (1 if i > 0 else 0))
        hi = min(num_blocks, ends[i] + (1 if i < num_tasks - 1 else 0))
        inc[i, lo:hi] = True
    return inc


def _partial(
    num_tasks: int, num_blocks: int, rng: np.random.Generator, overlap_probability: float
) -> BoolArray:
    inc = rng.random((num_tasks, num_blocks)) < overlap_probability
    for i in range(num_tasks):
        if not inc[i].any():
            inc[i, rng.integers(num_blocks)] = True
    col_counts = inc.sum(axis=0)
    if not np.any(col_counts >= 2):
        # Force one shared block.
        b = rng.integers(num_blocks)
        rows = rng.choice(num_tasks, size=2, replace=False)
        inc[rows, b] = True
    if not np.any(col_counts == 1):
        # Force one private block.
        b = rng.integers(num_blocks)
        inc[:, b] = False
        inc[rng.integers(num_tasks), b] = True
    return inc


def _random_sparse(
    num_tasks: int, num_blocks: int, rng: np.random.Generator, sparsity_probability: float
) -> BoolArray:
    inc = rng.random((num_tasks, num_blocks)) < sparsity_probability
    for i in range(num_tasks):
        if not inc[i].any():
            inc[i, rng.integers(num_blocks)] = True
    return inc


def _validate_family(family: GraphFamily, inc: BoolArray) -> None:
    col_counts = inc.sum(axis=0)
    if not np.all(inc.sum(axis=1) >= 1):
        raise SelectorError(f"family {family!r} left a task with no blocks")
    if family == "disjoint" and np.any(col_counts > 1):
        raise SelectorError("disjoint family must not share any block")
    if family == "full_overlap" and not np.all(col_counts == inc.shape[0]):
        raise SelectorError("full_overlap family must select every block for every task")
    if family == "partial":
        if not np.any(col_counts >= 2):
            raise SelectorError("partial family must realize at least one shared block")
        if not np.any(col_counts == 1):
            raise SelectorError("partial family must realize at least one private block")


def build_task_selectors(
    layout: BlockLayout, incidence: BoolArray
) -> tuple[FloatArray, ...]:
    """Build one selector per task from an incidence matrix."""

    selectors = []
    for row in incidence:
        blocks_touched = [b for b, touched in enumerate(row) if touched]
        selectors.append(build_selector(layout, blocks_touched))
    return tuple(selectors)
