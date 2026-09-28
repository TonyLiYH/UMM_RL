"""Parameter block registry: explicit shared/private ownership and overlap IDs.

Per `AGENTS.md` ("Model adapters must expose shared and private parameter
blocks explicitly") and this task's pass/fail gate ("ownership must be
complete"), every trainable parameter of the model under instrumentation must
belong to exactly one declared block, and every block must declare exactly
which tasks' forward pass touches it (its *overlap set*). A block touched by
two or more tasks is a genuine sharing point and needs conflict-aware
negotiation; a block touched by exactly one task is that task's private
block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .params import Parameter


class OwnershipError(ValueError):
    """Raised when parameter ownership is incomplete or ambiguous."""


@dataclass(frozen=True)
class ParameterBlock:
    """One named, contiguous-in-declaration group of parameters.

    ``overlap_tasks`` is the tuple of task IDs whose forward pass uses this
    block's parameters. ``overlap_id`` is the canonical string form of that
    set, used to group blocks with an identical sharing pattern.
    """

    block_id: str
    parameters: tuple[Parameter, ...]
    overlap_tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.block_id:
            raise OwnershipError("block_id must be non-empty")
        if not self.overlap_tasks:
            raise OwnershipError(f"block {self.block_id!r} has no owning tasks")
        if len(set(self.overlap_tasks)) != len(self.overlap_tasks):
            raise OwnershipError(f"block {self.block_id!r} lists duplicate owning tasks")

    @property
    def overlap_id(self) -> str:
        return "+".join(self.overlap_tasks)

    @property
    def is_shared(self) -> bool:
        return len(self.overlap_tasks) > 1

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters)


@dataclass
class BlockRegistry:
    """Ordered collection of :class:`ParameterBlock` with completeness checks."""

    blocks: tuple[ParameterBlock, ...] = field(default_factory=tuple)

    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.block_id for block in self.blocks)

    def blocks_for_task(self, task_id: str) -> tuple[ParameterBlock, ...]:
        return tuple(b for b in self.blocks if task_id in b.overlap_tasks)

    def shared_blocks(self) -> tuple[ParameterBlock, ...]:
        return tuple(b for b in self.blocks if b.is_shared)

    def all_task_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for block in self.blocks:
            for task_id in block.overlap_tasks:
                seen.setdefault(task_id, None)
        return tuple(seen)

    def validate_ownership(self, trainable_parameters: tuple[Parameter, ...]) -> None:
        """Raise :class:`OwnershipError` unless every trainable parameter
        belongs to exactly one declared block (no gaps, no double ownership).
        """
        by_id = {id(p): p for p in trainable_parameters}
        assigned: dict[int, str] = {}
        for block in self.blocks:
            for parameter in block.parameters:
                key = id(parameter)
                if key not in by_id:
                    raise OwnershipError(
                        f"block {block.block_id!r} references a parameter that "
                        "is not in the trainable-parameter set"
                    )
                if key in assigned:
                    raise OwnershipError(
                        f"parameter is owned by both block {assigned[key]!r} and "
                        f"{block.block_id!r}"
                    )
                assigned[key] = block.block_id
        unassigned = [p for p in trainable_parameters if id(p) not in assigned]
        if unassigned:
            total = sum(p.numel() for p in unassigned)
            raise OwnershipError(
                f"{len(unassigned)} trainable parameter tensor(s) "
                f"({total} scalar values) are not assigned to any block"
            )

    def unassigned_trainable_parameters(
        self, trainable_parameters: tuple[Parameter, ...]
    ) -> int:
        """Count of trainable scalar values not covered by any block (0 on a
        valid registry; used to populate ``ownership.unassigned_trainable_parameters``
        even when the caller wants a soft count rather than a raised error).
        """
        by_id = {id(p): p for p in trainable_parameters}
        assigned: set[int] = set()
        for block in self.blocks:
            for parameter in block.parameters:
                if id(parameter) in by_id:
                    assigned.add(id(parameter))
        return sum(p.numel() for p in trainable_parameters if id(p) not in assigned)
