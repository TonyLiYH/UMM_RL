"""Stage 1 (continued): deterministic toy tests for parameter blocks with
shared and private modules, and ownership completeness.
"""

from __future__ import annotations

import numpy as np
import pytest

from comppareto.instrumentation.blocks import BlockRegistry, OwnershipError, ParameterBlock
from comppareto.instrumentation.params import Parameter


def _param(shape: tuple[int, ...]) -> Parameter:
    return Parameter(np.zeros(shape, dtype=np.float32))


def test_block_overlap_id_and_is_shared() -> None:
    shared = ParameterBlock("shared", (_param((2,)),), ("a", "b"))
    private = ParameterBlock("private_a", (_param((3,)),), ("a",))
    assert shared.overlap_id == "a+b"
    assert shared.is_shared is True
    assert private.overlap_id == "a"
    assert private.is_shared is False
    assert shared.param_count == 2
    assert private.param_count == 3


def test_block_rejects_empty_id_and_duplicate_owners() -> None:
    with pytest.raises(OwnershipError):
        ParameterBlock("", (_param((1,)),), ("a",))
    with pytest.raises(OwnershipError):
        ParameterBlock("b", (_param((1,)),), ())
    with pytest.raises(OwnershipError):
        ParameterBlock("b", (_param((1,)),), ("a", "a"))


def test_registry_blocks_for_task_and_shared_blocks() -> None:
    shared_param = _param((2,))
    private_a_param = _param((2,))
    private_b_param = _param((2,))
    registry = BlockRegistry(
        blocks=(
            ParameterBlock("shared", (shared_param,), ("a", "b")),
            ParameterBlock("private_a", (private_a_param,), ("a",)),
            ParameterBlock("private_b", (private_b_param,), ("b",)),
        )
    )
    assert registry.block_ids() == ("shared", "private_a", "private_b")
    assert [b.block_id for b in registry.blocks_for_task("a")] == ["shared", "private_a"]
    assert [b.block_id for b in registry.blocks_for_task("b")] == ["shared", "private_b"]
    assert [b.block_id for b in registry.shared_blocks()] == ["shared"]
    assert registry.all_task_ids() == ("a", "b")


def test_ownership_validation_passes_on_complete_registry() -> None:
    shared_param = _param((2,))
    private_param = _param((2,))
    registry = BlockRegistry(
        blocks=(
            ParameterBlock("shared", (shared_param,), ("a", "b")),
            ParameterBlock("private_a", (private_param,), ("a",)),
        )
    )
    registry.validate_ownership((shared_param, private_param))
    assert registry.unassigned_trainable_parameters((shared_param, private_param)) == 0


def test_ownership_validation_detects_unassigned_parameter() -> None:
    shared_param = _param((2,))
    orphan_param = _param((5,))
    registry = BlockRegistry(blocks=(ParameterBlock("shared", (shared_param,), ("a",)),))
    with pytest.raises(OwnershipError):
        registry.validate_ownership((shared_param, orphan_param))
    assert registry.unassigned_trainable_parameters((shared_param, orphan_param)) == 5


def test_ownership_validation_detects_double_ownership() -> None:
    shared_param = _param((2,))
    registry = BlockRegistry(
        blocks=(
            ParameterBlock("block_x", (shared_param,), ("a",)),
            ParameterBlock("block_y", (shared_param,), ("b",)),
        )
    )
    with pytest.raises(OwnershipError):
        registry.validate_ownership((shared_param,))


def test_ownership_validation_detects_unknown_parameter_reference() -> None:
    known = _param((2,))
    unknown = _param((2,))
    registry = BlockRegistry(blocks=(ParameterBlock("block", (unknown,), ("a",)),))
    with pytest.raises(OwnershipError):
        registry.validate_ownership((known,))
