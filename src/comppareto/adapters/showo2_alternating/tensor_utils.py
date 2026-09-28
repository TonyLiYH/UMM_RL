"""Deterministic flatten/unflatten helpers for lists of parameter tensors.

Order-preserving: ``unflatten(flatten(params), params)`` recovers a list of
tensors with the same shapes as ``params``, and repeated calls with the same
input list produce identical output, which is required for the shared
candidate-direction arithmetic in ``negotiators.py`` (every gradient/direction
is a flat vector over the *same* declared parameter ordering).
"""

from __future__ import annotations

from typing import Iterable, List

import torch


def flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of tensors (e.g. per-parameter gradients) into one 1-D vector."""
    parts = [t.reshape(-1) for t in tensors]
    if not parts:
        raise ValueError("flatten() requires at least one tensor")
    return torch.cat(parts, dim=0)


def unflatten(vector: torch.Tensor, like: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    """Split ``vector`` back into tensors with the shapes of ``like``, in order."""
    like = list(like)
    out: List[torch.Tensor] = []
    offset = 0
    for t in like:
        n = t.numel()
        out.append(vector[offset : offset + n].reshape(t.shape))
        offset += n
    if offset != vector.numel():
        raise ValueError(
            f"vector has {vector.numel()} elements but `like` tensors sum to {offset}"
        )
    return out
