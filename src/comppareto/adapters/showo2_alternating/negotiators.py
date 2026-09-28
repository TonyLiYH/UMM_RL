"""Shared candidate-direction negotiators: raw SUM, normalized SUM, PCGrad, MGDA.

All functions operate on flat 1-D gradient vectors over the *same* declared
shared-parameter ordering (see ``tensor_utils.flatten``). Formulas match
``reports/T216/first-report.md`` section 5, which was published and pushed
before any GPU execution, so these are not selected or tuned after observing
any real-model outcome.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

_EPS_FLOOR = 1e-12


def unit_normalize(d: torch.Tensor, *, eps: float = _EPS_FLOOR) -> torch.Tensor:
    """Rescale to unit L2 norm; the declared common trust-region metric.

    If ``d`` is (near-)zero, returns ``d`` unchanged (a zero direction stays
    zero rather than dividing by ~0) — this is logged by the caller, not
    silently hidden.
    """
    norm = torch.linalg.vector_norm(d)
    if float(norm) <= eps:
        return d
    return d / norm


def raw_sum(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    out = grads[0].clone()
    for g in grads[1:]:
        out = out + g
    return out


def normalized_sum(grads: Sequence[torch.Tensor], *, eps: float = _EPS_FLOOR) -> torch.Tensor:
    out = torch.zeros_like(grads[0])
    for g in grads:
        n = torch.linalg.vector_norm(g)
        if float(n) > eps:
            out = out + g / n
        # else: this task's term is treated as exactly zero (guarded fallback,
        # declared in first-report.md section 5), not divided by near-zero norm.
    return out


def _pcgrad_pair(g_a: torch.Tensor, g_b: torch.Tensor) -> torch.Tensor:
    dot = torch.dot(g_a, g_b)
    if float(dot) < 0.0:
        b_sq = torch.dot(g_b, g_b)
        if float(b_sq) <= _EPS_FLOOR:
            return g_a.clone()
        return g_a - (dot / b_sq) * g_b
    return g_a.clone()


def pcgrad(grads: Sequence[torch.Tensor], *, order: Sequence[int] | None = None) -> torch.Tensor:
    """Two-task PCGrad with an explicit, frozen task order.

    ``order`` is a permutation of ``range(len(grads))`` declaring which task's
    gradient is projected against which; the default order is the identity
    (task 0 first). Callers must pass an explicit reversed order separately
    for the declared sensitivity check, per the task file's "PCGrad with a
    frozen task order and a separately declared reversed-order sensitivity
    check."
    """
    if len(grads) != 2:
        raise NotImplementedError("pcgrad() here implements the mandatory two-task case only")
    idx = list(order) if order is not None else [0, 1]
    a, b = idx
    g_a_proj = _pcgrad_pair(grads[a], grads[b])
    g_b_proj = _pcgrad_pair(grads[b], grads[a])
    out = torch.zeros_like(grads[0])
    out_terms = {a: g_a_proj, b: g_b_proj}
    for k in sorted(out_terms):
        out = out + out_terms[k]
    return out


def mgda_two_task(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Exact closed-form two-task MGDA convex combination.

    alpha* = clip( (g2-g1).g2 / ||g1-g2||^2, 0, 1 ), d = alpha*g1 + (1-alpha*)g2.
    Degenerate g1 == g2 (or ||g1-g2||^2 ~ 0) falls back to alpha* = 0.5.
    """
    diff = g1 - g2
    denom = torch.dot(diff, diff)
    if float(denom) <= _EPS_FLOOR:
        alpha = 0.5
    else:
        alpha = float(torch.clamp(torch.dot(-diff, g2) / denom, 0.0, 1.0))
    return alpha * g1 + (1.0 - alpha) * g2


def all_mandatory_directions(
    grads: Sequence[torch.Tensor], *, reversed_order: bool = False
) -> dict[str, torch.Tensor]:
    """Convenience: build every mandatory shared candidate direction for P0/P2."""
    if len(grads) != 2:
        raise NotImplementedError("only the mandatory two-task case is implemented")
    order = [1, 0] if reversed_order else [0, 1]
    return {
        "raw_sum": raw_sum(grads),
        "normalized_sum": normalized_sum(grads),
        "pcgrad": pcgrad(grads, order=order),
        "mgda": mgda_two_task(grads[0], grads[1]),
    }
