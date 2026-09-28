"""P0-P3 protocol implementations.

Each function is a pure tensor operation over cloned parameter tensors; none
of them touch the real model weights permanently — the caller is responsible
for snapshot/restore around every call.

TaskModel is the minimal interface this module requires from the model side;
the real Show-o2 runner (run_k1.py) passes concrete callables that satisfy
it, while the toy tests (tests/adapters/showo2_alternating/) pass toy
functions with the same signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.optim


# ---------------------------------------------------------------------------
# Public interface types
# ---------------------------------------------------------------------------

@dataclass
class TaskModel:
    """Minimal interface required by this module.

    ``shared_params``: list of leaf tensors representing the shared subspace
        (P_i theta); each must have ``requires_grad=True`` during backward.
    ``private_params``: list of leaf tensors for this task's private subspace.
    ``loss_fn``: callable(shared_params, private_params) -> scalar tensor
        with ``requires_grad=True`` graph; must not modify any parameter in
        place; grad accumulation is the caller's responsibility.
    ``task_id``: short string for logging ("und" or "gen").
    """

    shared_params: List[torch.Tensor]
    private_params: List[torch.Tensor]
    loss_fn: Callable[[List[torch.Tensor], List[torch.Tensor]], torch.Tensor]
    task_id: str


@dataclass
class RowResult:
    """Per-row measurement stored in the run manifest."""

    protocol: str           # P0 / P1 / P2 / P3
    direction: str          # raw_sum / normalized_sum / pcgrad / pcgrad_rev / mgda / control
    step_scale: float
    task_id: str
    loss_before: float
    loss_after: float
    delta_controlled: Optional[float]  # None for P0/P1
    delta_private: Optional[float]
    delta_total: Optional[float]
    finite: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_LR = 5e-5
_BETAS = (0.9, 0.999)
_EPS = 1e-8
_WD = 0.0


def fresh_adamw_step(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    *,
    lr: float = _LR,
    betas: Tuple[float, float] = _BETAS,
    eps: float = _EPS,
    weight_decay: float = _WD,
) -> None:
    """Apply exactly one AdamW step to ``params`` using pre-computed ``grads``.

    Step count is 1 (fresh optimizer semantics — no pre-existing momentum).
    All updates are applied in place. ``grads`` and ``params`` must be in the
    same order and have the same shapes.

    This is a literal first-order optimizer step: no ``create_graph=True``,
    no second-order graph, so the T215 MMU-NAN mechanism (eps-outside-sqrt +
    d(sqrt(x))/dx singularity at step=1 with create_graph=True) cannot occur
    by construction.
    """
    b1, b2 = betas
    # bias-correction at step=1
    bc1 = 1.0 - b1
    bc2 = math.sqrt(1.0 - b2)
    step_size = lr * bc2 / bc1  # =lr exactly at step=1

    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is None:
                continue
            # weight decay applied to raw param (decoupled)
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            exp_avg = torch.zeros_like(p)
            exp_avg_sq = torch.zeros_like(p)
            exp_avg.mul_(b1).add_(g, alpha=1.0 - b1)
            exp_avg_sq.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            denom = exp_avg_sq.sqrt().add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)


def private_adapt(
    task: TaskModel,
    K: int,
    *,
    lr: float = _LR,
) -> List[float]:
    """Run exactly K AdamW private steps in place on ``task.private_params``.

    Shared params are held fixed (they do not appear in the optimizer).
    Returns the list of K per-step loss values for logging.
    """
    opt = torch.optim.AdamW(
        task.private_params,
        lr=lr,
        betas=_BETAS,
        eps=_EPS,
        weight_decay=_WD,
    )
    losses = []
    for _ in range(K):
        opt.zero_grad()
        loss = task.loss_fn(task.shared_params, task.private_params)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def raw_shared_gradient(task: TaskModel) -> List[torch.Tensor]:
    """Compute the raw shared gradient at the current state; no parameter update.

    Returns a list of gradient tensors in the same order as ``task.shared_params``,
    each detached and cloned so the autograd graph is released.
    """
    for p in task.shared_params:
        if p.grad is not None:
            p.grad = None
    loss = task.loss_fn(task.shared_params, task.private_params)
    loss.backward()
    grads = []
    for p in task.shared_params:
        g = p.grad
        if g is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(g.detach().clone())
        p.grad = None
    return grads


# ---------------------------------------------------------------------------
# Protocol implementations
# ---------------------------------------------------------------------------

def run_p1_control(
    tasks: Sequence[TaskModel],
    K: int,
    *,
    snapshot_fn: Callable,
    restore_fn: Callable,
    step_scales: Sequence[float],
) -> Dict[str, object]:
    """P1: private-only control.

    Returns:
        {
          "control_losses": {task_id: L_i after K private steps},
          "loss_before":    {task_id: L_i before any step},
          "delta_private":  {task_id: L_after - L_before},
          "private_params_snaps": {task_id: cloned private param tensors after K steps},
        }
    All params are restored to snapshot0 at the end.
    """
    from .snapshot import snapshot_params, restore_params

    snap0 = snapshot_fn()
    results: Dict[str, object] = {"control_losses": {}, "loss_before": {}, "delta_private": {}}

    private_snaps = {}
    for task in tasks:
        restore_fn(snap0)
        with torch.no_grad():
            l_before = float(task.loss_fn(task.shared_params, task.private_params).detach())
        private_adapt(task, K)
        with torch.no_grad():
            l_after = float(task.loss_fn(task.shared_params, task.private_params).detach())
        private_snaps[task.task_id] = snapshot_params(task.private_params)
        results["control_losses"][task.task_id] = l_after
        results["loss_before"][task.task_id] = l_before
        results["delta_private"][task.task_id] = l_after - l_before
        restore_fn(snap0)

    results["private_params_snaps"] = private_snaps
    return results


def run_p0(
    tasks: Sequence[TaskModel],
    directions: Dict[str, torch.Tensor],
    step_scales: Sequence[float],
    control_results: Dict[str, object],
    *,
    snapshot_fn: Callable,
    restore_fn: Callable,
) -> List[RowResult]:
    """P0: simultaneous raw baseline.

    Private steps do NOT see the shared update (uses same P1 control private
    params for the private side), so Δ_controlled measures only the shared
    displacement effect. Returns one RowResult per (direction, step_scale, task).
    """
    from .negotiators import unit_normalize
    from .snapshot import restore_params

    snap0 = snapshot_fn()
    rows = []
    control_losses = control_results["control_losses"]
    private_snaps = control_results["private_params_snaps"]

    for dir_name, d_raw in directions.items():
        d = unit_normalize(d_raw)
        zero_dir = float(torch.linalg.vector_norm(d_raw)) < 1e-12
        for eta in step_scales:
            for task in tasks:
                restore_fn(snap0)
                # shared displacement
                with torch.no_grad():
                    for p, delta in zip(task.shared_params, _split_direction(d, task.shared_params)):
                        p.add_(delta, alpha=eta)
                # restore private to post-P1 state (private steps were at original shared)
                restore_params(task.private_params, private_snaps[task.task_id])
                with torch.no_grad():
                    l_after = float(task.loss_fn(task.shared_params, task.private_params).detach())
                finite = math.isfinite(l_after)
                delta_c = l_after - control_losses[task.task_id] if finite else None
                l_before = control_results["loss_before"][task.task_id]
                rows.append(RowResult(
                    protocol="P0",
                    direction=dir_name,
                    step_scale=eta,
                    task_id=task.task_id,
                    loss_before=l_before,
                    loss_after=l_after,
                    delta_controlled=delta_c,
                    delta_private=control_results["delta_private"][task.task_id],
                    delta_total=(l_after - l_before) if finite else None,
                    finite=finite,
                    notes="zero_direction" if zero_dir else "",
                ))
                restore_fn(snap0)

    return rows


def run_p2(
    tasks: Sequence[TaskModel],
    directions: Dict[str, torch.Tensor],
    step_scales: Sequence[float],
    control_results: Dict[str, object],
    K: int,
    *,
    snapshot_fn: Callable,
    restore_fn: Callable,
) -> List[RowResult]:
    """P2: shared-then-private (SP).

    For each (direction, eta): apply shared update theta' = theta + eta*d, then
    run K private steps from the original private snapshot at theta'.
    Δ_controlled = L_i(theta', s^{d,K}) - L_i_control (from P1).
    """
    from .negotiators import unit_normalize
    from .snapshot import snapshot_params, restore_params

    snap0 = snapshot_fn()
    rows = []
    control_losses = control_results["control_losses"]
    loss_before = control_results["loss_before"]

    for dir_name, d_raw in directions.items():
        d = unit_normalize(d_raw)
        zero_dir = float(torch.linalg.vector_norm(d_raw)) < 1e-12
        for eta in step_scales:
            for task in tasks:
                restore_fn(snap0)
                # shared displacement
                with torch.no_grad():
                    for p, delta in zip(task.shared_params, _split_direction(d, task.shared_params)):
                        p.add_(delta, alpha=eta)
                # K private steps at new shared state, from original private snapshot
                private_adapt(task, K)
                with torch.no_grad():
                    l_after = float(task.loss_fn(task.shared_params, task.private_params).detach())
                finite = math.isfinite(l_after)
                delta_c = l_after - control_losses[task.task_id] if finite else None
                l_b = loss_before[task.task_id]
                rows.append(RowResult(
                    protocol="P2",
                    direction=dir_name,
                    step_scale=eta,
                    task_id=task.task_id,
                    loss_before=l_b,
                    loss_after=l_after,
                    delta_controlled=delta_c,
                    delta_private=control_results["delta_private"][task.task_id],
                    delta_total=(l_after - l_b) if finite else None,
                    finite=finite,
                    notes="zero_direction" if zero_dir else "",
                ))
                restore_fn(snap0)

    return rows


def run_p3(
    tasks: Sequence[TaskModel],
    commit_directions: Dict[str, torch.Tensor],
    step_scales: Sequence[float],
    control_results: Dict[str, object],
    K: int,
    *,
    snapshot_fn: Callable,
    restore_fn: Callable,
) -> List[RowResult]:
    """P3: virtual private-then-shared commit.

    Pre-adaptation (K private steps at original state) is the same computation
    as P1's control arm; commit gradients are computed at the original shared
    state with stop-gradient on the virtual private state. Virtual private
    transition is restored before any shared displacement is applied.

    For each (direction, eta): apply shared update from ORIGINAL snapshot,
    then K fresh private steps from original private snapshot.
    """
    from .negotiators import unit_normalize

    snap0 = snapshot_fn()
    rows = []
    control_losses = control_results["control_losses"]
    loss_before = control_results["loss_before"]

    for dir_name, d_raw in commit_directions.items():
        d = unit_normalize(d_raw)
        zero_dir = float(torch.linalg.vector_norm(d_raw)) < 1e-12
        for eta in step_scales:
            for task in tasks:
                restore_fn(snap0)
                # shared displacement at ORIGINAL state (virtual private transition already restored)
                with torch.no_grad():
                    for p, delta in zip(task.shared_params, _split_direction(d, task.shared_params)):
                        p.add_(delta, alpha=eta)
                # K fresh private steps from original private snapshot at new shared state
                private_adapt(task, K)
                with torch.no_grad():
                    l_after = float(task.loss_fn(task.shared_params, task.private_params).detach())
                finite = math.isfinite(l_after)
                delta_c = l_after - control_losses[task.task_id] if finite else None
                l_b = loss_before[task.task_id]
                rows.append(RowResult(
                    protocol="P3",
                    direction=dir_name,
                    step_scale=eta,
                    task_id=task.task_id,
                    loss_before=l_b,
                    loss_after=l_after,
                    delta_controlled=delta_c,
                    delta_private=control_results["delta_private"][task.task_id],
                    delta_total=(l_after - l_b) if finite else None,
                    finite=finite,
                    notes="zero_direction" if zero_dir else "",
                ))
                restore_fn(snap0)

    return rows


def _split_direction(
    d: torch.Tensor, shared_params: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Split flat shared-subspace direction vector back into per-tensor pieces."""
    out = []
    offset = 0
    for p in shared_params:
        n = p.numel()
        out.append(d[offset : offset + n].reshape(p.shape).to(p.dtype))
        offset += n
    if offset != d.numel():
        raise ValueError(
            f"direction vector has {d.numel()} elements but shared_params sum to {offset}"
        )
    return out
