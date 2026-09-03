"""Raw / commit-response / rerun-response gradient protocols (T215 sect. 4).

Generic over any differentiable ``loss_fn(theta_s, theta_p, batch) -> scalar``
so the same code path drives both the synthetic unit tests (small
``torch.nn.Module``s) and the real Show-o2 MMU/T2I losses via
``model_io.py``. Pure PyTorch; imports neither the real model nor CUDA.

Notation follows ``reports/T215/first-report.md`` section 4:
``theta_s`` = shared parameter tensor (e.g. ``fusion_proj``), differentiated
in the outer (meta) loss; ``theta_p`` = private parameter tensor (e.g.
``und_trans.layers[0]`` or ``diffusion_head_a[0]``), adapted on the inner
(adaptation) batch via one or more differentiable AdamW steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

LossFn = Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor]

ADAMW_BETA1 = 0.9
ADAMW_BETA2 = 0.999
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.0


@dataclass
class AdamWMomentState:
    """AdamW per-parameter moment state, kept explicit (not
    ``torch.optim.AdamW``) so the K=1/K=3 inner recurrence can optionally
    stay inside a ``create_graph=True`` autograd graph (rerun-response) or be
    detached at each step (commit-response / parameter-only variant).
    """

    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    @classmethod
    def zeros_like(cls, theta_p: torch.Tensor) -> "AdamWMomentState":
        return cls(
            step=torch.zeros((), dtype=torch.float32),
            exp_avg=torch.zeros_like(theta_p),
            exp_avg_sq=torch.zeros_like(theta_p),
        )

    def clone(self) -> "AdamWMomentState":
        return AdamWMomentState(
            step=self.step.clone(),
            exp_avg=self.exp_avg.clone(),
            exp_avg_sq=self.exp_avg_sq.clone(),
        )

    def detached(self) -> "AdamWMomentState":
        return AdamWMomentState(
            step=self.step.detach(),
            exp_avg=self.exp_avg.detach(),
            exp_avg_sq=self.exp_avg_sq.detach(),
        )


def adamw_step(
    theta_p: torch.Tensor,
    grad_p: torch.Tensor,
    opt_state: AdamWMomentState,
    lr: float,
    *,
    differentiable: bool,
    detach_moments: bool = False,
) -> tuple[torch.Tensor, AdamWMomentState]:
    """One exact AdamW update, matching ``torch==2.5.1`` semantics
    (``torch.optim.AdamW`` with default betas/eps, ``weight_decay=0`` inside
    this diagnostic's minimal subspace).

    ``differentiable=True`` keeps the moment-buffer recurrence in the
    autograd graph so backprop through this function reaches ``theta_p``,
    ``grad_p``, and (if ``grad_p`` itself depends on ``theta_s`` and was
    built with ``create_graph=True``) ``theta_s``.

    ``detach_moments`` implements T215's "parameter-only vs complete
    optimizer-state differentiation" variant (first-report section 4/6):
    when ``True``, ``opt_state.exp_avg``/``exp_avg_sq`` are detached before
    use in this step's update rule, so the moment-buffer recurrence itself is
    treated as a constant and only the raw ``theta_p - lr * (bias-corrected
    update)`` term stays differentiable in ``theta_p``/``grad_p``.
    """

    step = opt_state.step + 1
    exp_avg_in = opt_state.exp_avg.detach() if detach_moments else opt_state.exp_avg
    exp_avg_sq_in = opt_state.exp_avg_sq.detach() if detach_moments else opt_state.exp_avg_sq

    exp_avg = ADAMW_BETA1 * exp_avg_in + (1 - ADAMW_BETA1) * grad_p
    exp_avg_sq = ADAMW_BETA2 * exp_avg_sq_in + (1 - ADAMW_BETA2) * grad_p * grad_p

    bias_correction1 = 1 - ADAMW_BETA1 ** step
    bias_correction2 = 1 - ADAMW_BETA2 ** step
    denom = (exp_avg_sq / bias_correction2).sqrt() + ADAMW_EPS
    step_size = lr / bias_correction1

    theta_p_decayed = theta_p * (1 - lr * ADAMW_WEIGHT_DECAY) if ADAMW_WEIGHT_DECAY else theta_p
    theta_p_new = theta_p_decayed - step_size * (exp_avg / denom)

    if not differentiable:
        theta_p_new = theta_p_new.detach()
        exp_avg = exp_avg.detach()
        exp_avg_sq = exp_avg_sq.detach()

    return theta_p_new, AdamWMomentState(step=step.detach(), exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)


@dataclass
class ProtocolResult:
    """One protocol's (raw / commit / rerun) measured outer gradient plus the
    intermediates needed for reporting and rollback verification.
    """

    protocol: str
    grad_theta_s: torch.Tensor
    loss_adapt: float | None
    loss_meta: float
    theta_p_after: torch.Tensor
    opt_state_after: AdamWMomentState


def compute_raw(
    loss_fn: LossFn,
    theta_s: torch.Tensor,
    theta_p_0: torch.Tensor,
    meta_batch: object,
) -> ProtocolResult:
    """Baseline sensitivity: no private adaptation, gradient of the meta loss
    at ``theta_p_0`` directly w.r.t. ``theta_s``.
    """

    theta_s_req = theta_s.detach().clone().requires_grad_(True)
    theta_p_frozen = theta_p_0.detach()
    loss_meta = loss_fn(theta_s_req, theta_p_frozen, meta_batch)
    (grad,) = torch.autograd.grad(loss_meta, theta_s_req)
    return ProtocolResult(
        protocol="raw",
        grad_theta_s=grad.detach(),
        loss_adapt=None,
        loss_meta=float(loss_meta.detach()),
        theta_p_after=theta_p_frozen.clone(),
        opt_state_after=AdamWMomentState.zeros_like(theta_p_0),
    )


def _inner_adapt(
    loss_fn: LossFn,
    theta_s: torch.Tensor,
    theta_p_0: torch.Tensor,
    opt_state_0: AdamWMomentState,
    adapt_batch: object,
    lr: float,
    k_steps: int,
    *,
    differentiable: bool,
    detach_moments: bool,
) -> tuple[torch.Tensor, AdamWMomentState, float]:
    """Run ``k_steps`` inner AdamW adaptation steps on ``theta_p`` at the
    (fixed or differentiable) ``theta_s``. Returns the final ``theta_p``,
    optimizer state, and the FIRST step's adaptation loss (for reporting).
    """

    theta_p = theta_p_0
    opt_state = opt_state_0
    first_loss: float | None = None
    for _ in range(k_steps):
        if not differentiable and not theta_p.requires_grad:
            # adamw_step (differentiable=False) returns a detached
            # theta_p_new each iteration, so for k_steps>=2 we must
            # re-enable grad tracking on it before the next
            # torch.autograd.grad call below -- otherwise it has no
            # grad_fn and is not itself a leaf requiring grad.
            theta_p = theta_p.detach().requires_grad_(True)
        loss_adapt = loss_fn(theta_s, theta_p, adapt_batch)
        if first_loss is None:
            first_loss = float(loss_adapt.detach())
        (grad_p,) = torch.autograd.grad(
            loss_adapt,
            theta_p,
            create_graph=differentiable,
            retain_graph=differentiable,
        )
        theta_p, opt_state = adamw_step(
            theta_p,
            grad_p,
            opt_state,
            lr,
            differentiable=differentiable,
            detach_moments=detach_moments,
        )
    assert first_loss is not None
    return theta_p, opt_state, first_loss


def compute_commit_response(
    loss_fn: LossFn,
    theta_s: torch.Tensor,
    theta_p_0: torch.Tensor,
    opt_state_0: AdamWMomentState,
    adapt_batch: object,
    meta_batch: object,
    lr: float,
    k_steps: int = 1,
) -> ProtocolResult:
    """Commit-response (stop-gradient): adapt ``theta_p`` at the current
    ``theta_s`` (non-differentiable inner loop), hold the resulting private
    state fixed, then differentiate the meta loss w.r.t. ``theta_s`` only.
    """

    theta_s_req = theta_s.detach().clone().requires_grad_(True)
    theta_p_1, opt_state_1, loss_adapt = _inner_adapt(
        loss_fn,
        theta_s_req.detach(),
        theta_p_0.detach().clone().requires_grad_(True),
        opt_state_0,
        adapt_batch,
        lr,
        k_steps,
        differentiable=False,
        detach_moments=False,
    )
    theta_p_1_detached = theta_p_1.detach()
    loss_meta = loss_fn(theta_s_req, theta_p_1_detached, meta_batch)
    (grad,) = torch.autograd.grad(loss_meta, theta_s_req)
    return ProtocolResult(
        protocol="commit",
        grad_theta_s=grad.detach(),
        loss_adapt=loss_adapt,
        loss_meta=float(loss_meta.detach()),
        theta_p_after=theta_p_1_detached.clone(),
        opt_state_after=opt_state_1,
    )


def compute_rerun_response(
    loss_fn: LossFn,
    theta_s: torch.Tensor,
    theta_p_0: torch.Tensor,
    opt_state_0: AdamWMomentState,
    adapt_batch: object,
    meta_batch: object,
    lr: float,
    k_steps: int = 1,
    *,
    detach_moments: bool = False,
) -> ProtocolResult:
    """Rerun-response (exact finite unroll): rerun the private response at
    the CANDIDATE ``theta_s`` and differentiate through the complete
    selected state transition (no detach in the inner loop).

    ``detach_moments`` selects T215's parameter-only (``True``) vs. complete
    optimizer-state (``False``) differentiation variant: complete keeps the
    AdamW moment-buffer recurrence itself in the ``create_graph=True`` chain,
    so ``grad_theta_s`` also flows through how ``theta_s`` affected
    ``exp_avg``/``exp_avg_sq``; parameter-only detaches the moments each
    inner step so only the raw update term stays in the graph.
    """

    theta_s_req = theta_s.detach().clone().requires_grad_(True)
    theta_p_1, opt_state_1, loss_adapt = _inner_adapt(
        loss_fn,
        theta_s_req,
        theta_p_0.detach().clone().requires_grad_(True),
        opt_state_0,
        adapt_batch,
        lr,
        k_steps,
        differentiable=True,
        detach_moments=detach_moments,
    )
    loss_meta = loss_fn(theta_s_req, theta_p_1, meta_batch)
    (grad,) = torch.autograd.grad(loss_meta, theta_s_req)
    return ProtocolResult(
        protocol="rerun",
        grad_theta_s=grad.detach(),
        loss_adapt=loss_adapt,
        loss_meta=float(loss_meta.detach()),
        theta_p_after=theta_p_1.detach().clone(),
        opt_state_after=opt_state_1.detached(),
    )


def rerun_loss_only(
    loss_fn: LossFn,
    theta_s: torch.Tensor,
    theta_p_0: torch.Tensor,
    opt_state_0: AdamWMomentState,
    adapt_batch: object,
    meta_batch: object,
    lr: float,
    k_steps: int = 1,
) -> float:
    """Non-differentiable rerun-response meta loss at a given ``theta_s``,
    used by the central finite-difference reference (first-report section 5:
    "using the rerun-response protocol...non-differentiable/create_graph=False
    since only the scalar loss is needed").
    """

    with torch.no_grad():
        theta_p = theta_p_0.clone()
        opt_state = opt_state_0.clone()
        for _ in range(k_steps):
            with torch.enable_grad():
                theta_p_req = theta_p.clone().requires_grad_(True)
                loss_adapt = loss_fn(theta_s, theta_p_req, adapt_batch)
                (grad_p,) = torch.autograd.grad(loss_adapt, theta_p_req)
            theta_p, opt_state = adamw_step(
                theta_p, grad_p, opt_state, lr, differentiable=False, detach_moments=False
            )
        loss_meta = loss_fn(theta_s, theta_p, meta_batch)
    return float(loss_meta)
