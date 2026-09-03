"""Deterministic snapshot/restore of the T215 diagnostic subspace's state.

Per ``reports/T215/first-report.md`` section 2, the state inventory for each
of the 3 subspace parameter tensors is:

- parameters (raw ``.data`` tensors);
- gradients (``.grad`` tensors, zeroed/None-checked before and after every
  diagnostic pass);
- AdamW optimizer state (``step``, ``exp_avg``, ``exp_avg_sq`` per parameter);
- RNG state (``torch.get_rng_state()``, ``torch.cuda.get_rng_state_all()``,
  Python ``random.getstate()``);
- data-order state (a fixed, explicit index list per batch).

Snapshot is deep-copy-on-CPU (``.clone().cpu()`` for tensors,
``copy.deepcopy`` for RNG state objects), taken immediately before every
diagnostic transition and reasserted immediately after. This module is pure
PyTorch (+ Python stdlib) and requires neither GPU nor the real Show-o2
model, so it is exercised directly by ``tests/adapters/showo2/`` on small
synthetic modules.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW


@dataclass(frozen=True)
class ParamSnapshot:
    """One parameter tensor's snapshot: value, grad, and owning AdamW state."""

    name: str
    data: torch.Tensor
    grad: torch.Tensor | None
    opt_step: torch.Tensor | None
    opt_exp_avg: torch.Tensor | None
    opt_exp_avg_sq: torch.Tensor | None


@dataclass(frozen=True)
class RngSnapshot:
    """Snapshot of every RNG stream the diagnostic protocol touches."""

    torch_rng_state: torch.Tensor
    cuda_rng_states: list[torch.Tensor]
    python_random_state: tuple[Any, ...]


@dataclass(frozen=True)
class DataOrderSnapshot:
    """Fixed, explicit per-batch index list (no stateful shuffling sampler)."""

    indices: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class SubspaceSnapshot:
    """Full snapshot for the declared 3-tensor diagnostic subspace."""

    params: tuple[ParamSnapshot, ...]
    rng: RngSnapshot
    data_order: DataOrderSnapshot


def _param_by_name(module: nn.Module, name: str) -> nn.Parameter:
    for pname, p in module.named_parameters():
        if pname == name:
            return p
    raise KeyError(f"parameter {name!r} not found in module")


def snapshot_params(
    module: nn.Module,
    param_names: list[str],
    optimizer: AdamW | None = None,
) -> tuple[ParamSnapshot, ...]:
    """Deep-copy-on-CPU snapshot of the named parameters, their grads, and
    their owning AdamW optimizer state (if ``optimizer`` is given and has
    state for that parameter).
    """

    snapshots: list[ParamSnapshot] = []
    for name in param_names:
        p = _param_by_name(module, name)
        grad = p.grad.detach().clone().cpu() if p.grad is not None else None
        opt_step = opt_exp_avg = opt_exp_avg_sq = None
        if optimizer is not None:
            state = optimizer.state.get(p)
            if state:
                step = state.get("step")
                opt_step = (
                    step.detach().clone().cpu()
                    if torch.is_tensor(step)
                    else (torch.tensor(step) if step is not None else None)
                )
                exp_avg = state.get("exp_avg")
                opt_exp_avg = exp_avg.detach().clone().cpu() if exp_avg is not None else None
                exp_avg_sq = state.get("exp_avg_sq")
                opt_exp_avg_sq = (
                    exp_avg_sq.detach().clone().cpu() if exp_avg_sq is not None else None
                )
        snapshots.append(
            ParamSnapshot(
                name=name,
                data=p.data.detach().clone().cpu(),
                grad=grad,
                opt_step=opt_step,
                opt_exp_avg=opt_exp_avg,
                opt_exp_avg_sq=opt_exp_avg_sq,
            )
        )
    return tuple(snapshots)


def snapshot_rng(cuda_available: bool = torch.cuda.is_available()) -> RngSnapshot:
    cuda_states: list[torch.Tensor] = []
    if cuda_available and torch.cuda.is_available():
        cuda_states = [s.clone().cpu() for s in torch.cuda.get_rng_state_all()]
    return RngSnapshot(
        torch_rng_state=torch.get_rng_state().clone(),
        cuda_rng_states=cuda_states,
        python_random_state=copy.deepcopy(random.getstate()),
    )


def snapshot_data_order(indices: dict[str, list[int]]) -> DataOrderSnapshot:
    return DataOrderSnapshot(indices={k: tuple(v) for k, v in indices.items()})


def snapshot_subspace(
    module: nn.Module,
    param_names: list[str],
    optimizer: AdamW | None = None,
    data_indices: dict[str, list[int]] | None = None,
) -> SubspaceSnapshot:
    return SubspaceSnapshot(
        params=snapshot_params(module, param_names, optimizer),
        rng=snapshot_rng(),
        data_order=snapshot_data_order(data_indices or {}),
    )


def restore_params(
    module: nn.Module,
    snapshots: tuple[ParamSnapshot, ...],
    optimizer: AdamW | None = None,
) -> None:
    """Reassert every parameter/grad/optimizer-state tensor from ``snapshots``.

    This mutates ``module`` (and ``optimizer``, if given) in place -- it is
    the rollback half of the reversible-diagnostic contract.
    """

    for snap in snapshots:
        p = _param_by_name(module, snap.name)
        with torch.no_grad():
            p.data.copy_(snap.data.to(p.data.device, p.data.dtype))
        if snap.grad is None:
            p.grad = None
        else:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            with torch.no_grad():
                p.grad.copy_(snap.grad.to(p.grad.device, p.grad.dtype))
        if optimizer is not None and snap.opt_step is not None:
            state = optimizer.state.setdefault(p, {})
            state["step"] = snap.opt_step.to(p.data.device).clone()
            if snap.opt_exp_avg is not None:
                state["exp_avg"] = snap.opt_exp_avg.to(p.data.device, p.data.dtype).clone()
            if snap.opt_exp_avg_sq is not None:
                state["exp_avg_sq"] = snap.opt_exp_avg_sq.to(p.data.device, p.data.dtype).clone()


def restore_rng(snapshot: RngSnapshot) -> None:
    torch.set_rng_state(snapshot.torch_rng_state.clone())
    if snapshot.cuda_rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.clone() for s in snapshot.cuda_rng_states])
    random.setstate(copy.deepcopy(snapshot.python_random_state))


def restore_subspace(
    module: nn.Module,
    snapshot: SubspaceSnapshot,
    optimizer: AdamW | None = None,
) -> None:
    restore_params(module, snapshot.params, optimizer)
    restore_rng(snapshot.rng)


@dataclass(frozen=True)
class ParamCompareResult:
    name: str
    data_max_abs_diff: float
    data_matches: bool
    grad_max_abs_diff: float | None
    grad_matches: bool
    opt_step_matches: bool
    opt_exp_avg_max_abs_diff: float | None
    opt_exp_avg_matches: bool
    opt_exp_avg_sq_max_abs_diff: float | None
    opt_exp_avg_sq_matches: bool

    @property
    def all_matches(self) -> bool:
        return (
            self.data_matches
            and self.grad_matches
            and self.opt_step_matches
            and self.opt_exp_avg_matches
            and self.opt_exp_avg_sq_matches
        )


def _dtype_tolerance(dtype: torch.dtype) -> float:
    """Declared dtype-appropriate exact/numerical tolerance for restore checks.

    fp32/fp64 restoration goes through an exact ``.copy_`` round-trip
    (no re-quantization), so the only expected nonzero difference is
    floating round-off from the CPU<->device clone; fp16/bf16 use a looser
    bound consistent with their reduced mantissa.
    """

    if dtype in (torch.float64,):
        return 0.0
    if dtype in (torch.float32,):
        return 1e-12
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3
    return 0.0


def compare_param_snapshot(
    module: nn.Module,
    snapshot: ParamSnapshot,
    optimizer: AdamW | None = None,
) -> ParamCompareResult:
    """Compare the module/optimizer's CURRENT state against ``snapshot``.

    Floating tensors (data, grad, moments) are compared within
    :func:`_dtype_tolerance`; counters (``step``) must match exactly.
    """

    p = _param_by_name(module, snapshot.name)
    current_data = p.data.detach().cpu()
    tol = _dtype_tolerance(current_data.dtype)
    data_diff = float((current_data - snapshot.data).abs().max()) if current_data.numel() else 0.0
    data_matches = data_diff <= tol

    grad_diff: float | None = None
    grad_matches = True
    current_grad = p.grad.detach().cpu() if p.grad is not None else None
    if snapshot.grad is None and current_grad is None:
        grad_matches = True
    elif snapshot.grad is None or current_grad is None:
        grad_matches = False
    else:
        grad_diff = float((current_grad - snapshot.grad).abs().max()) if current_grad.numel() else 0.0
        grad_matches = grad_diff <= _dtype_tolerance(current_grad.dtype)

    opt_step_matches = True
    exp_avg_diff: float | None = None
    exp_avg_matches = True
    exp_avg_sq_diff: float | None = None
    exp_avg_sq_matches = True
    if optimizer is not None:
        state = optimizer.state.get(p, {})
        cur_step = state.get("step")
        if snapshot.opt_step is None and cur_step is None:
            opt_step_matches = True
        elif snapshot.opt_step is None or cur_step is None:
            opt_step_matches = False
        else:
            cur_step_val = cur_step.detach().cpu() if torch.is_tensor(cur_step) else torch.tensor(cur_step)
            opt_step_matches = bool((cur_step_val == snapshot.opt_step).all())

        cur_exp_avg = state.get("exp_avg")
        if snapshot.opt_exp_avg is not None and cur_exp_avg is not None:
            exp_avg_diff = float((cur_exp_avg.detach().cpu() - snapshot.opt_exp_avg).abs().max())
            exp_avg_matches = exp_avg_diff <= _dtype_tolerance(cur_exp_avg.dtype)
        elif snapshot.opt_exp_avg is not None or cur_exp_avg is not None:
            exp_avg_matches = False

        cur_exp_avg_sq = state.get("exp_avg_sq")
        if snapshot.opt_exp_avg_sq is not None and cur_exp_avg_sq is not None:
            exp_avg_sq_diff = float(
                (cur_exp_avg_sq.detach().cpu() - snapshot.opt_exp_avg_sq).abs().max()
            )
            exp_avg_sq_matches = exp_avg_sq_diff <= _dtype_tolerance(cur_exp_avg_sq.dtype)
        elif snapshot.opt_exp_avg_sq is not None or cur_exp_avg_sq is not None:
            exp_avg_sq_matches = False

    return ParamCompareResult(
        name=snapshot.name,
        data_max_abs_diff=data_diff,
        data_matches=data_matches,
        grad_max_abs_diff=grad_diff,
        grad_matches=grad_matches,
        opt_step_matches=opt_step_matches,
        opt_exp_avg_max_abs_diff=exp_avg_diff,
        opt_exp_avg_matches=exp_avg_matches,
        opt_exp_avg_sq_max_abs_diff=exp_avg_sq_diff,
        opt_exp_avg_sq_matches=exp_avg_sq_matches,
    )


def compare_rng_snapshot(current: RngSnapshot, reference: RngSnapshot) -> bool:
    """RNG/counter state must match EXACTLY (not within tolerance) per the
    frozen protocol ("restore counters, RNG, and data-order state exactly").
    """

    if not torch.equal(current.torch_rng_state, reference.torch_rng_state):
        return False
    if len(current.cuda_rng_states) != len(reference.cuda_rng_states):
        return False
    for cur, ref in zip(current.cuda_rng_states, reference.cuda_rng_states):
        if not torch.equal(cur, ref):
            return False
    return current.python_random_state == reference.python_random_state


def compare_data_order_snapshot(current: DataOrderSnapshot, reference: DataOrderSnapshot) -> bool:
    return current.indices == reference.indices


def verify_rollback(
    module: nn.Module,
    snapshot: SubspaceSnapshot,
    optimizer: AdamW | None = None,
    data_indices: dict[str, list[int]] | None = None,
) -> tuple[bool, list[ParamCompareResult]]:
    """Full restore-correctness check: every parameter/grad/optimizer tensor
    within dtype tolerance, RNG and data-order state exactly.

    Returns ``(all_passed, per_param_results)``.
    """

    per_param = [compare_param_snapshot(module, s, optimizer) for s in snapshot.params]
    params_ok = all(r.all_matches for r in per_param)
    rng_ok = compare_rng_snapshot(snapshot_rng(), snapshot.rng)
    data_order_ok = compare_data_order_snapshot(
        snapshot_data_order(data_indices or {}), snapshot.data_order
    )
    return (params_ok and rng_ok and data_order_ok), per_param
