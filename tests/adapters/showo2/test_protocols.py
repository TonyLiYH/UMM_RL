"""Unit tests for ``comppareto.adapters.showo2.protocols`` (T215).

Uses a small synthetic, exactly-differentiable ``loss_fn(theta_s, theta_p,
batch) -> scalar`` (a quadratic form with an explicit theta_s/theta_p cross
term) instead of the real Show-o2 losses, so these tests run without GPU or
the real model. The cross term is what makes the rerun-response gradient
differ from the commit-response (stop-gradient) gradient -- exercised
explicitly below.
"""

from __future__ import annotations

import pytest
import torch

from comppareto.adapters.showo2 import protocols


def _quadratic_loss_fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Builds a ``loss_fn(theta_s, theta_p, batch)`` closure for fixed random
    matrices ``a`` (n_p x n_s), ``b`` (n_p x n_p), and vector ``c`` (n_p,).

    ``loss(theta_s, theta_p) = 0.5 * || a @ theta_s + b @ theta_p - c ||^2``

    The ``a @ theta_s`` term inside the private-parameter's own loss is what
    makes ``theta_p``'s adaptation gradient (and hence the rerun-response
    outer gradient) depend on ``theta_s`` -- exactly the effect
    commit-response's stop-gradient is designed to discard.
    """

    def loss_fn(theta_s: torch.Tensor, theta_p: torch.Tensor, batch: object) -> torch.Tensor:
        residual = a @ theta_s + b @ theta_p - c
        return 0.5 * (residual * residual).sum()

    return loss_fn


def _make_problem(seed: int = 0, n_s: int = 3, n_p: int = 2):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n_p, n_s, generator=gen)
    b = torch.randn(n_p, n_p, generator=gen)
    c_adapt = torch.randn(n_p, generator=gen)
    c_meta = torch.randn(n_p, generator=gen)
    theta_s0 = torch.randn(n_s, generator=gen)
    theta_p0 = torch.randn(n_p, generator=gen)
    loss_adapt_fn = _quadratic_loss_fn(a, b, c_adapt)
    loss_meta_fn = _quadratic_loss_fn(a, b, c_meta)

    def loss_fn(theta_s: torch.Tensor, theta_p: torch.Tensor, batch: str) -> torch.Tensor:
        if batch == "adapt":
            return loss_adapt_fn(theta_s, theta_p, batch)
        elif batch == "meta":
            return loss_meta_fn(theta_s, theta_p, batch)
        raise ValueError(batch)

    return loss_fn, theta_s0, theta_p0


LR = 0.1


def test_adamw_step_matches_torch_optim_adamw_first_step() -> None:
    torch.manual_seed(0)
    theta_p = torch.randn(5, requires_grad=True)
    grad_p = torch.randn(5)

    opt_state = protocols.AdamWMomentState.zeros_like(theta_p.detach())
    theta_new, opt_state_1 = protocols.adamw_step(
        theta_p.detach(), grad_p, opt_state, LR, differentiable=False
    )

    ref_param = theta_p.detach().clone().requires_grad_(True)
    ref_opt = torch.optim.AdamW([ref_param], lr=LR, weight_decay=0.0)
    ref_param.grad = grad_p.clone()
    ref_opt.step()

    assert torch.allclose(theta_new, ref_param.detach(), atol=1e-7)
    ref_state = ref_opt.state[ref_param]
    assert torch.allclose(opt_state_1.exp_avg, ref_state["exp_avg"], atol=1e-7)
    assert torch.allclose(opt_state_1.exp_avg_sq, ref_state["exp_avg_sq"], atol=1e-7)
    assert int(opt_state_1.step.item()) == int(ref_state["step"].item())


def test_adamw_step_matches_torch_optim_adamw_after_two_steps() -> None:
    torch.manual_seed(1)
    theta_p = torch.randn(4)
    grads = [torch.randn(4), torch.randn(4)]

    opt_state = protocols.AdamWMomentState.zeros_like(theta_p)
    theta_cur = theta_p.clone()
    for g in grads:
        theta_cur, opt_state = protocols.adamw_step(
            theta_cur, g, opt_state, LR, differentiable=False
        )

    ref_param = theta_p.clone().requires_grad_(True)
    ref_opt = torch.optim.AdamW([ref_param], lr=LR, weight_decay=0.0)
    for g in grads:
        ref_param.grad = g.clone()
        ref_opt.step()

    assert torch.allclose(theta_cur, ref_param.detach(), atol=1e-6)


def test_compute_raw_gradient_matches_manual_autograd() -> None:
    loss_fn, theta_s0, theta_p0 = _make_problem(seed=2)
    result = protocols.compute_raw(loss_fn, theta_s0, theta_p0, meta_batch="meta")

    theta_s_ref = theta_s0.detach().clone().requires_grad_(True)
    loss_ref = loss_fn(theta_s_ref, theta_p0.detach(), "meta")
    (grad_ref,) = torch.autograd.grad(loss_ref, theta_s_ref)

    assert torch.allclose(result.grad_theta_s, grad_ref, atol=1e-7)
    assert result.loss_adapt is None


def test_commit_response_matches_hand_rolled_stop_gradient_reference() -> None:
    loss_fn, theta_s0, theta_p0 = _make_problem(seed=3)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    result = protocols.compute_commit_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=1
    )

    # Hand-rolled reference: adapt theta_p at the FIXED theta_s0 (no grad
    # tracking needed on theta_s for the inner loop), then differentiate the
    # meta loss w.r.t. theta_s only, with theta_p held fixed at its adapted
    # value.
    theta_p_req = theta_p0.detach().clone().requires_grad_(True)
    loss_adapt_ref = loss_fn(theta_s0.detach(), theta_p_req, "adapt")
    (grad_p_ref,) = torch.autograd.grad(loss_adapt_ref, theta_p_req)
    theta_p_1_ref, _ = protocols.adamw_step(
        theta_p_req.detach(), grad_p_ref, opt_state0, LR, differentiable=False
    )

    theta_s_req = theta_s0.detach().clone().requires_grad_(True)
    loss_meta_ref = loss_fn(theta_s_req, theta_p_1_ref, "meta")
    (grad_theta_s_ref,) = torch.autograd.grad(loss_meta_ref, theta_s_req)

    assert torch.allclose(result.grad_theta_s, grad_theta_s_ref, atol=1e-6)
    assert torch.allclose(result.theta_p_after, theta_p_1_ref, atol=1e-6)


def test_rerun_response_matches_commit_response_at_k1_due_to_adamw_eps_saturation() -> None:
    """At k_steps=1 starting from zero AdamW moments, the update reduces to
    ``theta_p - lr * grad_p / (|grad_p| + eps)`` (bias corrections cancel
    exactly on step 1), whose derivative w.r.t. ``grad_p`` is
    ``eps / (|grad_p| + eps)^2``. With ``eps=1e-8`` and O(1)-scale gradients
    (as produced by this synthetic problem), that derivative underflows
    float32 precision, so ``theta_p_1``'s dependence on ``theta_s`` is
    numerically indistinguishable from zero and commit-response's
    stop-gradient and rerun-response's exact unroll agree at k_steps=1.
    The cross term's effect only becomes numerically visible once nonzero
    moment history accumulates -- i.e. k_steps>=2 (see the k_steps=2 test
    below).
    """

    loss_fn, theta_s0, theta_p0 = _make_problem(seed=4)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    commit = protocols.compute_commit_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=1
    )
    rerun = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=1
    )

    assert torch.allclose(commit.grad_theta_s, rerun.grad_theta_s, atol=1e-4)
    assert torch.allclose(commit.theta_p_after, rerun.theta_p_after, atol=1e-6)


def test_rerun_response_differs_from_commit_response_at_k2_due_to_cross_term() -> None:
    """At k_steps=2 the second inner AdamW step's moments carry a nonzero
    (non-underflowing) dependence of ``theta_p_1`` on ``theta_s`` forward
    into the second step, so rerun-response's outer gradient (which
    backprops through that dependency) must differ from commit-response's
    (which stop-gradients it every inner iteration).
    """

    loss_fn, theta_s0, theta_p0 = _make_problem(seed=4)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    commit = protocols.compute_commit_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=2
    )
    rerun = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=2
    )

    assert not torch.allclose(commit.grad_theta_s, rerun.grad_theta_s, atol=1e-4)
    # theta_p_after (the adapted private parameter value) should still match
    # between commit and rerun since both take numerically-identical
    # forward AdamW steps at the same theta_s0 -- only the outer *gradient*
    # differs (whether it flows through the adaptation or not).
    assert torch.allclose(commit.theta_p_after, rerun.theta_p_after, atol=1e-6)


def test_rerun_response_gradient_matches_full_manual_second_order_autograd() -> None:
    """Cross-check rerun-response's ``torch.autograd.grad(create_graph=True)``
    based implementation against a fully independent, hand-unrolled
    computation graph for k_steps=1.
    """

    loss_fn, theta_s0, theta_p0 = _make_problem(seed=5)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    rerun = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=1
    )

    theta_s_req = theta_s0.detach().clone().requires_grad_(True)
    theta_p_req = theta_p0.detach().clone().requires_grad_(True)
    loss_adapt = loss_fn(theta_s_req, theta_p_req, "adapt")
    (grad_p,) = torch.autograd.grad(loss_adapt, theta_p_req, create_graph=True)
    theta_p_1, _ = protocols.adamw_step(theta_p_req, grad_p, opt_state0, LR, differentiable=True)
    loss_meta = loss_fn(theta_s_req, theta_p_1, "meta")
    (grad_theta_s,) = torch.autograd.grad(loss_meta, theta_s_req)

    assert torch.allclose(rerun.grad_theta_s, grad_theta_s, atol=1e-6)


def test_rerun_response_k3_only_attempted_after_reasonable_k1_and_runs() -> None:
    loss_fn, theta_s0, theta_p0 = _make_problem(seed=6)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    k1 = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=1
    )
    assert torch.isfinite(k1.grad_theta_s).all()

    k3 = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=3
    )
    assert torch.isfinite(k3.grad_theta_s).all()
    assert k3.grad_theta_s.shape == k1.grad_theta_s.shape


def test_detach_moments_parameter_only_vs_complete_variant_differ_after_two_steps() -> None:
    """With k_steps=1 the initial moments are zero regardless of
    ``detach_moments``, so the two variants only diverge once there is a
    nonzero moment history flowing back through theta_s -- i.e. k_steps>=2.
    """

    loss_fn, theta_s0, theta_p0 = _make_problem(seed=7)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    complete = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=2,
        detach_moments=False,
    )
    param_only = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=2,
        detach_moments=True,
    )

    assert not torch.allclose(complete.grad_theta_s, param_only.grad_theta_s, atol=1e-6)
    # theta_p_after should still match: detach_moments only changes what the
    # gradient *w.r.t. theta_s* sees, not the forward-mode numerical update
    # (adamw_step's forward math is identical either way).
    assert torch.allclose(complete.theta_p_after, param_only.theta_p_after, atol=1e-6)


def test_rerun_loss_only_matches_differentiable_rerun_forward_loss_value() -> None:
    loss_fn, theta_s0, theta_p0 = _make_problem(seed=8)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    rerun = protocols.compute_rerun_response(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=1
    )
    loss_only = protocols.rerun_loss_only(
        loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR, k_steps=1
    )

    assert loss_only == pytest.approx(rerun.loss_meta, abs=1e-6)


def test_rerun_loss_only_is_non_differentiable_and_matches_perturbed_theta_s() -> None:
    loss_fn, theta_s0, theta_p0 = _make_problem(seed=9)
    opt_state0 = protocols.AdamWMomentState.zeros_like(theta_p0)

    base = protocols.rerun_loss_only(loss_fn, theta_s0, theta_p0, opt_state0, "adapt", "meta", LR)
    perturbed = protocols.rerun_loss_only(
        loss_fn, theta_s0 + 0.01, theta_p0, opt_state0, "adapt", "meta", LR
    )
    assert base != pytest.approx(perturbed, abs=1e-9)
