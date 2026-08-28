"""Longdouble-precision recheck of the exact quadratic loss-change identity.

Addresses ``reports/T155/local-review.md`` R5's two float64 tolerance
failures: case_index 41 (task_index 4) and case_index 287 (task_index 1),
both ``momentum``/``unstable``. Does not relax the frozen 1e-9 relative
tolerance in ``crosscheck.py`` and does not remove either failure from
``failure_ledger.json``; this module only adds independent higher-precision
evidence for a local-reviewer decision.

NumPy's LAPACK-backed ``linalg`` routines (``solve``, ``eigvalsh``, ``inv``,
``cond``) reject ``np.longdouble`` arrays outright (``TypeError: array type
float128 is unsupported in linalg``), so every longdouble function below is
built only from ``matrix_power``/``matmul``/``@``/``hstack``/``vstack``,
which do preserve ``longdouble`` precision. Each mirrors the corresponding
float64 formula in ``momentum.py``/``hypergradient.py`` line for line -- a
deliberately independent re-derivation at extended precision, not a
wrapper around the float64 pipeline under investigation. ``OracleTask``
cannot be reused directly for the longdouble side since its
``__post_init__`` forces every field to ``float64``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from comppareto.oracle import generation as gen
from comppareto.oracle import stability as st
from comppareto.oracle.case import CaseSpec
from comppareto.oracle.hypergradient import exact_loss_change, quadratic_model, rerun_gradient
from comppareto.oracle.momentum import momentum_closed_form_state, momentum_sensitivity
from comppareto.oracle.noise import NoiseModel
from comppareto.oracle.seeds import case_seeds
from comppareto.oracle.selectors import BlockLayout, build_incidence
from comppareto.oracle.sweep import enumerate_cases
from comppareto.oracle.tasks import OracleTask

FloatArray = NDArray[np.float64]
LongArray = NDArray[np.longdouble]

KNOWN_FAILURES: tuple[tuple[int, int], ...] = ((41, 4), (287, 1))


def _hp(*arrays: FloatArray) -> tuple[LongArray, ...]:
    return tuple(np.asarray(a, dtype=np.longdouble) for a in arrays)


def _transition_jacobian_hp(c: LongArray, eta: float, beta: float) -> LongArray:
    d = c.shape[0]
    eye = np.eye(d, dtype=np.longdouble)
    top = np.hstack([eye - eta * c, -eta * beta * eye])
    bottom = np.hstack([c, beta * eye])
    return np.vstack([top, bottom])


def _input_jacobian_hp(h_phix: LongArray, eta: float) -> LongArray:
    return np.vstack([-eta * h_phix, h_phix])


def _affine_offset_hp(h_phix: LongArray, private_linear: LongArray, x_i: LongArray, eta: float) -> LongArray:
    const = h_phix @ x_i + private_linear
    return np.concatenate([-eta * const, const])


def closed_form_state_hp(
    c: LongArray,
    h_phix: LongArray,
    private_linear: LongArray,
    x_i: LongArray,
    phi_0: LongArray,
    v_0: LongArray,
    eta: float,
    beta: float,
    noise: LongArray,
) -> tuple[LongArray, LongArray]:
    """Longdouble mirror of :func:`momentum_closed_form_state`."""

    steps = noise.shape[0]
    d = c.shape[0]
    a = _transition_jacobian_hp(c, eta, beta)
    b = _affine_offset_hp(h_phix, private_linear, x_i, eta)
    u = np.linalg.matrix_power(a, steps) @ np.concatenate([phi_0, v_0])
    for j in range(steps):
        power = steps - 1 - j
        a_power = np.linalg.matrix_power(a, power)
        n_j = np.concatenate([-eta * noise[j], noise[j]])
        u = u + a_power @ (b + n_j)
    return u[:d], u[d:]


def sensitivity_hp(c: LongArray, h_phix: LongArray, eta: float, beta: float, steps: int) -> LongArray:
    """Longdouble mirror of :func:`momentum_sensitivity`."""

    a = _transition_jacobian_hp(c, eta, beta)
    b = _input_jacobian_hp(h_phix, eta)
    w = np.zeros((a.shape[0], b.shape[1]), dtype=np.longdouble)
    for l in range(steps):
        w = w + np.linalg.matrix_power(a, l) @ b
    return w


def quadratic_model_hp(
    h_xx: LongArray,
    h_xphi: LongArray,
    h_phiphi: LongArray,
    a_meta: LongArray,
    b_meta: LongArray,
    z_k_phi: LongArray,
    r_k_phi: LongArray,
) -> tuple[LongArray, LongArray]:
    """Longdouble mirror of :func:`quadratic_model`."""

    h_phix = h_xphi.T
    q = h_xx + h_xphi @ z_k_phi + z_k_phi.T @ h_phix + z_k_phi.T @ h_phiphi @ z_k_phi
    g_hat = a_meta + h_xphi @ r_k_phi + z_k_phi.T @ b_meta + z_k_phi.T @ (h_phiphi @ r_k_phi)
    return g_hat, q


def rerun_gradient_hp(
    a_meta: LongArray,
    h_xx: LongArray,
    h_xphi: LongArray,
    b_meta: LongArray,
    h_phiphi: LongArray,
    x_i: LongArray,
    phi_k: LongArray,
    z_k_phi: LongArray,
) -> LongArray:
    """Longdouble mirror of :func:`rerun_gradient`."""

    h_phix = h_xphi.T
    grad_x = a_meta + h_xx @ x_i + h_xphi @ phi_k
    grad_phi = b_meta + h_phix @ x_i + h_phiphi @ phi_k
    return grad_x + z_k_phi.T @ grad_phi


def exact_loss_change_hp(gradient_at_point: LongArray, q: LongArray, local_step: LongArray) -> np.longdouble:
    """Longdouble mirror of :func:`exact_loss_change`, kept unrounded (no ``float()`` cast)."""

    return gradient_at_point @ local_step + np.longdouble(0.5) * (local_step @ (q @ local_step))


def meta_loss_hp(
    a_meta: LongArray,
    h_xx: LongArray,
    h_xphi: LongArray,
    b_meta: LongArray,
    h_phiphi: LongArray,
    x_i: LongArray,
    phi_i: LongArray,
) -> np.longdouble:
    """Longdouble mirror of :meth:`OracleTask.meta_loss`."""

    half = np.longdouble(0.5)
    return (
        a_meta @ x_i
        + half * (x_i @ h_xx @ x_i)
        + x_i @ h_xphi @ phi_i
        + b_meta @ phi_i
        + half * (phi_i @ h_phiphi @ phi_i)
    )


def recheck_momentum_loss_change(
    task: OracleTask,
    x_i: FloatArray,
    phi_0: FloatArray,
    v_0: FloatArray,
    eta: float,
    beta: float,
    noise: FloatArray,
    step: FloatArray,
) -> dict:
    """Cross-check the momentum exact-loss-change identity at float64 and longdouble precision."""

    steps = noise.shape[0]
    d = task.private_dim

    phi_k_closed, _ = momentum_closed_form_state(task, x_i, phi_0, v_0, eta, beta, noise)
    w_k = momentum_sensitivity(task, eta, beta, steps)
    z_k_phi = w_k[:d]
    r_k_phi, _ = momentum_closed_form_state(task, np.zeros_like(x_i), phi_0, v_0, eta, beta, noise)
    g_hat_f64, q_f64 = quadratic_model(task, z_k_phi, r_k_phi)
    grad_f64 = rerun_gradient(task, x_i, phi_k_closed, z_k_phi)
    exact_delta_f64 = exact_loss_change(grad_f64, q_f64, step)

    def direct_loss(xi: FloatArray) -> float:
        phi_k, _ = momentum_closed_form_state(task, xi, phi_0, v_0, eta, beta, noise)
        return task.meta_loss(xi, phi_k)

    direct_delta_f64 = direct_loss(x_i + step) - direct_loss(x_i)
    f64_abs_error = abs(exact_delta_f64 - direct_delta_f64)
    # Denominator matches crosscheck.compare's convention exactly: the
    # reference passed to compare() is exact_delta, not direct_delta, so
    # crosscheck.py's frozen failure_ledger.json errors are relative to
    # |exact_delta|. Dividing by |direct_delta| instead would silently
    # reproduce a value close to, but not bit-identical with, the ledger.
    f64_relative_error = f64_abs_error / abs(exact_delta_f64)

    c_hp, h_phix_hp, h_xx_hp, h_xphi_hp, h_phiphi_hp, private_linear_hp, a_meta_hp, b_meta_hp = _hp(
        task.private_curvature,
        task.h_phix,
        task.h_xx,
        task.h_xphi,
        task.h_phiphi,
        task.private_linear,
        task.a_meta,
        task.b_meta,
    )
    x_i_hp, phi_0_hp, v_0_hp, noise_hp, step_hp = _hp(x_i, phi_0, v_0, noise, step)

    phi_k_hp, _ = closed_form_state_hp(c_hp, h_phix_hp, private_linear_hp, x_i_hp, phi_0_hp, v_0_hp, eta, beta, noise_hp)
    w_k_hp = sensitivity_hp(c_hp, h_phix_hp, eta, beta, steps)
    z_k_phi_hp = w_k_hp[:d]
    zero_x_hp = np.zeros_like(x_i_hp)
    r_k_phi_hp, _ = closed_form_state_hp(
        c_hp, h_phix_hp, private_linear_hp, zero_x_hp, phi_0_hp, v_0_hp, eta, beta, noise_hp
    )
    _, q_hp = quadratic_model_hp(h_xx_hp, h_xphi_hp, h_phiphi_hp, a_meta_hp, b_meta_hp, z_k_phi_hp, r_k_phi_hp)
    grad_hp = rerun_gradient_hp(a_meta_hp, h_xx_hp, h_xphi_hp, b_meta_hp, h_phiphi_hp, x_i_hp, phi_k_hp, z_k_phi_hp)
    exact_delta_hp = exact_loss_change_hp(grad_hp, q_hp, step_hp)

    def direct_loss_hp(xi_hp: LongArray) -> np.longdouble:
        phi_k_hp_, _ = closed_form_state_hp(
            c_hp, h_phix_hp, private_linear_hp, xi_hp, phi_0_hp, v_0_hp, eta, beta, noise_hp
        )
        return meta_loss_hp(a_meta_hp, h_xx_hp, h_xphi_hp, b_meta_hp, h_phiphi_hp, xi_hp, phi_k_hp_)

    direct_delta_hp = direct_loss_hp(x_i_hp + step_hp) - direct_loss_hp(x_i_hp)
    hp_abs_error = abs(exact_delta_hp - direct_delta_hp)
    hp_relative_error = float(hp_abs_error / abs(exact_delta_hp))

    forward_error_exact_delta = abs(exact_delta_f64 - float(exact_delta_hp))
    forward_error_direct_delta = abs(direct_delta_f64 - float(direct_delta_hp))

    cond_q = float(np.linalg.cond(q_f64))
    term_grad_dot_step = abs(float(grad_f64 @ step))
    term_half_step_q_step = abs(0.5 * float(step @ (q_f64 @ step)))
    cancellation_ratio = (term_grad_dot_step + term_half_step_q_step) / max(abs(exact_delta_f64), 1e-300)

    return {
        "float64": {
            "exact_delta": exact_delta_f64,
            "direct_delta": direct_delta_f64,
            "absolute_error": f64_abs_error,
            "relative_error": f64_relative_error,
        },
        "longdouble": {
            "exact_delta": float(exact_delta_hp),
            "direct_delta": float(direct_delta_hp),
            "absolute_error": float(hp_abs_error),
            "relative_error": hp_relative_error,
        },
        "forward_error_vs_float64": {
            "exact_delta": forward_error_exact_delta,
            "direct_delta": forward_error_direct_delta,
        },
        "conditioning": {
            "cond_q": cond_q,
            "cancellation_ratio": cancellation_ratio,
            "term_grad_dot_step": term_grad_dot_step,
            "term_half_step_q_step": term_half_step_q_step,
        },
        "pure_cancellation": hp_relative_error <= f64_relative_error / 10.0,
    }


def reconstruct_task_inputs(spec: CaseSpec, task_index: int):
    """Replay ``case.run_case``'s per-task setup to reconstruct one task's exact inputs.

    Duplicated from ``case.py`` rather than imported, since the RNG streams
    (``seeds.gradient``/``seeds.probe_direction``/``seeds.noise``) are
    stateful and shared across the per-task loop: reproducing task
    ``task_index``'s exact values requires replaying every earlier task's
    draws in the same order, not just re-deriving that one task in isolation.
    """

    seeds = case_seeds(spec.config_seed, spec.case_index)
    layout = BlockLayout(tuple(spec.block_width for _ in range(spec.num_blocks)))
    incidence = build_incidence(spec.family, spec.num_tasks, spec.num_blocks, seeds.graph_structure)
    noise_model = NoiseModel(kind=spec.noise_kind, sigma=spec.noise_sigma, rho=spec.noise_rho)  # type: ignore[arg-type]
    tasks, _ = gen.generate_tasks(
        layout,
        incidence,
        seeds.curvature,
        private_dims=spec.private_dims,
        condition_number=spec.condition_number,
        coupling_rank=spec.coupling_rank,
        mu=spec.mu,
        gradient_cosine_target=spec.gradient_cosine_target,
        gradient_scale_target=spec.gradient_scale_target,
    )

    for i, task in enumerate(tasks):
        p_i = task.shared_dim
        x_i = seeds.gradient.standard_normal(p_i)
        phi_0 = seeds.gradient.standard_normal(task.private_dim)
        seeds.probe_direction.standard_normal(p_i)  # probe direction, unused by this recheck
        step = 0.01 * seeds.probe_direction.standard_normal(p_i)

        if spec.optimizer == "sgd":
            eigs_c = np.linalg.eigvalsh(task.private_curvature)
            eta, _ = st.sgd_eta_for_regime(eigs_c, spec.stability_regime, seeds.noise)
            noise = noise_model.sample(seeds.noise, spec.horizon, task.private_dim)
            beta, v_0 = None, None
        else:
            beta = spec.beta if spec.beta is not None else 0.9
            eta, _ = st.momentum_eta_for_regime(task, beta, spec.stability_regime, seeds.noise)
            noise = noise_model.sample(seeds.noise, spec.horizon, task.private_dim)
            v_0 = seeds.gradient.standard_normal(task.private_dim)

        if i == task_index:
            return task, x_i, phi_0, v_0, eta, beta, noise, step

    raise IndexError(f"task_index {task_index} out of range for case {spec.case_index}")


def recheck_known_failures(config: dict) -> dict[int, dict]:
    """Run :func:`recheck_momentum_loss_change` for every entry in :data:`KNOWN_FAILURES`."""

    cases = {spec.case_index: spec for spec in enumerate_cases(config)}
    report = {}
    for case_index, task_index in KNOWN_FAILURES:
        spec = cases[case_index]
        task, x_i, phi_0, v_0, eta, beta, noise, step = reconstruct_task_inputs(spec, task_index)
        report[case_index] = {
            "task_index": task_index,
            "family": spec.family,
            "optimizer": spec.optimizer,
            "horizon": spec.horizon,
            "stability_regime": spec.stability_regime,
            **recheck_momentum_loss_change(task, x_i, phi_0, v_0, eta, beta, noise, step),
        }
    return report
