## T120 KKT Reference Comparison Suite — Run 2026-09-28

### Overview

First formal run of the T120 independent-reference comparison suite.
All 111 cases passed. Elapsed: 2.15 s (Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1).

### Case breakdown

- private_response_schur: 45 (5 fixed + 40 random)
- trust_region: 44 (4 fixed + 40 random)
- negotiation: 22 (2 fixed + 20 random)

### Max observed residuals vs. thresholds

| Check | Max observed | Threshold | Margin |
|---|---|---|---|
| private_response_param | 1.11e-15 | 1e-08 | > 7 orders |
| schur_param | 1.78e-15 | 1e-08 | > 7 orders |
| exact_rational_param | 8.88e-16 | 1e-08 | > 7 orders |
| trust_region_eigen_param | 1.78e-15 | 1e-06 | > 6 orders |
| trust_region_eigen_objective | 4.44e-15 | 1e-08 | > 6 orders |
| trust_region_eigen_kkt_stationarity | 1.47e-14 | 1e-06 | > 8 orders |
| trust_region_eigen_kkt_complementarity | 7.37e-15 | 1e-06 | > 8 orders |
| trust_region_blackbox_param | 4.27e-06 | 5e-03 | > 3 orders |
| trust_region_blackbox_objective | 6.40e-06 | 1e-03 | > 2 orders |
| trust_region_blackbox_kkt_stationarity | 1.53e-08 | 1e-04 | > 4 orders |
| trust_region_blackbox_kkt_complementarity | 1.28e-05 | 5e-04 | > 1 order |
| negotiation_param | 9.90e-06 | 5e-03 | > 2 orders |
| negotiation_tau | 1.28e-05 | 5e-03 | > 2 orders |
| negotiation_retained_gain | 2.00e-05 | 5e-03 | > 2 orders |
| negotiation_kkt_stationarity | 1.03e-12 | 1e-04 | > 8 orders |
| negotiation_kkt_complementarity | 2.56e-05 | 1e-04 | ~ 4x |

All thresholds met with substantial margin. No failures.

### Key bugs found and fixed during development (not visible in this run)

1. Late-binding closure in `independent_negotiation_solve`: per-task constraint
   functions `task_jac` and `task_hess` were capturing `grad_i`/`hess_i` via loop
   closure rather than as default arguments, so all tasks shared the last task's
   gradient/hessian. Fixed by explicit default args (`g_i=grad_i`, `h_i=hess_i`).

2. Non-PSD Schur complement in random test generator: sampling `h_xx` and
   `h_phiphi` as independently-PD blocks does not guarantee PSD Schur complement.
   Fixed by sampling a single joint PD block matrix over [x; phi] and splitting
   into blocks (principal submatrices of a PD matrix are PD; Schur complement of
   a PD matrix in any square subblock is PD).
