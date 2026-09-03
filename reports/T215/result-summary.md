# T215 — Result summary

Run: `runs/feasibility-showo2-v1/` (`manifest.json:status = "fail"`, `metrics.json`, `notes.md`).
Executed against the real Show-o2 checkpoint (`217d183b30995db4ac82158259f45800e57e2eb1`) on 1
H20 GPU inside the accepted admission environment. No K=3 run was attempted for either task path
(K=1 gate failed on both — see below and `reports/T215/failure-ledger.md`).

## Headline result

**The K=1 gate did not pass for either task path (MMU, T2I).** Per the task's explicit rule ("if
K=1 fails, do NOT proceed to K=3 — document the K=1 failure honestly instead"), K=3 was correctly
skipped for both task paths.

## Numeric K=1 results

### MMU (understanding), `fusion_proj` (shared) + `und_trans.layers[0]` (private)

| Variant | raw grad norm | commit grad norm | rerun (param-only) grad norm | rerun (complete) grad norm | FD gate passed |
|---|---|---|---|---|---|
| `disjoint_k1` | 1.45392 | 1.40478 | NaN | NaN | **False** |
| `same_batch_k1` | 1.45392 | 1.27992 | NaN | NaN | **False** |

The rerun-response (finite-unroll, `create_graph=True`) analytic gradient w.r.t. `theta_s` is
`NaN` for both variants, in both the parameter-only and complete optimizer-state differentiation
modes. All 4 finite-difference directions (Rademacher seeds 42/43/44 + natural/gradient-aligned)
therefore report `analytic_value: NaN`, `error: NaN`, `passed: false` in each variant. The
finite-difference *reference* value itself (`fd_value`, computed via the separate
non-differentiable `rerun_loss_only` path) is finite and numerically unremarkable
(e.g. `disjoint_k1`/`natural_raw_grad`: `fd_value = 1.36163`) — only the differentiable/analytic
side is `NaN`.

### T2I (generation), `fusion_proj` (shared) + `diffusion_head_a[0]` (private)

| Variant | raw grad norm | commit grad norm | rerun (param-only) grad norm | rerun (complete) grad norm | FD gate passed |
|---|---|---|---|---|---|
| `disjoint_k1` | 0.16750 | 0.16233 | 0.15057 | 0.15057 | **False** |
| `same_batch_k1` | 0.16750 | 0.15007 | 0.14938 | 0.14938 | **False** |

Unlike MMU, no `NaN` occurs — the analytic rerun-response gradient is finite in both
differentiation modes. The FD gate still fails: 3 of 4 directions exceed tolerance per variant.

`disjoint_k1` FD detail:

| direction | fd_value | analytic_value | error | mode | tol | passed |
|---|---|---|---|---|---|---|
| `rademacher_seed_42` | -6.435e-05 | 5.354e-05 | 1.832 | relative | 1e-3 | False |
| `rademacher_seed_43` | 0.0 | -1.199e-05 | 1.199e-05 | absolute | 1e-6 | False |
| `rademacher_seed_44` | 1.287e-04 | 7.310e-05 | 0.432 | relative | 1e-3 | False |
| `natural_raw_grad` | 0.14717 | 0.14563 | 0.01046 | relative | 1e-3 | False |

`same_batch_k1` FD detail:

| direction | fd_value | analytic_value | error | mode | tol | passed |
|---|---|---|---|---|---|---|
| `rademacher_seed_42` | 0.0 | 4.069e-05 | 4.069e-05 | absolute | 1e-6 | False |
| `rademacher_seed_43` | -6.435e-05 | -5.692e-06 | 0.9115 | relative | 1e-3 | False |
| `rademacher_seed_44` | 6.435e-05 | 1.003e-04 | 0.5579 | relative | 1e-3 | False |
| `natural_raw_grad` | 0.14453 | 0.14542 | 0.00618 | relative | 1e-3 | False |

The natural (gradient-aligned) direction is closest to passing in both variants (0.6-1.0% relative
error, an order of magnitude over the 0.1% tolerance) but still fails the gate as declared; the 3
Rademacher directions are off by 43%-183% relative error (or fail the absolute-error branch
outright when the FD reference itself rounds to exactly 0 at the declared `eps`).

## Rollback (reversibility) result

Every actual parameter/gradient tensor in the declared 3-tensor subspace restores exactly
(`rollback_detail[*].all_matches: true`, `data_max_abs_diff: 0.0`, in all 4 variants x both task
paths). `snapshot_restore.failed = 4` in `metrics.json` reflects a narrower issue:
`verify_rollback()`'s combined boolean also requires the RNG stream to restore to an *exact*
byte-for-byte match, and each `_run_split()` call re-seeds `torch.manual_seed(1215)` multiple
times, with the model's own internal randomness (dropout/sampling) consuming a different amount of
the RNG stream on each of the raw/commit/rerun-param-only/rerun-complete calls — so the RNG
position at the point `verify_rollback()` is called differs from the RNG position at the original
snapshot, even though the observed tensor VALUES roll back exactly. **No persistent parameter
mutation occurred in any run** (`persistent_updates: 0` in every `metrics.json`, confirmed both by
construction — `model_io.py`'s `call_with_overrides` swaps parameters via
`_reparametrize_module` and always restores the model's own stored tensors on context exit — and
observationally, via the `0.0` max-abs-diff on every parameter). `rollback_ok_after_fd` is `True`
in all 4 variants (the restore step immediately preceding the finite-difference check does
re-synchronize state correctly by that point in the sequence).

## Resource accounting

- GPU-hours this run: 0.01601 h (wall clock 57.6 s), well under the 8-h cap.
- Cumulative GPU-hours across all 4 attempts this session (3 failed on infra bugs, 1 completed):
  v1 = 0.00735 h, v2 = 0.00129 h, v3 = 0.00768 h, v4 = 0.01601 h → **≈ 0.0323 h total**, i.e. 0.4%
  of the 8-h cap.
- GPU count: 1 (`cuda:0`) in every run, under the 2-GPU cap.
- Peak allocated/reserved CUDA memory (v4): 44.43 GB / 47.54 GB.

## Claim status

This result does **not** support the research claim as stated ("Show-o2 can support a reversible,
optimizer-state-aware finite-response diagnostic on a selected shared/private parameter subspace
before formal D0 experiments are authorized"), for the two declared task paths and the K=1
protocol as specified. See `reports/T215/claim-check.md` for the itemized claim-by-claim
breakdown and `reports/T215/failure-ledger.md` for the confirmed root causes of the MMU `NaN` and
the T2I tolerance miss.
