# T215 — Claim check

Research claim (from `tasks/T215-showo2-finite-response-feasibility.md`):

> Show-o2 can support a reversible, optimizer-state-aware finite-response diagnostic on a selected
> shared/private parameter subspace before formal D0 experiments are authorized.

Pass/fail gate (verbatim, from the same task file):

> At least one declared shared/private subspace must:
> - restore persistent floating state within the declared dtype tolerance and restore counters,
>   RNG, and data-order state exactly;
> - produce separate raw, commit-response, and rerun-response measurements;
> - match the directional finite-difference reference to relative error at most 1e-3 when the
>   reference magnitude exceeds 1e-8, or absolute error at most 1e-6 near zero;
> - record complete peak-memory, wall-clock, FLOPs or gradient-evaluation, and extra-data
>   accounting;
> - remain inside the two-GPU, eight-H20-equivalent-GPU-hour default envelope.

## Sub-claim table

| Sub-claim | MMU (`fusion_proj` + `und_trans.layers[0]`) | T2I (`fusion_proj` + `diffusion_head_a[0]`) |
|---|---|---|
| Parameter/gradient state restores within dtype tolerance | **Pass** — every parameter's `data_max_abs_diff = 0.0` in all variants | **Pass** — same |
| Counters/RNG/data-order restore *exactly* | **Fail** — RNG stream position does not restore exactly across the multi-reseed `_run_split()` sequence (see below); this is the sole reason `rollback_ok_after_rerun_variants = False` | **Fail** — same mechanism |
| Raw/commit/rerun measured separately | **Pass** — `protocols.raw_measured/commit_measured/rerun_measured = true` | **Pass** — same |
| FD match within tolerance | **Fail** — analytic gradient is `NaN` on all 4 directions x 2 variants | **Fail** — analytic gradient is finite but exceeds tolerance on 3/4 directions x 2 variants |
| Resource/memory/runtime accounting recorded | **Pass** — `resources.{gpu_hours, wall_clock_seconds, peak_memory_allocated_bytes, peak_memory_reserved_bytes, gpu_count_used}` all populated | **Pass** — same |
| Within 2-GPU / 8-GPU-hour envelope | **Pass** — 1 GPU used, 0.01601 h this run (≈0.0323 h cumulative across all 4 attempts) | **Pass** — same |

## Overall gate determination

The task's gate requires that **at least one** declared subspace pass **all** of the listed
conditions simultaneously. Neither MMU nor T2I passes the RNG-exact-restore sub-condition or the
finite-difference sub-condition. **The overall pass/fail gate is therefore not met by either task
path**, and `runs/feasibility-showo2-v1/manifest.json:status = "fail"` reflects this accurately.

This diagnostic run does not support the claim that Show-o2 can, as currently implemented in this
diagnostic harness and with the AdamW-at-K=1 protocol as specified, produce a rerun-response
finite-unroll gradient that (a) is numerically defined (MMU: it is not — see failure-ledger) and
(b) matches the finite-difference reference within the declared tolerance (T2I: it does not, by
43%-183% relative error on 3 of 4 fixed directions).

## What DID work (partial positive evidence, for completeness)

- The reversible snapshot/restore machinery correctly preserves every actual parameter and
  gradient tensor value across every transition, with `persistent_updates: 0` confirmed both by
  construction and observation — no persistent joint post-training occurred, satisfying the task's
  hard "no persistent parameter update" constraint regardless of the diagnostic's pass/fail
  outcome.
- All three protocols (raw, commit-response, rerun-response) were successfully computed and
  produced finite, plausible gradient norms and loss values for T2I, and finite raw/commit
  gradients (only the rerun-response analytic gradient is `NaN`) for MMU — i.e., the harness itself
  runs to completion on the real checkpoint and produces measurable quantities for 5 of the 6
  protocol computations per task path.
- Resource usage is far inside budget (≈0.4% of the 8-GPU-hour cap across all attempts this
  session, including the 3 failed infrastructure-debugging attempts).
- Two genuine infrastructure defects (SDPA backend double-backward support; FD direction device
  placement) were identified and fixed as permitted "infrastructure retries" without altering any
  frozen protocol quantity (subspace, batches, seeds, K, lr, tolerances) — this establishes that
  the harness CAN reach a fully-executing state on the real model, distinct from the diagnostic
  itself passing.

## Caveat on non-correspondence pseudo-targets

Per `model_io.py`'s module docstring and `configs/feasibility/showo2/diagnostic-v1.yaml`'s
documented design, neither task path has a real ground-truth target available from the accepted
asset manifest — MMU uses a frozen, seeded, no-grad-generated pseudo-caption as its NTP target;
T2I uses a disjoint real image's VAE-encoded, transport-sampled latent as its flow-matching
target. Both are documented data-construction choices for exercising the loss/gradient numerics,
not claims of caption quality or prompt-image correspondence. This does not affect the gate
determination above (the gate is about differentiation/restoration correctness, not target
quality), but is noted here for completeness since it affects how the raw/commit/rerun LOSS VALUES
themselves should be interpreted (not as meaningful task performance).
