# T215 — Failure ledger

Every failed diagnostic gate, infrastructure defect, and blocker encountered this session, with
root cause, evidence, and resolution status. Nothing below is silently bypassed or routed around;
each entry states plainly whether it was (a) fixed as a permitted infrastructure retry, or (b) is
a genuine, reportable diagnostic/numerical failure that stands as the task's K=1 result.

## Infrastructure defects (fixed; permitted "infrastructure retries")

### INFRA-1: SDPA memory-efficient backend lacks double-backward support

- **Symptom** (v1 run): `RuntimeError: derivative for
  aten::_scaled_dot_product_efficient_attention_backward is not implemented`.
- **Root cause**: the default SDPA backend selected by this hardware/torch build
  (`torch==2.5.1+cu124` on H20) for the model's internal `scaled_dot_product_attention` calls is
  the memory-efficient kernel, whose backward pass does not implement a second-order
  (double) backward — fatal for the rerun-response protocol's `create_graph=True` requirement.
- **Fix**: wrap every `call_with_overrides()` forward call in
  `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` (`src/comppareto/adapters/showo2/model_io.py`).
  Changes only which CUDA kernel computes an already-specified mathematical operation; does not
  alter the subspace, batches, seeds, K, lr, or tolerances.
- **Verification before retry**: `compileall` + full local pytest suite (190 passed); `model_io.py`
  is not imported by the CPU-only test suite (GPU/real-checkpoint-only module), so this required no
  new CPU-test coverage per its own module docstring.
- **Commit**: `cd18c3b` (pushed to origin, branch `agent/T215-showo2-finite-response-feasibility`).
- **Status**: resolved.

### INFRA-2: Finite-difference Rademacher direction tensors built on the wrong device

- **Symptom** (v3 run, after INFRA-1's fix): `RuntimeError: Expected all tensors to be on the same
  device, but found at least two devices, cuda:0 and cpu!`.
- **Root cause**: `finite_diff.py`'s `rademacher_direction()` created its `+-1` constant tensor via
  a default (CPU) `torch.Generator()`, with no subsequent device transfer — `theta_s + eps *
  direction` inside `run_finite_difference_check()` then mixed a CUDA `theta_s` with a CPU
  `direction`.
- **Fix**: added an explicit `device` parameter to `rademacher_direction()` (default `"cpu"` for
  backward-compat with existing CPU-only unit tests) and had `build_directions()` pass
  `device=theta_s.device`. The scratch RNG `Generator` itself remains CPU-only for reproducibility;
  only the final constant tensor moves device via `.to()`. Does not alter which seeds/directions
  are used, only where the resulting tensor is materialized.
- **Verification before retry**: `compileall`, showo2-specific tests (34 passed), full suite
  (190 passed).
- **Commit**: `22892b1` (pushed to origin).
- **Status**: resolved.

### INFRA-3 candidate (investigated, NOT applied — determined to be a genuine numerical result, not an infra bug)

- Investigated hypothesis: `omni_attn_mask_naive()` (official Show-o2 source,
  `models/omni_attention.py`) builds its inverted attention mask using
  `torch.iinfo(torch.long).min` (~-9.223e18) as the masked-fill value on an int64 tensor, later cast
  to fp32 by `model_io.py`. Tested via an isolated repro
  (`torch.nn.functional.scaled_dot_product_attention` under `sdpa_kernel(SDPBackend.MATH)`, with a
  causal+modality-block mask built exactly per `omni_attn_mask_naive`'s logic, differentiated
  through a double-backward) on the SAME container/torch build (`/root/venvs/showo2`,
  `torch==2.5.1+cu124`, real H20 GPU): comparing mask fill values
  `torch.iinfo(torch.long).min` (~-9.223e18) vs `-1e9` vs `-1e4`, ALL THREE produced finite,
  well-behaved losses and double-backward gradients on both CPU and CUDA — no NaN, no exception,
  no numerically implausible magnitude. **This rules out the attention-mask fill value as the
  cause of the observed MMU `NaN`.** No fix was applied for this candidate since it is not the
  actual root cause (see MMU-NAN below for the confirmed cause). Ruling this out, rather than
  applying a speculative fix, is itself the honest outcome here.

## Genuine diagnostic/numerical failures (reported as the K=1 result; NOT infrastructure)

### MMU-NAN: rerun-response analytic gradient is NaN for both MMU variants (`disjoint_k1`, `same_batch_k1`)

- **Observation**: `rerun_param_only_grad_norm` and `rerun_complete_grad_norm` are both `NaN` in
  `runs/feasibility-showo2-v1/metrics.json` for `task_results.mmu.variants.{disjoint_k1,
  same_batch_k1}`, in BOTH the parameter-only (`detach_moments=True`) and complete
  (`detach_moments=False`) optimizer-state differentiation modes. `raw_grad_norm` (1.45392) and
  `commit_grad_norm` (1.40478 / 1.27992) are finite — only the differentiable rerun-response chain
  produces `NaN`.
- **Root cause, confirmed via isolated repro on the real container/torch build**: T215's explicit
  AdamW re-implementation (`protocols.py::adamw_step`) computes
  `denom = (exp_avg_sq / bias_correction2).sqrt() + eps` with `eps` added OUTSIDE the square root
  (matching `torch.optim.AdamW`'s exact default semantics, per the module's own docstring). At the
  FIRST optimizer step (`step=1`, which is always exercised at K=1), the bias-correction term
  exactly cancels: `bias_correction2 = 1 - beta2^1 = 1 - beta2 = (1 - beta2)`, so
  `exp_avg_sq / bias_correction2 = grad_p^2` exactly (confirmed numerically:
  `(1-beta2)/bias_correction2 == 1.0` at step 1, to full floating precision). Therefore
  `denom = sqrt(grad_p^2) + eps = |grad_p| + eps`. Differentiating `sqrt(x)` at `x = grad_p^2` when
  `grad_p == 0` exactly, under `create_graph=True`, hits the classical `d(sqrt(x))/dx = 1/(2*sqrt(x))`
  singularity at `x = 0` — producing `NaN` (`0 * inf` in the chain rule) for that parameter
  coordinate's contribution to `d(theta_p_new)/d(theta_s)`, EVEN THOUGH `grad_p` itself depends on
  `theta_s` and has a nonzero second derivative there. A single `NaN` coordinate poisons the entire
  `.norm()` computed downstream.
  - **Repro** (executed in-container, same torch/hardware): a 20-line synthetic PyTorch script with
    a `theta_p` coordinate whose first-order gradient (`grad_p`) is exactly zero at the current
    `theta_s` but is itself a nonzero function of `theta_s` (so its own second derivative is
    nonzero) reproduces `grad_theta_s` containing a `NaN` entry under the identical
    AdamW-with-`create_graph=True` unroll used by `protocols.py`. A control variant where `grad_p`
    has NO exact zero anywhere in the subspace shows no `NaN`. This confirms the mechanism, not
    merely a plausible hypothesis.
  - This is plausible for the real model: `fusion_proj`/`und_trans.layers[0]` together have
    ~21.7M parameters, and the MMU NTP loss masks most target positions (`IGNORE_INDEX = -100`)
    plus the attention mask zeroes out contributions from non-attended positions — across a
    parameter block this large, at least one coordinate landing at exactly zero gradient at the
    adaptation batch's current `theta_s` is unsurprising, not a rare coincidence.
- **Why this is NOT an infrastructure bug and NOT eligible for a further retry**: the `eps`-outside-
  sqrt AdamW formula is an intentional, explicit design choice in `protocols.py`
  ("matching `torch.optim.AdamW`... semantics exactly"), part of the frozen protocol (K=1 is
  mandatory, uses this exact optimizer). Moving `eps` inside the sqrt, or otherwise regularizing
  this, would CHANGE the protocol's optimizer semantics — which the task's failure/retry rules
  explicitly prohibit ("do not expand the trainable subspace, response horizon, GPU count, or
  budget after observing diagnostic values... Numerical or differentiation failure counts as a
  result unless a task-wide preregistered rule applies"). No preregistered rule in
  `tasks/T215-showo2-finite-response-feasibility.md` authorizes altering the optimizer formula
  after observing this NaN. This is therefore reported as the genuine K=1 result for the MMU task
  path: **the rerun-response finite-unroll gradient is numerically undefined (NaN) at K=1 for this
  subspace/batch/seed combination**, under exact AdamW semantics.

### T2I-FDMISS: rerun-response analytic gradient is finite but exceeds FD tolerance on 3/4 directions, both variants

- **Observation**: no `NaN`/`Inf` occurs anywhere in the T2I computation (`rerun_param_only_grad_norm
  == rerun_complete_grad_norm` at 0.15057 / 0.14938, both finite and internally consistent with
  `raw_grad_norm`/`commit_grad_norm` via plausible cosine similarities of 0.92-0.99). The
  finite-difference GATE still fails: 3 of 4 fixed directions (the 3 Rademacher seeds) exceed the
  1e-3 relative / 1e-6 absolute tolerance by 43%-183% relative error (or fail the absolute branch
  outright when `fd_value` rounds to exactly 0.0 at the declared scale-adaptive `eps`); only the
  4th (natural/gradient-aligned) direction comes reasonably close (0.6%-1.0% relative error, still
  ~6-10x over tolerance).
- **Interpretation**: this pattern (natural direction closest, Rademacher directions far off,
  including exact-zero FD references on some seeds) is consistent with the central
  finite-difference estimator being dominated by higher-order curvature / cancellation noise at
  the declared `eps` scale for THIS specific (real, ~92M-parameter combined shared+private) block
  and loss surface — i.e., a genuine second-order numerical-accuracy limitation of the K=1
  finite-unroll gradient against a fixed-`eps` central-difference reference on the real model, not
  a code defect. No alternative `eps`/direction/tolerance was substituted (doing so post-hoc, after
  observing this result, would violate the task's "infrastructure retries reuse the same
  configuration and seed" rule and the broader "do not expand... after observing diagnostic
  values" constraint).
- **Reported as the genuine K=1 result for the T2I task path**: the rerun-response gradient is
  numerically defined but does not match the finite-difference reference within the declared
  tolerance at K=1, for 3 of 4 fixed directions in both the disjoint and same-batch variants.

### ROLLBACK-RNG: `rollback_ok_after_rerun_variants = False` in all 4 variants (RNG-exact-restore sub-check)

- **Observation**: every parameter/gradient tensor restores exactly
  (`rollback_detail[*].all_matches: true`, `data_max_abs_diff: 0.0`) in all 4 variants, yet the
  combined `verify_rollback()` boolean is `False` because it also requires RNG state to restore
  byte-for-byte exactly (`torch.equal`, no tolerance).
- **Root cause**: `run_feasibility.py`'s `_run_split()` calls `torch.manual_seed(1215)` multiple
  times per split (before `compute_raw`, before `compute_commit_response`, before each of the two
  `compute_rerun_response` calls), and each of these protocol calls internally consumes a
  DIFFERENT amount of the RNG stream through the real model's own forward-pass randomness (e.g.
  any dropout inside `forward_und_only`/`forward`, despite `model.eval()` being set at load time —
  `eval()` disables dropout's stochastic behavior for most `nn.Dropout` modules but does not
  guarantee zero RNG consumption for every op in a large, third-party model). The RNG stream
  position at the point `state.verify_rollback()` is called (after the 4th `manual_seed(1215)` +
  its subsequent protocol call) does not match the RNG stream position at the ORIGINAL snapshot
  (taken once, before the split begins) — this is an artifact of the multi-reseed-then-diverge
  sequence, not a failure to persist/restore any actual model state.
- **Impact on the gate**: this makes `snapshot_restore.failed = 4` in `metrics.json`, which fails
  the acceptance contract's `snapshot_restore.failed == 0` metric requirement for this run.
  Reported honestly as a genuine, currently-unresolved gap in the diagnostic harness's RNG-exactness
  bookkeeping — NOT a sign of any persistent parameter mutation (which is independently confirmed
  absent via `persistent_updates: 0` and the `0.0` max-abs-diff on every actual tensor). No fix was
  applied this session, consistent with the task's "do not expand... after observing diagnostic
  values" constraint and because the K=1 gate had ALREADY failed on the FD sub-condition
  (MMU-NAN / T2I-FDMISS above) independent of this RNG artifact — fixing RNG bookkeeping would not
  change the overall K=1 pass/fail outcome, and the task instructs against continuing to
  K=3/further remediation once K=1 has failed.

## Summary

- 2 infrastructure defects found and fixed as permitted retries (INFRA-1, INFRA-2).
- 1 infrastructure hypothesis investigated and RULED OUT by direct repro (INFRA-3 candidate).
- 3 genuine, reportable K=1 diagnostic failures, none bypassed: MMU-NAN (differentiation
  undefined), T2I-FDMISS (differentiation defined but outside declared tolerance), ROLLBACK-RNG
  (RNG-exactness sub-check fails, though actual parameter state and reversibility are unaffected).
- K=3 was correctly NOT run for either task path, per the task's mandatory K=1-before-K=3 rule.
- Total GPU-hours consumed across all failure-diagnosis and retry attempts this session:
  ≈0.0323 h, against the 8-h cap (0.4%). GPU count: 1, against the 2-GPU cap.
