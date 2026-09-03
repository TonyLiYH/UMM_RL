# `feasibility-showo2-v1` — run notes

Real-checkpoint diagnostic run for T215, executed inside the H20-FoldUMM GPU container
(`/root/venvs/showo2`, `torch==2.5.1+cu124`), against the official Show-o2 checkout pinned at
`217d183b30995db4ac82158259f45800e57e2eb1`, using the local-SSD checkpoint/tokenizer/VAE cache at
`/dockerdata/t210-showo2` (`configs/feasibility/showo2/storage-preflight.json`: `status: pass`,
`filesystem_class: local`).

Invocation (executed from the Show-o2 repo root, per the demo config's relative asset paths):

```bash
cd /apdcephfs_cq9/share_1447896/yihangli/workspace/showo2_admission/Show-o/show-o2
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/apdcephfs_cq9/share_1447896/yihangli/workspace/.worktrees/T215/src \
  /root/venvs/showo2/bin/python -m comppareto.adapters.showo2.run_feasibility \
  --config /apdcephfs_cq9/share_1447896/yihangli/workspace/.worktrees/T215/configs/feasibility/showo2/diagnostic-v1.yaml \
  --output-dir /apdcephfs_cq9/share_1447896/yihangli/workspace/.worktrees/T215/runs/feasibility-showo2-v1
```

## Run history (this session)

Four attempts were made; the first three failed on infrastructure defects (fixed, one commit
each, verified against the full local test suite before each retry); the fourth completed without
a crash and is this run's final, reported `metrics.json`/`manifest.json`.

1. **v1** (pre-fix): `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward
   is not implemented`. Root cause: default SDPA backend lacks double-backward support needed for
   `create_graph=True` in the rerun-response protocol. Fixed in `model_io.py`
   (`sdpa_kernel(SDPBackend.MATH)`), commit `cd18c3b`.
2. **v2** (launch mistake, not a code defect): `FileNotFoundError: Wan2.1_VAE.pth` — wrong working
   directory (T215 worktree instead of the Show-o2 repo root, whose relative asset paths require
   it). No commit; corrected invocation only.
3. **v3** (post-fix-1): `RuntimeError: Expected all tensors to be on the same device, but found at
   least two devices, cuda:0 and cpu!`. Root cause: `finite_diff.py`'s `rademacher_direction()`
   always built its `+-1` direction tensor on the CPU (default `torch.Generator()` device), never
   moved to `theta_s`'s device. Fixed (`device` parameter threaded through, `build_directions()`
   passes `theta_s.device`), commit `22892b1`.
4. **v4 = this run**: completed end-to-end without a Python exception (`failure_reason: null`
   in `metrics.json`), `manifest.json:status == "fail"` — the diagnostic itself, not
   infrastructure, produced a K=1 gate failure for both task paths. See
   `reports/T215/result-summary.md` and `reports/T215/failure-ledger.md` for the full numeric
   breakdown and root-cause analysis (confirmed via an isolated synthetic repro run in the same
   container/torch build, not by hypothesis).

All four attempts used exactly 1 GPU (`cuda:0`) and stayed far under the 8-GPU-hour cap:
v1 = 0.00735 h, v2 = 0.00129 h, v3 = 0.00768 h, v4 = 0.01601 h — cumulative ≈ 0.0323 h.

## What this run's `metrics.json`/`manifest.json` report

- `status: "fail"` — `k1_gate_passed` is `False` for both `mmu` and `t2i` task paths (K=1 is
  mandatory-first per the task's protocol; since it failed, **K=3 was not run** for either path —
  `variants.disjoint_k3` is `null` in both, by `run_feasibility.py`'s own gating logic, matching
  the task's explicit "if K=1 fails, do NOT proceed to K=3" instruction).
- `snapshot_restore.failed: 4` — every one of the 4 variant-runs (`mmu.disjoint_k1`,
  `mmu.same_batch_k1`, `t2i.disjoint_k1`, `t2i.same_batch_k1`) reports
  `rollback_ok_after_rerun_variants: False` even though every individual parameter's
  `rollback_detail[*].all_matches` is `True` (`data_max_abs_diff: 0.0`). This is because
  `verify_rollback()`'s overall boolean is `params_ok AND rng_ok AND data_order_ok`, and the RNG
  sub-check (`compare_rng_snapshot`, exact `torch.equal`) fails: `_run_split()` calls
  `torch.manual_seed(1215)` before each of `compute_raw`/`compute_commit_response`/
  `compute_rerun_response`(x2), and these protocols consume differing amounts of the RNG stream
  internally (through the model's own dropout/sampling ops inside `forward_und_only`/`forward`),
  so the RNG stream's state at the point `verify_rollback()` is called no longer matches the
  snapshot taken before the first `manual_seed(1215)` call of that split. **Every actual
  parameter/gradient tensor DOES roll back exactly** (`data_matches`/`grad_matches` all `True`);
  only the RNG-stream-position component of the combined rollback check fails. No persistent
  parameter mutation occurred (`persistent_updates: 0`, verified by construction per
  `model_io.py`'s `_reparametrize_module`-based substitution, and confirmed observationally by
  every `data_max_abs_diff: 0.0`). `rollback_ok_after_fd` is `True` for all 4 variants (the
  restore performed immediately before the finite-difference check does re-synchronize state
  correctly at that later point).
- `finite_difference.failed: 4` — all 4 variants fail the FD gate. For MMU, the analytic
  (autograd) gradient itself is `NaN` (`rerun_param_only_grad_norm`/`rerun_complete_grad_norm`:
  `NaN`); for T2I, the analytic gradient is finite but does not match the FD reference within
  tolerance on 3 of 4 directions per variant (see `reports/T215/failure-ledger.md` for exact
  numbers and root-cause explanation).
- `protocols.{raw,commit,rerun}_measured: true` for both task paths — all three protocols were
  successfully computed and are present in `task_results`, independent of whether their FD/rollback
  gates passed.
- `resources.gpu_hours: 0.01601` (this run alone), `gpu_count_used: 1`.
- `persistent_updates: 0`.

## Honesty note

This run's `status` is `"fail"`. Per the task's explicit instruction ("if K=1 fails, do NOT
proceed to K=3 — document the K=1 failure honestly instead"), no K=3 run was attempted for either
task path, and no further "infrastructure retry" was applied beyond the two genuine infrastructure
bugs already fixed (commits `cd18c3b`, `22892b1`) — the v4 result is reported as-is.
