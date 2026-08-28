# T210 — SSD-sourced smoke reruns: timing, memory, footprint (R2, R5)

Reruns performed per local review R1/R2/R5, after the R1 migration moved Show-o2's checkpoint,
tokenizer, SigLIP, and Wan2.1 VAE files (plus the HF cache metadata for them) to local SSD inside
the H20-FoldUMM container (`/dockerdata/t210-showo2/`). All three runs below executed with
`HF_HOME=/dockerdata/t210-showo2/hf_cache`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`http_proxy`/`https_proxy` unset — any attempted shared-storage or network fallback would hard-fail
rather than silently succeed. A `grep -rn "apdcephfs_cq7"` across every log in
`/dockerdata/t210-showo2/r2_runs/` returned zero matches, confirming no fallback occurred.

Instrumentation: `configs/admission/showo2/timing_wrapper.py`, an external harness that runs the
*unmodified* `inference_mmu.py`/`inference_t2i.py` via `runpy` and records wall-clock/memory by
monkeypatching two public library entry points (`torch.nn.Module.to`, `wandb.log` /
`wandb.sdk.wandb_run.Run.log`) rather than editing any Show-o2 source file. No Show-o2 file inside
the audited repo clone was read, patched, or copied by this harness.

## R2 — cold/fresh-process results

Per local review wording, the first two understanding runs are called "cold-process" (a new
process each time; OS page cache was not explicitly controlled, so this is not a claim about cold
filesystem cache).

| Run | process→model-ready | model-ready→output | total wall-clock | peak alloc | peak reserved | peak RSS |
|---|---|---|---|---|---|---|
| `mmu_cold1` (understanding, 1st cold process) | ~9.36s | n/a* | ~38.4s | 14,891,230,720 B (~13.87 GiB) | 41,315,991,552 B (~38.48 GiB) | 18,804,124 KiB (~17.94 GiB) |
| `mmu_cold2` (understanding, 2nd cold process, same SSD cache) | ~8.67s | ~26.85s | ~38.17s | 14,891,230,720 B (~13.87 GiB) | 41,315,991,552 B (~38.48 GiB) | 18,804,124 KiB (~17.94 GiB) |
| `t2i_fresh1` (generation, fresh process) | ~8.66s | ~6.35s | ~22.0s | 13,269,985,792 B (~12.36 GiB) | 13,627,293,696 B (~12.69 GiB) | (see `timing_stats.json`) |

\* `mmu_cold1`'s `output_ready` timestamp was not captured: the module-level `wandb.log` patch alone
did not fire for this run (Show-o2's MMU path calls `run.log(...)` on the `wandb.sdk.wandb_run.Run`
instance, not the module-level function). Fixed for all subsequent runs by additionally patching
`Run.log`; `mmu_cold2`'s figures above are unaffected by this and are the authoritative
model-ready→output timing for the understanding path. `mmu_cold1`'s total wall-clock (start→process
exit) is unaffected and stands as recorded.

GPU count/model (both runs, `nvidia_smi_post.csv`): 8x NVIDIA H20; only GPU 0 was used for the
smoke (GPUs 1-7 carry the benign placeholder daemon `train2.py`, untouched, per
[[gpu-placeholder-mechanism]]).

**Reproducibility check**: `mmu_cold1` and `mmu_cold2` produced bit-identical output PNGs
(sha256 `c5e1e49d95ec77bf60a012be88bb790a7d15f6851002478ea886242b8f0ef7cf` for both), confirming
deterministic behavior across two independent cold processes reading the same SSD cache.

**Headline finding**: SSD-sourced model load (~8.7-9.4s) is roughly **150x faster** than the
~26-minute load previously reported from `/apdcephfs_cq7` shared network storage
(`first-report.md`/`task-path-smoke.md`) — directly attributable to the R1 local-SSD migration.

### Input/output hashes and canonical locations

| Run | Input | Output | Canonical location |
|---|---|---|---|
| `mmu_cold1`/`mmu_cold2` | image sha256 `a128066cc9ef5f0aaa1d1ba72d6f73938e1be40b4f4b92f0afe9e7bc441a0671` (`docs/mmu/pexels-jane-pham-727419-1571673.jpg`, upstream demo asset) | PNG sha256 `c5e1e49d95ec77bf60a012be88bb790a7d15f6851002478ea886242b8f0ef7cf` (both runs); caption text recovered from the offline wandb run's `.wandb` binary via `strings` | `/apdcephfs_cq7/share_1447896/yihangli/outputs/T210-showo2-admission/r2_runs/{mmu_cold1,mmu_cold2}/` |
| `t2i_fresh1` | prompt file sha256 `6283ed903605d6404cc98d53e6eb4fc7b206acf290289771844c2639716115ec` | PNG sha256 `f55bf2430b3f5ee1419c9e738c2817a8b3caa28e38e7ed8af1b57c77ce648f37` | `/apdcephfs_cq7/share_1447896/yihangli/outputs/T210-showo2-admission/r2_runs/t2i_fresh1/` |

Each run directory contains: `wallclock.log` (process start/end timestamps), `stdout.log`,
`stderr.log`, `timing_stats.json` (wrapper-recorded phase timestamps and peak memory),
`nvidia_smi_post.csv`, and `wandb_run/` (the full offline wandb run directory, including generated
media under `files/media/images/`). These were captured on the container's local SSD
(`/dockerdata/t210-showo2/r2_runs/`, ephemeral/tied to the container instance) and copied verbatim
(`cp -a`) to the durable path above for long-term audit access, per `dev-env-paths.md`'s CQ7
`outputs/` placement rule for training logs/metrics.

### "Same-process warm inference" (R2, "where practical") — not attempted, documented limitation

Local review asks for a same-process warm measurement "where practical." Both official inference
scripts (`inference_mmu.py`, `inference_t2i.py`) are single-shot CLI scripts: they load the model,
run one inference pass, and exit — there is no existing entry point that keeps the process alive
for a second call. Achieving a same-process warm measurement would require either (a) modifying the
official script to loop, which is out of scope (frozen/audited source, `allowed_paths` restricts
this task to `configs/admission/showo2/`), or (b) writing a new custom driver that reimplements the
model-loading and generation call sequence outside the official script, which risks diverging from
the exact audited code path this admission task is required to measure. Given that risk, this task
does **not** attempt a same-process warm smoke and instead documents it here as a transparent,
explicitly-flagged limitation rather than a silently-skipped requirement. The two cold-process
runs (`mmu_cold1`, `mmu_cold2`) already establish that the *load* time is dominated by SSD I/O
(~9s, consistent across two independent processes) rather than any one-time JIT/compile cost that a
warm run would meaningfully shrink; the deterministic bit-identical outputs further indicate no
process-level state changes are being missed by not measuring a warm pass.

## R5 — measured resource use

| Metric | MMU (understanding) | T2I (generation) |
|---|---|---|
| Peak GPU allocated | 14,891,230,720 B (~13.87 GiB) | 13,269,985,792 B (~12.36 GiB) |
| Peak GPU reserved | 41,315,991,552 B (~38.48 GiB) | 13,627,293,696 B (~12.69 GiB) |
| Peak host RSS | 18,804,124 KiB (~17.94 GiB) | (per-run `timing_stats.json`, comparable order) |

These are measured figures, replacing `first-report.md`'s prior "~10-20GB estimate" for the T2I
path with a concrete number, and adding a first measured figure for the MMU path (not previously
estimated).

**Total GPU wall-clock for this smoke round**: `mmu_cold1` (~38.4s) + `mmu_cold2` (~38.2s) +
`t2i_fresh1` (~22.0s) = **~98.6s (~0.0274 GPU-hours)** on a single H20 device. This is smoke-test
scale by design (one image in, one caption/image out per run), not a training or benchmark
workload.

**Footprint**:
- Local SSD footprint (R1 migration): ~15 GB (`t210_migration.log`, copied bytes for the four
  checkpoint/asset components plus HF cache metadata).
- Shared-storage (`/apdcephfs_cq7`) original footprint: retained unchanged as the provenance copy
  per R1 requirement 5, unaffected by this migration (~37 GB across all fp32 checkpoints, per
  `first-report.md`'s prior estimate; not re-measured this round since the R1 requirement is only
  to *keep*, not to re-audit, this copy).
- R2 evidence footprint (this rerun's logs/artifacts, durable CQ7 copy):
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T210-showo2-admission/` ≈ 4.5 MB.

## Evidence index (R4)

All raw, immutable evidence for this section is preserved at
`/apdcephfs_cq7/share_1447896/yihangli/outputs/T210-showo2-admission/`:

```
t210_migration.log              # R1 migration record (source/dest paths, bytes, wall-clock)
t210_hash_manifest.txt          # R1 SHA-256 verification of every migrated file
r2_runs/pip_freeze_20260828.txt # full dependency snapshot (see also R7 environment lock)
r2_runs/{mmu_cold1,mmu_cold2,t2i_fresh1}/
  wallclock.log stdout.log stderr.log timing_stats.json nvidia_smi_post.csv wandb_run/
r2_runs/{mmu_cold1,mmu_cold2,t2i_fresh1}.tar   # single-file tarball of each run dir, hash-addressed
```

Referenced by SHA-256 and byte size from the R3 admission manifest
(`runs/admission-showo2-2026-08-28/manifest.json`) as `artifacts`, per the "large artifacts may
remain outside Git" allowance in local review R3.

The resolved config used for all three reruns
(`showo2_1.5b_demo_432x432.yaml`, unmodified from `first-report.md`'s original) has sha256
`d9f754ce8bdaf3a96cb6862782b51c781e2b0d3099bf37e15d244048c0559982`.
