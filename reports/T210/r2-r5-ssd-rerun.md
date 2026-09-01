# T210 — SSD-sourced smoke reruns: timing, memory, footprint (R2, R5)

Reruns performed per local review R1/R2/R5, after the R1 migration moved Show-o2's checkpoint,
tokenizer, SigLIP, and Wan2.1 VAE files (plus the HF cache metadata for them) to local SSD inside
the H20-FoldUMM container (`/dockerdata/t210-showo2/`). All three runs below executed with
`HF_HOME=/dockerdata/t210-showo2/hf_cache`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`http_proxy`/`https_proxy` unset. **No shared-storage or network fallback was observed**: a
`grep -rn "apdcephfs_cq7"` across every log in `/dockerdata/t210-showo2/r2_runs/` returned zero
matches. Per local review R11, this is qualified rather than overstated: the evidence is
log-content-based (absence of a shared-storage path string in captured stdout/stderr/wandb logs),
not file-access-syscall-level evidence — no `strace`/`lsof` tracing of the process tree was
performed this round, so it does not prove every individual file-read syscall targeted the SSD path.
A stronger guarantee (e.g. `strace -f -e trace=openat,open` across the full inference process tree)
is a documented limitation, deferred rather than performed this round.

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
| `t2i_fresh1` (generation, fresh process) | ~8.66s | ~6.35s | ~22.0s | 13,269,985,792 B (~12.36 GiB) | 13,627,293,696 B (~12.69 GiB) | 18,805,096 KiB (~17.94 GiB) |

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
| Peak host RSS | 18,804,124 KiB (~17.94 GiB) | 18,805,096 KiB (~17.94 GiB) |

These are measured figures, replacing `first-report.md`'s prior "~10-20GB estimate" for the T2I
path with a concrete number, and adding a first measured figure for the MMU path (not previously
estimated).

### R12 — completed resource/storage fields

- **Local-SSD filesystem type**: `xfs`, on `/dev/mapper/gpu-gpu_volume`, a local block device (not a
  network filesystem) — confirmed via a live `df -T /dockerdata/t210-showo2/hf_cache` run this round
  and classified by `comppareto.repo_state.storage_preflight.classify_filesystem` as `local`; see
  `configs/admission/showo2/storage-preflight.json` (`status: pass`, `filesystem_class: local`).
- **Available capacity before migration**: not separately captured by a dedicated pre-migration `df`
  snapshot; inferred from the post-migration snapshot in `t210_migration.log` (`9.0T size, 15G used,
  9.0T avail, 1% use`) combined with the exact copied-bytes total below — the destination was
  effectively empty before migration (headroom exceeded the ~15GB payload by more than 3 orders of
  magnitude), so pre-migration free space was ~9.0TiB within measurement rounding.
- **Available capacity after migration** (measured directly, `storage_preflight` run):
  capacity 9,895,604,649,984 bytes (~9.0TiB), free 9,880,387,366,912 bytes.
- **Exact copied bytes**: 15,212,441,037 bytes total, sum of `t210_migration.log`'s five
  per-component `dst_bytes` fields (5,661,863,511 + 3,098,960,731 + 3,511,951,720 + 2,432,055,195 +
  507,609,880 — the third-largest figure is the full `CompVis/stable-diffusion-safety-checker` repo
  directory, which includes files beyond the single blob referenced by the manifest's
  `safety-checker-checkpoint` artifact). This is ~4.8MB less than the 15,217,283,072-byte delta
  implied by the post-migration `storage_preflight` used-space reading, attributable to filesystem
  metadata/directory overhead not counted by the per-component `cp` byte totals.
- **Migration wall-clock**: 256.269s (~4m16s) summed across the five `COPY` operations in
  `t210_migration.log` (96.571 + 48.452 + 58.666 + 43.797 + 8.783s); the log's own start/end
  timestamps span 4m17s (`16:09:51` to `16:14:08`), consistent within rounding/logging overhead.
- **T2I host RSS**: 18,805,096 KiB (~17.94 GiB), added to the R5 table above.
- **Measured durable evidence footprint**: 38,941,345 bytes (~37.13 MiB) measured via
  `du -sb /apdcephfs_cq7/.../T210-showo2-admission/` this round — supersedes the prior "~4.5MB"
  estimate below, which undercounted the `.tar` bundle copies.

**Total GPU wall-clock for this smoke round**: `mmu_cold1` (~38.4s) + `mmu_cold2` (~38.2s) +
`t2i_fresh1` (~22.0s) = **~98.6s (~0.0274 GPU-hours)** on a single H20 device. This is smoke-test
scale by design (one image in, one caption/image out per run), not a training or benchmark
workload.

**Footprint**:
- Local SSD footprint (R1 migration): 15,212,441,037 bytes exactly (~14.17 GiB); see R12 field above.
- Shared-storage (`/apdcephfs_cq7`) original footprint: retained unchanged as the provenance copy
  per R1 requirement 5, unaffected by this migration (~37 GB across all fp32 checkpoints, per
  `first-report.md`'s prior estimate; not re-measured this round since the R1 requirement is only
  to *keep*, not to re-audit, this copy).
- R2 evidence footprint (this rerun's logs/artifacts, durable CQ7 copy):
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T210-showo2-admission/` = 38,941,345 bytes
  (~37.13 MiB), measured via `du -sb` this round (R12; supersedes the prior ~4.5MB estimate, which
  undercounted the `.tar` bundle copies).

## Exit-code evidence (metrics.json)

`configs/admission/showo2/timing_wrapper.py` runs each official script in-process via `runpy` rather
than as a subprocess, so it never captures a literal `$?` exit code. `runs/admission-showo2-2026-08-28/metrics.json`'s
`smoke.mmu.exit_code`/`smoke.t2i.exit_code` fields are instead justified by indirect evidence,
checked this round across all three run directories' `stderr.log`/`stdout.log`:

- No `Traceback` string (a Python unhandled-exception marker) appears in any of the six log files.
- `grep -in "traceback\|error"` on each `stderr.log` matches only two benign, non-fatal lines per
  run: a TensorFlow oneDNN informational notice, and `wandb`'s internal git-root detection failing
  gracefully (`git root error: Cmd('git') failed due to: exit code(128)`) — `wandb` catches this
  internally and continues; it does not abort the run.
- Each run's `timing_stats.json` has a populated `output_ready`/`process_end` timestamp (except
  `mmu_cold1`, explained above) and a fully populated `wandb_run/` directory including generated
  media, which would not exist had the script raised before completing its output-write path.
- `mmu_cold1`/`mmu_cold2` produced bit-identical output hashes (reproducibility check above),
  inconsistent with a partial/crashed run.

This is process-output-level evidence, not a literal captured exit code — recorded transparently as
such rather than asserting a `$?` value that was never captured.

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
remain outside Git" allowance in local review R3. Remote reverification of every manifest artifact's
existence, byte size, and SHA-256 (R13) is recorded in
`configs/admission/showo2/artifact-verification.json` (21/21 pass, 0 failed, covering both the
5 immutable-provenance and 5 local-SSD-execution-copy artifacts added per R10). Storage preflight
evidence (R11/R12) is recorded in `configs/admission/showo2/storage-preflight.json`. Exit-code and
resource-summary evidence (R12) is recorded in `runs/admission-showo2-2026-08-28/metrics.json`.

The resolved config used for all three reruns
(`showo2_1.5b_demo_432x432.yaml`, unmodified from `first-report.md`'s original) has sha256
`d9f754ce8bdaf3a96cb6862782b51c781e2b0d3099bf37e15d244048c0559982`.
