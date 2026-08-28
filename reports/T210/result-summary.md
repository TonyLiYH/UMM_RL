# T210 result summary

Task: `tasks/T210-showo2-admission.md`. Branch: `agent/T210-showo2-admission`. Source revision
audited: Show-o2 commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23` (recorded in `first-report.md`).

## Current admission run (authoritative status)

The formal admission run is `runs/admission-showo2-2026-08-28/` — see
`runs/admission-showo2-2026-08-28/manifest.json` (`status: pass`, 21 hash/byte-addressed artifacts,
each with a separate immutable-provenance `canonical_uri`; for the four migrated model
components plus the safety checker, the local-SSD copy actually read at execution time is recorded
as its own separate sibling artifact — e.g. `showo2-1.5b-checkpoint-ssd-execution` — rather than a
nested manifest field, since `schemas/run-manifest.schema.json` sets `additionalProperties: false`
on artifact objects and only the "separate artifact" alternative offered by local review R10 is
schema-valid) and
`runs/admission-showo2-2026-08-28/metrics.json` (smoke exit-code evidence and resource metrics, per
R12). This is an admission/smoke run (one MMU pass, one T2I pass), not a training run; see "No
training run" below for that distinction.

Measured, current (SSD-sourced) figures, superseding every timing/VRAM number below this section
that predates the R1 SSD migration (full detail: `reports/T210/r2-r5-ssd-rerun.md`):

- **Model load time**: ~8.7-9.4s per cold process from local SSD (`/dockerdata/t210-showo2/`,
  `xfs` local block device, confirmed via live `df -T` — `configs/admission/showo2/storage-preflight.json`),
  vs. the ~26 minute load previously measured from `/apdcephfs_cq7` shared network storage.
- **Peak GPU memory**: MMU ~13.87GiB allocated / ~38.48GiB reserved; T2I ~12.36GiB allocated /
  ~12.69GiB reserved.
- **Peak host RSS**: MMU ~17.94GiB (18,804,124-18,804,728 KiB across two cold runs); T2I ~17.94GiB
  (18,805,096 KiB) — both paths have comparable host RSS, dominated by the same resident model
  weights.
- **GPU wall-clock / GPU-hours**: ~98.6s total across the three reruns (~0.0274 GPU-hours) on a
  single H20 device — smoke-test scale by design.
- **Local-SSD storage footprint**: exactly 15,212,441,037 bytes (~14.17GiB) copied across the four
  required components plus HF cache metadata (sum of `t210_migration.log`'s per-component
  `dst_bytes`); durable evidence footprint (raw logs/artifacts, CQ7 copy) ≈ 4.5MB
  (38,941,345 bytes measured via `du -sb`, including the tarball copies of each run directory).
- **Safety-checker-free evaluation path**: defined and unexercised. `reports/T210/failure-ledger.md`
  specifies that any downstream scientific evaluation (FID/CLIP-score/etc.) should read
  `images`/`pil_images` directly at `inference_t2i.py:198-199`, before the safety-checker call at
  line 201, and never import `StableDiffusionSafetyChecker`. This task's own T2I smoke used the
  official script unmodified (safety checker included, matching the audited code path); the
  safety-checker-free path itself was not separately executed this round, since doing so would
  require a modified/forked script outside this task's frozen-protocol scope.
- **No-fallback evidence**: qualified per local review R11 — see "Fallback evidence" below.

## What was built/run

- First report (audit, no GPU execution yet): `reports/T210/first-report.md`.
- Environment + checkpoint smoke: `reports/T210/environment-checkpoint-smoke.md` — GPU environment
  built on H20-FoldUMM, checkpoint(s) downloaded and hash-verified, one matmul smoke on `cuda:0`.
  Fixed one upstream defect (`build_env.sh`'s unpinned torch/torchvision install clobbered by a
  later step; restored the official pinned versions).
- Understanding + generation task-path smoke: `reports/T210/task-path-smoke.md` —
  `inference_mmu.py` and `inference_t2i.py` both run to completion on GPU with the pinned
  `showlab/show-o2-1.5B` checkpoint, using only official code/config plus environment-variable and
  official-CLI-override adaptation (`HF_HOME` redirection, `WANDB_MODE=offline`, a symlinked
  `Wan2.1_VAE.pth`, a reduced `num_inference_steps` for smoke speed). Fixed a second upstream
  defect (`build_env.sh`'s unpinned `wandb` install, incompatible with the resolved `protobuf`
  version and missing an API the inference scripts call).
- Parameter-block registry draft: `reports/T210/parameter-block-registry.md` and machine-readable
  `configs/admission/showo2/parameter-block-registry.yaml` — exact weight-level enumeration of
  `named_parameters()` on the loaded model (3,063,740,640 params), cross-checked against forward-pass
  data flow, refining and correcting the stage-1 provisional config-schema-only reading.
- No adapter code was written under `src/comppareto/adapters/showo2/` or `tests/adapters/` — none
  was required to satisfy this task's pass/fail gate (both paths execute using unmodified official
  scripts); adapter code, if any, is scoped to a successor task's training-interface work.
- No joint post-training run — none was authorized or attempted, per the frozen protocol ("do not
  begin joint post-training"). This is distinct from the admission/smoke run itself: a formal
  `runs/admission-showo2-2026-08-28/` manifest and run note now exist (added per local review R3),
  but they record one MMU smoke pass and one T2I smoke pass on the frozen pinned checkpoint — not a
  training run of any kind.

## Checkpoints, revisions, and licenses recorded (cumulative across all stages)

| Component | Repo ID / source | Resolved revision | Hash | Size | License |
|---|---|---|---|---|---|
| Show-o2 LLM+heads | `showlab/show-o2-1.5B` | see `environment-checkpoint-smoke.md` | see `environment-checkpoint-smoke.md` | ~22GB (fp32) | Apache-2.0 (repo `LICENSE`) |
| LLM tokenizer/backbone | `Qwen/Qwen2.5-1.5B-Instruct` | see `first-report.md` | see `first-report.md` | — | Apache-2.0 |
| Vision encoder (und-private source) | `google/siglip-so400m-patch14-384` | see `first-report.md` | see `first-report.md` | — | Apache-2.0 |
| 3D causal VAE (frozen) | `Wan-AI/Wan2.1-T2V-14B` (`Wan2.1_VAE.pth` only) | `a064a6c71f5be440641209c07bf2a5ce7a2ff5e4` (resolved R6 via live HF Hub query) | sha256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` | 507,609,880 bytes | Apache-2.0 (resolved R6 — no longer an open item) |
| Safety checker (generation-path only; newly discovered this stage) | `CompVis/stable-diffusion-safety-checker` | `697f152c6b6183309a1509fb3589fcbaae7f34e5` | sha256 `64b8393f1afd5a0c1ed2aa5f341fa7c08286839a48f3743162a76a2835c808bd` (`model.safetensors`) | 2.3GB (cache) | **still unspecified on model card ("More information needed"); formally constrained as an optional, display-only dependency with a documented safety-checker-free evaluation path — not blocking, see failure-ledger.md** |

## Environment defects found and fixed (both upstream, both unpinned transitive deps)

1. `build_env.sh`'s unpinned `pip3 install torch`/a later step clobbering the pinned CUDA build —
   fixed by reinstalling the pinned `torch==2.5.1+cu124`/`torchvision==0.20.1+cu124`. Detailed in
   `environment-checkpoint-smoke.md`.
2. `build_env.sh`'s unpinned `pip3 install wandb` resolving to `wandb==0.29.0`, incompatible with
   the resolved `protobuf==4.25.9` (missing dispatcher branch) and missing `wandb.util.generate_id`
   (removed upstream) — fixed by pinning `wandb==0.17.0`. Detailed in `task-path-smoke.md`.

Both are dependency-version restorations for packages the official build script itself installs
unpinned, not substitutions of any Show-o2 component, checkpoint, or logic — consistent with "do
not use unofficial fixes... without local authorization," and fully recorded rather than silently
patched.

## GPU/hardware evidence

H20-FoldUMM container, 8x H20 96GB. GPU0 kept exclusively free of the benign `train2.py` placeholder
occupancy and used for all smoke work (`cuda:0`); GPU1-7's placeholder processes were never touched.
`inference_mmu.py`'s original shared-storage run (`task-path-smoke.md`, dated 2026-08-27) measured
~26 min model load + <1 min inference; this figure is historical and superseded — after the R1
local-SSD migration, three reruns (`reports/T210/r2-r5-ssd-rerun.md`) measured ~8.7-9.4s model load
for both paths, ~150x faster, with peak VRAM/RSS figures listed in "Current admission run" above.
Both `inference_mmu.py` and `inference_t2i.py` completed with exit 0 in every run performed
(original shared-storage run and all three SSD reruns); the SSD reruns' `stderr.log` files contain
one benign, non-fatal `wandb` git-root detection warning (`git root error: Cmd('git') failed due to:
exit code(128)`) and TensorFlow's routine oneDNN informational line, and no `Traceback` or fatal
`Error` string — checked via `grep -in "traceback\|error"` across all three run directories'
`stderr.log` files this round.

## Fallback evidence (R11)

Local review R11 required either file-access-level evidence of zero shared-storage/network fallback,
or a softened claim with a documented limitation. This task takes the latter: **no shared-storage or
network fallback was observed** during the three SSD reruns (`HF_HOME` pointed at
`/dockerdata/t210-showo2/hf_cache`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, proxy variables
unset; a `grep -rn "apdcephfs_cq7"` across every run log returned zero matches). This is
log-content-based evidence, not file-access-syscall-level evidence (no `strace`/`lsof` tracing was
performed this round); it does not prove every individual file read targeted the SSD path, only
that no shared-storage path string appeared in any captured log or error output. A stronger
guarantee would require `strace -f -e trace=openat,open` (or equivalent) across the full inference
process tree, which is deferred as a documented limitation rather than performed this round.

## Conclusion

**Supports gate.** Both task paths execute end-to-end on GPU with the pinned checkpoint using only
official code/config; every checkpoint/revision touched is recorded, including one newly-discovered
external component this stage; shared/private parameter blocks are enumerated exactly at the weight
level and are auditable. Two upstream environment defects were found and fixed via dependency-pin
restoration, both documented. Of the two original license-status open items, the Wan2.1 VAE item is
now fully resolved (Apache-2.0, revision `a064a6c71f5be440641209c07bf2a5ce7a2ff5e4`) and the safety
checker item remains formally constrained as an optional, non-blocking, display-only dependency with
a documented safety-checker-free evaluation path (`failure-ledger.md`) — neither is an unresolved
blocker. The formal admission run (`runs/admission-showo2-2026-08-28/manifest.json`, `status: pass`)
records 21 hash/byte-addressed artifacts, each remotely reverified this round
(`configs/admission/showo2/artifact-verification.json`, 21/21 pass, 0 failed) with existence, byte
size, and SHA-256 confirmed against the manifest's declared values. No joint post-training was
started or attempted.
