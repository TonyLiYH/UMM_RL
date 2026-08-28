# Run note — `admission-showo2-2026-08-28`

Formal admission run for T210 (Show-o2 checkpoint/training-interface/evaluation admission),
executed per local review R1-R5 revisions. `manifest.json` in this directory is the schema-conformant
record; this note explains what it summarizes and where the full narrative lives.

## What this run covers

Three read-only inference smoke passes, all sourced from local SSD
(`/dockerdata/t210-showo2/`) inside the H20-FoldUMM GPU container, after the R1 asset migration:

1. `mmu_cold1` — first cold-process understanding (MMU) smoke.
2. `mmu_cold2` — second cold-process understanding smoke, same SSD cache, run to (a) confirm
   determinism against `mmu_cold1` and (b) recover the `output_ready` timestamp missed in
   `mmu_cold1` (see limitation below).
3. `t2i_fresh1` — fresh-process generation (T2I) smoke.

No training, joint or otherwise, was run. No Show-o2 source file was modified — timing/memory
instrumentation (`configs/admission/showo2/timing_wrapper.py`) works by monkeypatching public
`torch`/`wandb` entry points around an unmodified `runpy.run_path` invocation of the official
`inference_mmu.py`/`inference_t2i.py` scripts.

## Where the full evidence and narrative live

- **Timing, memory, hash, and footprint results**: `reports/T210/r2-r5-ssd-rerun.md`.
- **R1 migration record** (paths, filesystem type, capacity, bytes, wall-clock, hash
  verification): `t210_migration.log` / `t210_hash_manifest.txt`, referenced as manifest artifacts
  `r1-migration-log` / `r1-hash-manifest`.
- **License/provenance resolution (R6)**: `reports/T210/failure-ledger.md`, section "Resolution
  update (T210 R6, 2026-08-28)".
- **Environment freeze (R7)**: `configs/admission/showo2/environment-lock.md`.
- **Block registry**: `configs/admission/showo2/parameter-block-registry.yaml` (produced in an
  earlier T210 stage, unchanged by this run; listed in `result_files` for completeness since R3
  requires the manifest to reference it).
- **Raw per-run logs/stats/wandb media**: durable copies at
  `/apdcephfs_cq7/share_1447896/yihangli/outputs/T210-showo2-admission/r2_runs/{mmu_cold1,mmu_cold2,t2i_fresh1}/`,
  each also bundled into a single hash-addressed `.tar` referenced as a manifest artifact.

## Known limitations (documented, not silently omitted)

1. **`mmu_cold1` output timing gap**: its `timing_stats.json` lacks an `output_ready` timestamp,
   because the wrapper's first-iteration `wandb.log` patch did not catch Show-o2's actual call
   style (`run.log(...)` on the `Run` instance, not the module-level function). The wrapper was
   fixed before `mmu_cold2` ran (also patches `wandb.sdk.wandb_run.Run.log`); `mmu_cold1`'s total
   wall-clock and model-load timing are unaffected, only its inference-phase sub-split is missing.
   `mmu_cold2`'s figures are used as the authoritative model-ready→output timing for the
   understanding path.
2. **Same-process warm inference (R2, "where practical") was not attempted.** Both official
   inference scripts are single-shot CLIs with no loop entry point; building a custom driver that
   keeps the process warm would require reimplementing the model-load/generation call sequence
   outside the audited script, risking divergence from the exact code path this admission is
   required to measure. Documented as a limitation in `reports/T210/r2-r5-ssd-rerun.md` rather than
   attempted with a risky custom harness.
3. **`CompVis/stable-diffusion-safety-checker` license remains genuinely unspecified upstream** (not
   an oversight on our side — reconfirmed via a live HF Hub API query on this run's execution date).
   Formally constrained per R6 as an optional, display-only dependency with a documented
   safety-checker-free evaluation path; not blocking for this read-only smoke.

4. **One pre-existing repository test now fails, out of scope to fix here.**
   `tests/repo_state/test_cli.py::test_cli_validates_repository` hardcodes
   `"run_manifests=pass manifests=1"`, assuming exactly one run manifest exists repo-wide
   (`runs/t1_synthetic/t1_manifest.json`). Adding this run's `manifest.json` makes the true count 2,
   so that literal string no longer appears in the CLI's stdout (the validator itself still reports
   `run_manifests=pass`, i.e. both manifests are schema-valid — only the hardcoded count assertion
   fails). `tests/repo_state/` is outside T210's declared `allowed_paths`
   (`tasks/T210-showo2-admission.md`, `configs/admission/showo2/`, `runs/admission-showo2/`,
   `reports/T210/`, `src/comppareto/adapters/showo2/`, `tests/adapters/`), so this task does not edit
   that test. Flagged here for the local reviewer rather than silently worked around.

## Status

`pass` — both task paths executed to completion on H20 GPU 0, from local SSD only, with zero
observed shared-storage/network fallback, deterministic reproduction confirmed for the
understanding path, and measured (not estimated) peak VRAM for both paths.
