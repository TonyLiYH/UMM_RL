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
- **Storage preflight (R11/R12)**: `configs/admission/showo2/storage-preflight.json`.
- **Remote artifact reverification (R13)**: `configs/admission/showo2/artifact-verification.json`.
- **Exit-code/resource-summary evidence (R12)**: `runs/admission-showo2-2026-08-28/metrics.json`.
- **Immutable provenance vs. SSD-execution copies (R10)**: `manifest.json` records each migrated
  component's local-SSD copy as its own separate sibling artifact (`*-ssd-execution` artifact IDs)
  rather than a nested field on the provenance artifact, since `schemas/run-manifest.schema.json`
  sets `additionalProperties: false` on artifact objects — the "separate artifact" alternative from
  local review R10's wording is the only schema-valid option.

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
4. **~~Remote artifact reverification (R13) covers `canonical_uri` (provenance) paths, not
   `ssd_execution` paths~~ — no longer applicable.** An earlier draft of this manifest recorded the
   local-SSD execution copy as a nested `ssd_execution` field, which `scripts/verify_manifest_artifacts.py`
   does not inspect. That draft was schema-invalid (`schemas/run-manifest.schema.json` sets
   `additionalProperties: false` on artifact objects) and was replaced with 5 separate sibling
   `*-ssd-execution` artifact entries, each with its own `canonical_uri` pointing at the
   `/dockerdata` copy. `scripts/verify_manifest_artifacts.py` verifies these the same as any other
   artifact — `configs/admission/showo2/artifact-verification.json` now reports 21/21 pass (10 model
   artifacts covering 5 components x {provenance, SSD-execution}, plus 11 pre-existing run-evidence
   artifacts), so both the shared-storage provenance copy and the local-SSD execution copy of every
   migrated component are independently reverified. Note on *how*: the 11 pre-existing artifacts
   and the 5 provenance artifacts resolve to `/apdcephfs_cq7`/`/apdcephfs_cq9` shared-storage paths,
   reachable directly from this checkout; the 5 `*-ssd-execution` artifacts resolve to `/dockerdata`
   paths, which only the H20-FoldUMM GPU container mounts, so `scripts/verify_manifest_artifacts.py`
   was run inside that container (via `taiji_client exec`) to produce their portion of
   `artifact-verification.json`; the combined output is committed here as the durable, hash-addressed
   log R13 requires.
5. **Migration-log discrepancy of ~4.8MB between exact copied bytes and post-migration used-space
   reading**: see `reports/T210/r2-r5-ssd-rerun.md`'s R12 "Exact copied bytes" field — attributed to
   filesystem metadata/directory overhead, not re-measured at the byte level this round.

## Superseded limitation (resolved by the `origin/main` merge, left here for the record)

The previous version of this note flagged
`tests/repo_state/test_cli.py::test_cli_validates_repository` as failing because it hardcoded
`manifests=1`. The `origin/main` merge (commit `a07a939` and predecessors) already fixed this
upstream: the test now computes `expected_manifest_count = len(list(Path("runs").rglob("*manifest.json")))`
dynamically rather than hardcoding a count, so it passes regardless of how many run manifests exist
repo-wide. No longer a limitation; confirmed by the full test suite run below.

## Status

`pass` — both task paths executed to completion on H20 GPU 0, from local SSD only, with no observed
shared-storage/network fallback (log-content-based evidence; see
`reports/T210/r2-r5-ssd-rerun.md`'s R11 qualification for the file-access-level-evidence limitation),
deterministic reproduction confirmed for the understanding path, and measured (not estimated) peak
VRAM/RSS for both paths. Remote reverification of every declared manifest artifact
(`configs/admission/showo2/artifact-verification.json`): 21/21 pass, 0 failed. Storage preflight
(`configs/admission/showo2/storage-preflight.json`): `status: pass`, `filesystem_class: local`.
Exit-code and resource-summary evidence: `runs/admission-showo2-2026-08-28/metrics.json`.
