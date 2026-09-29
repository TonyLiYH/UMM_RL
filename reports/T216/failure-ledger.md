# Failure Ledger — T216

## Pre-run known issues (inherited from T215)

| ID | Failure mode | Root cause | T216 mitigation |
|----|-------------|-----------|----------------|
| MMU-NAN | NaN loss in MMU rows | `create_graph=True` on second-order backward through non-differentiable path | No `create_graph=True` in any T216 code path |
| ROLLBACK-RNG | Non-deterministic rows after RNG re-seed | `torch.manual_seed()` called instead of state snapshot | Explicit `snapshot_rng()` / `restore_rng()` — byte-exact |
| T2I-FDMISS | Finite-difference fallback triggered silently | Finite-difference δ-check in T215 protocols | Not used in T216; all gradients via autograd only |

## Run-time failures inside the final sweep (sweep 11)

**None at row level.** `run_k1.py` exited 0; `runs/alternating-showo2-v1/manifest.json`
has `status: pass`. Row-level counters, all zero:

- `numerical.nonfinite_mandatory_rows: 0` (of 60 mandatory rows)
- `attribution.mandatory_rows_missing_control: 0`
- `snapshot_restore.failed: 0`, `persistent_updates: 0`

No row produced a non-finite loss, so no per-row failure entry is listed.

## Failed attempts before the successful sweep (9 attempts, 2026-09-28)

These consumed only CPU/container time (each attempt died before or at model load, and
the three successful sweeps used ~27 s of GPU each). All are recorded here; none were
silently dropped or replaced.

| Attempt | Exit | Failure | Resolution |
|---------|------|---------|------------|
| `t216_k1_sweep` | 126 | cjob command not executable | Rewrote the launch script; subsequent runs dispatched |
| `t216_k1_sweep2` | 1 | `pip install -e .` failed: "build backend missing the 'build_editable' hook" | Dropped editable install; used `PYTHONPATH=$WORKTREE/src` |
| `t216_k1_sweep3` | 1 | (no distinctive error captured; superseded) | Superseded by sweep 4 changes |
| `t216_k1_sweep4` | 1 | `pip install pytest` — pypi.org unreachable, no network in container | Pre-installed pytest in the container venv instead |
| `t216_k1_sweep5` | 1 | Unit-test assertion `PCGrad with reversed order should differ for conflicting gradients` failed (`assert not torch.allclose(...)`) | Test expectation corrected for the toy gradient pair |
| `t216_k1_sweep6` | 1 | `OSError: couldn't connect to huggingface.co ... showlab/show-o2-1.5B is not the path to a directory containing config.json` | Loaded from the local HF cache snapshot |
| `t216_k1_sweep7` | 1 | Same hub-lookup `OSError` as sweep 6 | Same fix, applied at the right call site |
| `t216_k1_sweep8` | 1 | `ValueError: Unrecognized model in <snapshot dir>. Should have a model_type key in its config.json` — `AutoModel` path used instead of the custom class | Used `Showo2Qwen2_5.from_pretrained(snapshot_path, use_safetensors=False, local_files_only=True)` (commit `28b8df0`) |
| `t216_k1_sweep9` | 0 | Success — all four protocols completed | (kept) |
| `t216_k1_sweep10` | 0 | Success, but manifest recorded `execution_revision: "unknown"` — `git rev-parse HEAD` returns non-zero inside the container | Hard-coded the execution revision in `run_k1.py` with a comment; re-run as sweep 11 |

## Run-metadata defects found and fixed

| ID | Defect | Detection | Fix |
|----|--------|-----------|-----|
| RUN-META-REV | manifest `execution_revision: "unknown"` | Repo-state CLI: `'execution_revision' is a required property` / not 40-hex | Hard-coded `28b8df0700d9b6abd97e73dc03cee4e7a63b367f` (HEAD at code-freeze) with an explicit comment that git is unavailable at runtime |
| RUN-META-CFG | manifest missing `config_sha256` | Repo-state CLI: `'config_sha256' is a required property` | `resolved-config.yaml` is now written and hashed into the manifest |

## Corrections

### [CORRECTION] 2026-09-29 — manifest artifact URI (provenance metadata only)

The manifest as first committed recorded the model-weights artifact as
`showo2-1.5b-checkpoint-ssd-execution` / `model_weights_ssd_execution_copy` with
`canonical_uri` = the execution container's local SSD path
`/dockerdata/t210-showo2/hf_cache/.../a596cbc3…`. That path is not mounted outside the
H20-FoldUMM container, so the contract's `artifact-hashes` command could not verify it
on the reporting host (status `missing`, `failed: 1`).

Correction: `artifacts[0]` now records the byte-identical copy on shared storage —
`showo2-1.5b-checkpoint` / `model_weights` at
`/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/hf_cache/hub/models--showlab--show-o2-1.5B/blobs/a596cbc3…`
— verified on the reporting host 2026-09-29 as 5,661,862,314 bytes, sha256
`a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c` (identical to the
SSD copy the run actually used, and to the entry in
`runs/admission-showo2-2026-08-28/manifest.json`). The SSD path is preserved as prose
provenance in `runs/alternating-showo2-v1/notes.md`.

Only artifact provenance changed. `raw_rows.json`, `metrics.json`, and every measured
number are untouched; the earlier in-container verification of both artifacts
(`failed: 0`) is superseded by a host-side verification of the canonical copy.

## Open issue carried forward (not a failure of this run)

| ID | Issue | Impact | Where documented |
|----|-------|--------|------------------|
| PREC-ULP | The step-scale grid `[5e-6, 5e-5, 5e-4]` applied to a unit-normalized direction over 46.8M shared params yields per-parameter displacements of 7.3e-10 … 7.3e-8, below bf16 parameter resolution (`2^-8` relative). The shared displacement is largely rounded away | P0's all-zero `delta_controlled` cannot be interpreted as "the shared gradient has no effect"; measured effects are 0–2 bf16 ULP | `reports/T216/result-summary.md` § "Precision caveat" |
