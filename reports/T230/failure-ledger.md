# T230 failure ledger

## Unresolved anomalies (require local-reviewer decision)

### 1. No CQ7 shared-storage canonical copy of the checkpoint

- **Component**: `sensenova/SenseNova-U1-8B-MoT`, downloaded directly to local SSD
  (`/dockerdata/t230-sensenova/checkpoints/SenseNova-U1-8B-MoT`) with no intermediate
  HF-cache-on-CQ7 step.
- **Issue**: per `dev-env-paths.md`, pretrained model weights normally belong on CQ7
  (`models/pretrained/`). No CQ7 copy of this specific checkpoint exists (confirmed via `find`
  across `/apdcephfs_cq7/share_1447896/yihangli/models`, no match). Unlike T210, whose manifest
  paired every model artifact with both a CQ7 canonical copy and a local-SSD execution copy,
  T230's `manifest.json` lists each checkpoint file as a single `/dockerdata`-only artifact.
- **Disposition**: not blocking for this admission — both required smokes executed exclusively
  from the verified local-SSD copy (`configs/admission/sensenova-u1/storage-preflight.json`,
  `status: pass`, `filesystem_class: local`), satisfying the resource envelope's "weights and
  caches must execute from verified local SSD" requirement. Every checkpoint file is still
  hash/byte-verified (`configs/admission/sensenova-u1/artifact-verification.json`, 10/10 pass for
  the checkpoint files specifically).
- **Recommendation for local review**: if a durable CQ7 copy is required before any successor task
  builds on this checkpoint, a `cp`/`rsync` of the `/dockerdata` checkpoint to
  `/apdcephfs_cq7/share_1447896/yihangli/models/pretrained/` should be performed and the manifest
  updated with a paired canonical/ssd-execution artifact structure matching T210's pattern. Not
  done in this task since no requirement mandated it and the ~32.7GiB copy would add
  wall-clock/storage cost outside this admission's smoke-test scope.

## Resolved issues (not outstanding — recorded for completeness)

### 1. `flash-attn` optional dependency not installed

- **Symptom**: the repository's reference environment uses a CUDA-specific `flash-attn` wheel
  (`flash_attn 2.8.3+cu12torch28cxx11abitrue-cp311-*`) not generically hosted on PyPI; no
  locally-built or hand-matched wheel for this exact `torch==2.8.0`/CUDA 12.8/`cp310` combination
  was available.
- **Disposition**: non-blocking. The repository's own `pyproject.toml` documents that the model
  "transparently falls back to torch SDPA when absent" (comment referencing
  `src/sensenova_u1/models/neo_unify/modeling_qwen3.py`). Both smokes ran on the SDPA attention
  path (`--attn_backend sdpa`) and completed with `exit_code 0`; neither showed any
  attention-related error. Documented in
  `configs/admission/sensenova-u1/environment-lock.md`.

### 2. Mixed understanding+generation forward path unavailable (upstream `NotImplementedError`, issue #207)

- **Symptom**: `src/sensenova_u1/models/neo_unify/modeling_qwen3.py`'s decoder-layer/attention
  `forward()` methods only implement the two pure cases (all-understanding tokens or
  all-generation tokens). The mixed case explicitly raises
  `NotImplementedError("The mixed und/gen forward path is not yet validated (issue #207): known issues are fixed, but it has no parity test and no production caller. Split the sequence at token-type boundaries and use forward_und / forward_gen.")`.
- **Disposition**: not a failure of this admission. This is a static, code-visible limitation of
  the released repository, explicitly recorded as the required routed-overlap static assumption
  violation (`runs/admission-sensenova-u1-v1/metrics.json`'s
  `routed_overlap.static_assumption_violation`, `static_assumption_violations_recorded: true`).
  Both required smokes (VQA-only understanding, T2I-only generation) each exercise exactly one
  pure branch and never trigger this code path; both completed with `exit_code 0`. Per the frozen
  protocol ("an unreleased pipeline is a blocker, not permission to recreate it"), this admission
  did not attempt to implement, patch, or work around the missing mixed-modality path.

## Non-issues actively checked and ruled out

### 1. Non-Apache third-party dependency (VAE / safety-checker-style component)

- **Checked because**: T210/Show-o2 required tracking a separately-licensed VAE and safety
  checker; T230's `first-report.md` flagged this as an open item to re-verify once `config.json`
  was inspected.
- **Finding**: no such component was found. The 8B-MoT checkpoint bundles its own
  vision/generation components under the `sensenova_u1` package; both the repository `LICENSE`
  and the HF model card (`cardData.license`) report `apache-2.0` for the sole checkpoint used.
- **Disposition**: not an issue; recorded as a genuine structural difference from T210, not an
  oversight.

### 2. Environment corrective pins

- **Checked because**: T210 required three corrective dependency-pin restorations for its
  official-but-unpinned `build_env.sh`.
- **Finding**: SenseNova-U1's `requirements.txt`/`pyproject.toml` pins resolved exactly as
  declared; no repair was needed (`configs/admission/sensenova-u1/environment-lock.md`).
- **Disposition**: not an issue.

## Summary

- No task-path execution failure occurred: both `examples/vqa/inference.py` and
  `examples/t2i/inference.py` ran to completion on the first attempt, with `exit_code 0` captured
  directly from the wrapper script's `$?`.
- 0 environment defects found (vs. 2 for T210); the one documented optional-dependency omission
  (`flash-attn`) is non-blocking per the repository's own documented SDPA fallback.
- 1 unresolved-but-non-blocking anomaly (no CQ7 canonical copy of the checkpoint), left open for
  local-reviewer decision on whether a successor task should add one.
- 1 static, code-visible upstream limitation (mixed und/gen forward path, issue #207) recorded as
  the required routed-overlap static assumption violation, not worked around or reimplemented.
- No unofficial fix was applied to SenseNova-U1 source, config, or checkpoint; no U1.5 asset was
  substituted; no joint post-training or training-pipeline recreation was started, per the frozen
  protocol.
