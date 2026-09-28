# T220 failure ledger

## Unresolved anomalies (require local-reviewer decision)

### 1. No repository license (`MCG-NJU/UniDDT`)

- **Component**: the official GitHub repository itself.
- **Issue**: no `LICENSE`/`LICENSE.md` file exists at the repo root (confirmed via a full recursive
  tree listing and direct raw-content fetches, both returning 404, plus the GitHub license API
  returning "Not Found"). The repository is public with an active issue tracker, so this appears to
  be an authoring oversight, not an intentionally withheld license.
- **Disposition**: not blocking for this task's read-only audit/reproduction smoke — no
  redistribution occurred, and the code was used exactly as published, unmodified.
- **Recommendation for local review**: an explicit decision is needed before any onward use of this
  admission's evidence beyond audit/reproduction (e.g. training, redistribution, publication),
  since the absence of any license is a materially different situation from T210's "unspecified but
  formally constrained, non-blocking" safety-checker item — here it is the *primary* model's own
  code, not a peripheral optional dependency.

### 2. No checkpoint license (`MCG-NJU/UniDDT` HF model repo)

- **Component**: `vlm_uniddt_512.ckpt` (and the untargeted `vlm_uniddt_1024.ckpt`).
- **Issue**: the HF model card carries no `license` field (`tags: ["region:us"]` only) and the repo
  is not gated. This compounds item 1 above.
- **Disposition**: not blocking for this task's read-only smoke — the checkpoint was downloaded via
  the official `huggingface_hub` API, hash-verified, and used unmodified for inference only.
- **Recommendation for local review**: same as item 1 — requires explicit decision before onward
  use beyond audit/reproduction.

### 3. FLUX VAE license ambiguous (`diffusers/FLUX.1-vae`)

- **Component**: `diffusion_pytorch_model.safetensors`, the frozen visual-latent-space VAE loaded
  by `LatentAE`.
- **Issue**: this repacking repo declares no `license` field. Its `config.json` values match the
  published FLUX.1 VAE configuration exactly, but that configuration is shared verbatim between
  `black-forest-labs/FLUX.1-dev` (non-commercial license, gated) and
  `black-forest-labs/FLUX.1-schnell` (apache-2.0, gated) — config values alone cannot disambiguate
  which upstream license this specific export inherits.
- **Disposition**: not blocking for this task's read-only inference smoke (the VAE is used strictly
  for encode/decode, frozen, never trained, no redistribution).
- **Recommendation for local review**: resolving this exactly would require downloading and
  hash-comparing the VAE weights against both official FLUX.1-dev and FLUX.1-schnell sources; not
  attempted this stage (deferred, consistent with `first-report.md`'s "first report: metadata work
  only" framing carried forward — no new evidence was sought this round either). Needed before any
  onward use whose license implications depend on which FLUX variant this export derives from.

## Resolved issues (not outstanding — recorded for completeness)

### 1. `einops` missing from `requirements.txt` (environment gap)

- **Symptom**: `import app_uniddt` failed immediately with
  `ModuleNotFoundError: No module named 'einops'`. Traced to `src/utils/packed_seqs/seqs.py`, which
  imports `einops` unconditionally at module scope; that module is imported transitively by
  `app_uniddt.py` (and by every diffusion-sampling entry point), so the failure occurred before any
  model code ran.
- **Fix**: installed `einops==0.8.2` (current PyPI release at audit time) into the dedicated venv.
  No version is pinned upstream for this package since it is not listed in `requirements.txt` at
  all — this is a gap-fill for a missing transitive dependency, not a substitution of any pinned
  version the official file specifies.
- **Verified**: `import app_uniddt` and the full `build_pipeline` smoke stage succeeded afterward
  with `exit_code: 0`.

### 2. Missing `refs/main` files blocking offline resolution (environment/cache gap)

- **Symptom**: with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set (required per
  `dev-env-paths.md`'s local-execution requirement), loading either `diffusers/FLUX.1-vae` or
  `Qwen/Qwen3-VL-4B-Instruct` via `from_pretrained` raised
  `OSError: diffusers/FLUX.1-vae does not appear to have a file named config.json`, even though the
  snapshot directory for the pinned revision was present on disk with `config.json` inside it.
- **Root cause**: both repos were downloaded by explicit pinned revision (not `"main"`), so
  `huggingface_hub`'s cache never wrote a `refs/main` pointer file for either repo. In fully offline
  mode, `from_pretrained` calls that resolve `revision=None`/`"main"` (the default used by
  `app_uniddt.py`'s loading code) consult `refs/main` first and fail outright if it is absent, even
  though the actual snapshot data is present and correct.
- **Fix**: manually created `refs/main` files for both repos, each containing the exact revision
  hash already pinned and hash-verified (`da548cfb003bdeebaff6da0211fc8fbc67cb563a` for
  `diffusers/FLUX.1-vae`, `ebb281ec70b05090aa6165b016eac8ec08e71b17` for
  `Qwen/Qwen3-VL-4B-Instruct`). This does not substitute a different revision or file content — it
  only restores the pointer file the official download flow would have written had the repos been
  fetched via a `"main"`-relative call instead of an explicit-revision call.
- **Verified**: `build_pipeline` smoke stage succeeded afterward with `exit_code: 0`, loading both
  repos from the local offline cache with no network access.

### 3. Blob-filename mismatch in the first artifact-verification attempt (audit-script bug, not an
   environment/upstream defect)

- **Symptom**: the first `artifact-verification.json` draft reported 4 of 9 declared artifacts as
  "missing" (file-not-found at the expected `canonical_uri`).
- **Root cause**: the verification input manifest assumed every HF cache blob's on-disk filename
  equals its own sha256 hash. This holds for large binary blobs (the checkpoint, the VAE
  safetensors file) but not for small text/config files, where `huggingface_hub`'s cache instead
  names the blob file after the *source* blob hash reported by the Hub API (a different value from
  the file's own locally-computed sha256, since HF may serve some small files without content-based
  blob naming matching a simple sha256). Enumerating the actual on-disk blob filenames
  (`/apdcephfs_cq9/share_1447896/yihangli/tmp/t220/t220-list-hashes.sh`) and correcting the
  `canonical_uri` paths in the input manifest to match resolved this immediately — the file
  *contents* and their sha256 values were never in question, only the verification script's
  filename assumption.
- **Fix**: corrected the input manifest's `canonical_uri` values to the actual on-disk blob
  filenames; reran `scripts/verify_manifest_artifacts.py`.
- **Verified**: `configs/admission/uniddt/artifact-verification.json` — 9/9 checked, 9 passed, 0
  failed, on the corrected manifest.

### 4. Nested-quoting truncation in remote `cjob`/`taiji_client exec` invocations (tooling issue, not
   an UniDDT-specific defect)

- **Symptom**: several early attempts to run multi-layer-quoted commands (an outer `script -qec`
  wrapper around `taiji_client exec ... bash -c "..."` around an inner `bash cjob.sh start ...
  "bash -c \"...\""`) were silently truncated or misparsed by shell quote-nesting, producing
  partial or no output rather than a clean error.
- **Fix**: adopted the pattern of writing every non-trivial command to a standalone `.sh` file under
  `/apdcephfs_cq9/share_1447896/yihangli/tmp/t220/` and invoking it via
  `bash cjob.sh start <name> "bash <script-path>"`, eliminating all but one level of quoting.
- **Verified**: every subsequent long-running command (venv build, downloads, hash verification,
  smoke run, block registry enumeration) completed cleanly using this pattern, with output
  recoverable from the `cjob` log files.

## Summary

- No task-path execution failure occurred: both `understanding` and `generation` smoke stages ran
  to completion (`exit_code: 0`) once the two environment gaps below were fixed.
- Of the 3 license-status open items carried forward from `first-report.md` (repository, checkpoint,
  FLUX VAE): **none were resolved further this stage** — all three remain explicitly recorded,
  unresolved, non-blocking open items requiring local-reviewer decision before any onward use of
  this admission's evidence beyond audit/reproduction.
- 2 environment/cache gaps found and fixed (missing `einops` dependency, missing `refs/main` offline
  cache pointer files), both fully documented; neither substitutes any UniDDT source, config, or
  checkpoint.
- 1 audit-script bug (blob-filename assumption) found and fixed in this task's own verification
  tooling, not in any UniDDT or upstream component.
- 1 tooling/process issue (nested-quoting truncation in remote exec invocations) resolved by
  adopting a standalone-script invocation pattern; no data loss occurred.
- No unofficial fix was applied to UniDDT source, config, or checkpoint; no different checkpoint or
  architecture was substituted; no joint/duality post-training was started, per the frozen protocol.
