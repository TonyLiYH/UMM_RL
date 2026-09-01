# T210 local review — 2026-08-28

Decision: `revision_needed`.

Reviewed branch: `origin/agent/T210-showo2-admission` at
`ce2988832413e1c0defc6bf7095239a466894e02`.

## Verified progress

- The first report was committed before GPU execution.
- The official Show-o2 source and primary checkpoints were pinned.
- Both understanding and generation inference paths were reported to exit
  successfully on an H20 GPU.
- A weight-level parameter-block registry was produced.
- The branch passes the current repository validator and 30 local tests.

## Required revisions

### R1 — migrate model assets and runtime cache to GPU-container local SSD

The approximately 26-minute model-load time is likely dominated by reading
large weights and cache objects from shared `/apdcephfs_*` storage. Before
profiling or repeating the smoke, identify a persistent local SSD path inside
the GPU container using filesystem and mount information. The selected path
must not resolve to the shared network filesystem.

Copy the following to the local SSD:

- Show-o2 1.5B checkpoint;
- Qwen2.5 backbone/tokenizer files used at runtime;
- SigLIP files;
- Wan2.1 VAE;
- safety checker if it remains in the smoke path;
- Hugging Face cache metadata required to prevent network or shared-filesystem
  fallback.

Requirements:

1. Record source and destination paths, filesystem type, available capacity,
   copied bytes, and transfer wall-clock.
2. Verify every weight file against the already recorded SHA-256 hash; generate
   and retain hashes for files that lack one.
3. Configure `HF_HOME`, model paths, and any cache variables so the repeated
   smoke reads from the SSD only.
4. Prove from resolved paths and access logs, or an equivalent audit, that the
   run did not fall back to `/apdcephfs_*` or download weights from the network.
5. Keep the shared-storage copy as the provenance source; do not delete it.

### R2 — measure cold-process and warm-process load performance from SSD

Run and record at least:

- a fresh-process understanding smoke from the SSD;
- a second fresh-process understanding smoke from the same SSD cache;
- a fresh-process generation smoke from the SSD;
- where practical, a same-process warm inference measurement after the model is
  resident.

For each run record:

- process start to model-ready wall-clock;
- model-ready to output wall-clock;
- total wall-clock;
- peak GPU allocated and reserved memory;
- peak host RSS;
- GPU count and device model;
- input/output hashes and canonical artifact locations.

Call the first run “cold-process” rather than “cold filesystem cache” unless OS
page caches are explicitly controlled and documented.

### R3 — commit a schema-valid admission run manifest

Create `runs/admission-showo2-*/manifest.json` conforming to
`schemas/run-manifest.schema.json`, plus a run note. The manifest must reference
the smoke logs, resolved environment, input image, generated image, caption,
checkpoint/hash inventory, block registry, and SSD migration measurements.

Large artifacts may remain outside Git, but each requires:

- canonical URI/path;
- SHA-256;
- measured byte size;
- producing run ID;
- source and execution revisions.

### R4 — preserve raw smoke and environment evidence

Add or reference immutable raw stdout/stderr logs, resolved configuration,
environment lock or package snapshot, and the generated output artifact. The
current prose reports are not sufficient for independent audit of the claimed
exit codes and output hashes.

### R5 — report actual resource use

Replace the estimated model VRAM with measured peak VRAM for MMU and T2I.
Report total GPU wall-clock/GPU-hours and actual local-SSD and shared-storage
footprints.

### R6 — close or formally constrain license/provenance open items

- Resolve the exact Wan2.1 VAE source revision and license.
- Record the safety-checker license status. If the license remains unspecified,
  define it as an optional display-only dependency and provide a scientific
  evaluation path that does not require it, subject to local approval.

### R7 — freeze the repaired environment

Commit a task-scoped environment or lock description under
`configs/admission/showo2/` that includes the required torch/torchvision and
wandb pins. State clearly that the official environment script requires these
additional constraints on the audited platform.

## Re-review gate

Before resubmission:

- both task paths rerun from local SSD;
- load and inference timings are separately reported;
- measured VRAM and GPU-hours are present;
- formal admission manifest validates;
- raw artifacts are hash-addressed;
- repository validator and full tests pass;
- T210 is returned to `awaiting_review` only after the revised evidence is
  pushed.

## Second review — 2026-08-28

Reviewed branch: `origin/agent/T210-showo2-admission` at
`b05b4392815f9edaed7b4994a46cc57958139dcd`.

Decision remains `revision_needed`, with a narrow evidence-consistency repair.

The SSD migration is successful in operational terms: reported model loading
decreased from approximately 26 minutes on shared storage to approximately
8.7–9.4 seconds from `/dockerdata/t210-showo2/`. Both task paths were rerun,
measured VRAM was added, the environment was frozen, and a schema-valid
admission manifest was created.

### R8 — merge authoritative main and use the authorized run path

The task contract now authorizes date-stamped run directories through
`runs/admission-showo2-*/`. Merge the latest `origin/main` so the submitted
path is within the current contract and the repository manifest-count test is
no longer hardcoded to one run.

### R9 — reconcile all final reports with the revised evidence

Several earlier summary passages remain stale, including statements that no
admission run exists and that the Wan2.1 VAE remains unresolved.

Update the final result summary and claim check so they consistently state:

- the admission run and manifest path;
- the resolved Wan2.1 revision and Apache-2.0 license;
- the safety checker as the only remaining constrained license item;
- the measured SSD load times, VRAM, RSS, GPU-hours, and storage footprint;
- the exact status of the optional safety-checker-free scientific evaluation
  path.

Historical first-report statements may remain as dated observations, but the
final summary and claim check must not contradict the final evidence.

### R10 — distinguish provenance assets from SSD execution copies

The current manifest records shared-storage provenance URIs for model weights
but does not identify the local-SSD copies used by the reruns.

For every runtime component, record both:

- immutable provenance source URI and hash;
- local-SSD execution URI and the matching verified hash/byte size.

The SSD copy may be represented as a separate artifact or an additional
manifest field. Ensure the task remains valid if the ephemeral container is
later destroyed by preserving the migration/hash log in durable storage.

### R11 — strengthen and accurately word the no-fallback evidence

Offline environment variables and absence of `/apdcephfs_*` strings in logs
support, but do not strictly prove, that every model read came from SSD.

Either:

1. collect file-access evidence using `strace`, process open-file inspection,
   or an equivalent mechanism for one representative rerun; or
2. change the claim to “no shared-storage or network fallback was observed,”
   and list the evidence and its limitation.

Do not use “proved zero fallback” without file-access-level evidence.

### R12 — complete the missing resource fields

Add the numeric T2I peak host RSS currently referenced only as “see
`timing_stats.json`.” Summarize in the committed report:

- actual local-SSD filesystem type;
- available capacity before migration;
- exact copied bytes;
- migration wall-clock;
- T2I host RSS;
- measured durable evidence footprint.

### R13 — remotely reverify every external artifact

The local reviewer cannot access `/apdcephfs_*` or `/dockerdata/*`. Run an
artifact verification command on the remote host that checks every manifest
artifact's existence, byte size, and SHA-256. Commit the command output or a
hash-addressed durable log and reference it from the manifest/run note.

Resubmission requires the repository validator and complete test suite to pass
with zero failures.

## Final review — 2026-09-01

Reviewed branch: `origin/agent/T210-showo2-admission` at
`f869d9e1012c3312b21093695073b258ba0a1d5a`.

Decision: `accepted with recorded limitations`.

Integrated verification on `main` established:

- task graph and three run manifests pass validation;
- the complete repository suite passes 156 tests;
- the formal Show-o2 admission manifest has `status: pass`;
- understanding and generation smoke metrics report successful completion;
- remote artifact verification reports 21 checked, 21 passed, zero failed;
- storage preflight records an XFS local filesystem and an offline local cache;
- provenance assets and SSD execution copies are separately hash-addressed;
- measured loading time, VRAM, host RSS, GPU-hours, migration cost, and storage
  footprint are included in the final evidence.

Accepted limitations:

- the safety checker remains an optional display-only component with an
  unspecified upstream license; scientific evaluation must use the documented
  safety-checker-free path;
- file-access syscalls were not traced, so the evidence supports “no fallback
  observed,” not a proof that fallback was impossible;
- same-process warm inference was not measured because the audited official
  scripts are single-shot CLIs.

Accepted by: local-research-agent
