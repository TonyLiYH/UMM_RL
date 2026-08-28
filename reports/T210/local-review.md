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
