# Notes — `runs/janus-pro-r1-stack-v1`

## What this run is

A single formal run (`run_kind: formal`, `run_id: janus-pro-r1-stack-v1`)
covering the two GPU smokes required by T720: one SFT optimizer smoke (4
steps) and one GRPO rollout/reward/loss/backward/optimizer smoke (4 steps),
both against `deepseek-ai/Janus-Pro-7B`, executed on H20 GPU index 1 of the
H20-FoldUMM Taiji container. Full narrative and numbers are in
`reports/T720/result-summary.md`; this file documents the run's own
artifact/evidence layout and how to reproduce/re-check it.

## Layout

- `manifest.json` — schema-valid run manifest (`schemas/run-manifest.schema.json`), listing every externally-verifiable artifact this run consumed/produced (model weight shards, the one downloaded data shard, both smoke summary evidence files) by `canonical_uri`+sha256+bytes, all pointing at CQ7-canonical or in-repo paths reachable from a plain (non-container) shell — see the module docstring in `src/comppareto/adapters/janus_pro_r1/run_manifest_build.py` for why `/dockerdata` paths are deliberately excluded from this list.
- `metrics.json` — the acceptance-contract-checked metrics (`sft.loss_finite`, `sft.optimizer_steps`, `sft.checkpoint_reload_pass`, `grpo.*` equivalents, `resources.gpu_hours`), plus the full per-step evidence (losses, grad norms, rewards, advantages) for both smokes, parameter inventories, memory peaks, and the GPU-hours accounting methodology.
- `evidence/sft-smoke-summary.json`, `evidence/grpo-smoke-summary.json` — byte-exact copies of the smoke scripts' own `--summary-json` output (the un-truncated files, including the per-step `steps` array that each script's stdout print deliberately omits), copied out of container-local `/dockerdata` into the repo so they are both git-tracked and independently re-hashable.

## How each smoke was run

Both ran inside the Taiji container (`script -qec "taiji_client exec ..." /dev/null` fake-PTY wrapping, `CUDA_VISIBLE_DEVICES=1`), each inside its own separately pinned venv:

```
# SFT (environment-sft-lock.md)
PYTHONPATH=.../src /dockerdata/t720-janus-pro-r1/venvs/sft/bin/python \
  -m comppareto.adapters.janus_pro_r1.sft_smoke \
  --n-steps 4 --batch-size 2 --lr 1e-5 \
  --summary-json /dockerdata/t720-janus-pro-r1/runs/sft-smoke/summary.json

# GRPO (environment-rl-lock.md)
PYTHONPATH=.../src /dockerdata/t720-janus-pro-r1/venvs/rl/bin/python \
  -m comppareto.adapters.janus_pro_r1.grpo_smoke \
  --n-steps 4 --num-generations 4 --lr 1e-6 \
  --summary-json /dockerdata/t720-janus-pro-r1/runs/grpo-smoke/summary.json
```

## Why `/dockerdata` artifacts are not in `manifest.json`'s hashed list

`/dockerdata` is the container's local (non-ceph) SSD — required by the
frozen protocol's "execute from verified local SSD" clause, and confirmed via
`mount`/`df` to be `xfs` on `/dev/mapper/gpu-gpu_volume` (see
`configs/janus-pro-r1/admission/storage-preflight.json`). It is not mounted
in the plain CQ9 shell that runs `scripts/validate_task_submission.sh` /
`scripts/verify_manifest_artifacts.py`. Since that verification script
re-hashes every manifest artifact's `canonical_uri` from wherever it is
invoked, any artifact pointing at `/dockerdata` would be unconditionally
reported `"missing"`. Two categories of `/dockerdata` content are therefore
handled differently:

1. **Model checkpoints (Janus-Pro-7B, InternVL2_5-8B) and the data shard** —
   these all had (or were given) CQ7-canonical or in-repo copies, which were
   independently re-hashed this task and confirmed byte-identical to the
   container-execution copies. `manifest.json` references the CQ7-canonical
   copies.
2. **The two disposable smoke checkpoint `.pt` files**
   (`sft_smoke_trainable_state.pt`, 14,064,505,098 bytes;
   `grpo_smoke_state.pt`, 14,841,283,478 bytes) — these are training
   *outputs* specific to this smoke run, not source assets, and exist only on
   `/dockerdata`. They are intentionally excluded from `manifest.json`'s
   artifact list. Their sha256/reload-check facts are still real: both were
   produced and verified (via `strict=True` `load_state_dict` +
   trained-vs-reloaded value equality) by the smoke scripts themselves,
   inside the container, immediately after save — reported in
   `metrics.json`'s `sft.checkpoint_roundtrip`/`grpo.checkpoint_roundtrip`,
   not independently re-hashable outside the container. This is a documented
   scope decision (see `reports/T720/claim-check.md` section 5), not a
   missing check.

## Reproducing the admission/artifact checks from a plain CQ9 shell (no container needed)

```bash
cd <repo-root>
.venv/bin/python -m comppareto.adapters.janus_pro_r1.run_manifest_build \
  --repo-root . \
  --janus-checkpoint-dir /apdcephfs_cq7/share_1447896/yihangli/models/pretrained/Janus-Pro-7B \
  --internvl-checkpoint-dir /apdcephfs_cq7/share_1447896/yihangli/models/pretrained/InternVL2_5-8B \
  --data-shard /apdcephfs_cq7/share_1447896/yihangli/data/janus-pro-r1/train-0000-of-0524.parquet \
  --admission-dir configs/janus-pro-r1/admission \
  --evidence-dir runs/janus-pro-r1-stack-v1/evidence \
  --source-revision <repo HEAD sha at time of admission> \
  --status pass \
  --output runs/janus-pro-r1-stack-v1/manifest.json

.venv/bin/python scripts/verify_manifest_artifacts.py \
  --manifest runs/janus-pro-r1-stack-v1/manifest.json \
  --output configs/janus-pro-r1/admission/artifact-verification.json
```

Both commands are CPU-only (no torch, no GPU) and run directly from the
repo's own `.venv` — no Taiji container round-trip is required for this part
of the pipeline, only for the two GPU smokes themselves.
