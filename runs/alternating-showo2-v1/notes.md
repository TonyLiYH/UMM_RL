# Run Notes: alternating-showo2-v1 (T216)

## Run summary

Task: T216 — Show-o2 compute-matched alternating-protocol diagnostic
Run kind: formal, K=1
Date: 2026-09-28
GPU: H20-FoldUMM, index 3

## Subspaces (prefixes as they appear in the real state dict)

- Shared: `showo.model.layers.27` (last Qwen2.5-1.5B decoder layer, 46,797,824 params)
- Understanding-private: `und_trans.layers.7` (last understanding transformer layer)
- Generation-private: `diffusion_head_a.9` (last diffusion head block)

## Protocol schedule

- P0: simultaneous baseline (shared step on negotiated direction; private step in same pass)
- P1: private-only control (K=1 private steps, no shared update)
- P2: shared-then-private (shared step on negotiated direction, then K=1 private steps)
- P3: virtual private-then-shared commit (private rollback, virtual commit on negotiated direction)

## Negotiators

raw_sum, normalized_sum, pcgrad (both orders), mgda (closed-form two-task)

## Step-scale grid

[5e-6, 5e-5, 5e-4] — anchored on official lr=5e-5

## T215 failure modes avoided

- MMU-NAN: no `create_graph=True` anywhere in this run
- ROLLBACK-RNG: explicit `torch.get_rng_state()` snapshot/restore, not re-seeding
- T2I-FDMISS: finite-difference checking not used; direct loss evaluation only

## Post-run verification

- persistent_updates == 0: verified by bit-exact param assert after all protocol rows
- snapshot_restore.failed: reported in metrics.json

## Weight provenance (recorded 2026-09-29, [CORRECTION] to artifact entry)

The run read the checkpoint from the execution container's local SSD copy:

  /dockerdata/t210-showo2/hf_cache/hub/models--showlab--show-o2-1.5B/blobs/a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c

That path exists only inside the H20-FoldUMM container; it is not mounted on the
reporting/review host, so it cannot be hash-verified outside the container. The
manifest's `artifacts[0]` entry therefore records the byte-identical copy on shared
storage (same 5,661,862,314 bytes, same sha256 `a596cbc305c1…`, re-hashed on the
reporting host 2026-09-29):

  /apdcephfs_cq7/share_1447896/yihangli/models/pretrained/hf_cache/hub/models--showlab--show-o2-1.5B/blobs/a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c

This matches the convention used by `runs/admission-showo2-2026-08-28/manifest.json`,
whose primary entry is `showo2-1.5b-checkpoint` / `model_weights` at that CQ7 path.
The SSD execution copy was verified as `pass` (sha256 + size) inside the container
immediately after the sweep; see `reports/T216/failure-ledger.md` [CORRECTION].
No measurement changed: `raw_rows.json` and `metrics.json` are untouched.
