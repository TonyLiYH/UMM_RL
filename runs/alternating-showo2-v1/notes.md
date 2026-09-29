# Run Notes: alternating-showo2-v1 (T216)

## Run summary

Task: T216 — Show-o2 compute-matched alternating-protocol diagnostic
Run kind: formal, K=1
Date: 2026-09-28
GPU: H20-FoldUMM, index 3

## Subspaces

- Shared: `model.layers.27` (last Qwen2.5-1.5B decoder layer)
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
