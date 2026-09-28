# T720 result summary — Janus-Pro-R1 reusable SFT/GRPO stack smoke

## Outcome

Both required smokes (SFT optimizer smoke, GRPO rollout/backward/optimizer
smoke) ran real, single-GPU (H20 index 1, H20-FoldUMM container), and both
satisfy the task file's pass/fail gate: finite losses on every step,
authorized-only parameter changes (nonzero gradient/value change on a probe
parameter within the actually-exercised forward path, confirmed unchanged on
a frozen out-of-path probe), and reloadable checkpoints (`strict=True`
`load_state_dict`, zero missing/unexpected keys, reloaded values match the
trained values bit-for-bit).

This is a stack-audit smoke, not a CoRL reproduction, and no such claim is
made. Janus-Pro-R1 and CoRL are separate research programmes under T700's
child sequence — T720 is a sibling reusable-component audit, not a
prerequisite result for T730/T740's actual CoRL reproduction work.

## SFT smoke (4 optimizer steps)

- Base checkpoint: `deepseek-ai/Janus-Pro-7B` @ `5c3eb3fb2a3b61094328465ba61fcd4272090d67`, 14,840,868,118 bytes across 2 shards, hash-verified against the HF resolve-endpoint `X-Linked-ETag`.
- Losses (4 steps): `6.5033, 6.0479, 5.3463, 6.0796` — all finite.
- Grad norms (4 steps): `3.375, 2.844, 3.438, 3.391`.
- Parameter inventory: 286 trainable tensors / 7,032,201,216 elements (94.77% of 7,420,368,523 total), 649 frozen tensors / 388,167,307 elements, zero `requires_grad` mismatches against upstream's own `train_setup()` prefix split.
- Authorized parameter change: probe `gen_aligner.layers.0.weight` (on the exercised T2I forward path) changed, with nonzero gradient norm every step (`0.0756, 0.0743, 0.0716, 0.0871`); frozen probe `vision_model.vision_tower.pos_embed` confirmed unchanged.
- Checkpoint reload: 14,064,505,098-byte trainable-state checkpoint, `strict=True` reload, 0 missing/unexpected keys, reloaded values match trained values.
- Load 5.50s, train 4.53s, peak memory 67,653,477,376 bytes (~63.0GiB).

## GRPO smoke (4 optimizer steps)

- Policy: same `Janus-Pro-7B` checkpoint, full-parameter fine-tune (7,420,368,523 trainable elements — no freeze, matching upstream's own GRPO trainer's freeze policy, which only ever freezes a separate reference-model copy that this smoke does not instantiate).
- Reward: `OpenGVLab/InternVL2_5-8B` @ `e9e4c0dc1db56bfab10458671519b7fa3dd29463`, offline in-process, upstream's own `InternVLReward.evaluate()` unmodified.
- Losses (4 steps): `9.1787, 8.8786, 0.8441, 5.9818` — all finite; numpy-vs-torch cross-check absolute difference `< 1.1e-05` every step.
- Rewards ranged `0.0001`-`0.0534` across the 16 total generations (4 per step x 4 steps) — low absolute values are consistent with InternVL2.5-8B's yes/no-probability formulation on a small (`img_size=128`) generated image scored against a long, detailed prompt; no claim is made about reward quality/calibration, only that the reward path executes and produces finite, usable values for the advantage/loss computation.
- Authorized parameter change: probe `gen_aligner.layers.0.weight` changed, gradient norm every step `10.27, 11.13, 11.39, 7.92`.
- Checkpoint reload: 14,841,283,478-byte full-model checkpoint, `strict=True` reload, 0 missing/unexpected keys, reloaded values match trained values.
- Policy load 5.73s, reward load 2.98s, train 10.48s, peak memory 80,991,276,544 bytes (~75.4GiB) — fits comfortably within a single 96GB H20, confirming the memory-feasibility risk flagged in `reports/T720/first-report.md` section 7 did not materialize.

## Resource accounting

- GPU: H20 index 1 only (GPU0 concurrently running T710, not touched).
- GPU-hours: 2.004h against the 12h budget (reconstructed from full cjob START/END timestamp span across every container-attached step: venv builds, InternVL2.5-8B + data-shard downloads, both smokes, storage-preflight). Compute-attached time alone (sum of each smoke's own load+train seconds) is ~29.2s; the 2.004h figure is the conservative full container-session upper bound, not the narrower compute-only figure, and is the number reported in `runs/janus-pro-r1-stack-v1/metrics.json`'s `resources.gpu_hours`.
- 1 GPU used throughout (envelope allowed up to 8).

## Deviations, open items, and every fix applied

See `reports/T720/claim-check.md` (open licensing items, documented scope
reductions, environment-plan corrections) and `reports/T720/failure-ledger.md`
(every bug found and fixed, with exact error text, root cause, and
verification). None of those items change the pass/fail outcome above; all
are reported for completeness per the task's required-checks list.

## Reusable-component comparison against CoRL (T730/T740)

See `reports/T720/reuse-map.md`.
