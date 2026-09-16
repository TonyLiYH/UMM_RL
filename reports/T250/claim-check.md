# T250 — Claim check

Every load-bearing claim in this task's deliverables, with its evidence
source. Claims without a direct citation are flagged as hedged/inferred in
the source report itself (see `checkpoint-stage-evidence.md` and
`failure-ledger.md`) rather than asserted here as certain.

| Claim | Source evidence | Status |
|---|---|---|
| SenseNova-U1 `-SFT` checkpoints precede Multi-Expert RL/OPD training | `OpenSenseNova/SenseNova-U1/README.md` "Models" section, direct quote | Verified — direct official statement |
| SenseNova-U1's Stage 5 is explicit RL (Flow-GRPO + reward models) | arXiv 2605.12500, "Training Procedure" section (ar5iv HTML) | Verified — direct paper text |
| SenseNova-U1 resume restores model + optimizer + scheduler + counters | `sensenovalm/checkpoint/checkpoint_manager.py`, `try_load_internevo_ckpt` (direct source read) | Verified — code-level |
| SenseNova-U1 shared/private parameter split (1.245B/8.121B/8.186B of 17.552B) | `docs/parameter_breakdown.md` in official repo; independently reconciled 2026-09-16 against a zero-cost `torch.device("meta")` parameter count (measured: shared_backbone=9,348,413,952 + vision_shared_understanding=17,568,768 ≈ docs' shared+understanding_transformer=9,366,000,000; generation_private=8,186,358,272 matches docs' 8.186B exactly; total=17,552,340,992 matches docs' 17.552B exactly) | Verified — official published figures, now cross-validated by an independent measurement, not merely re-quoted |
| SenseNova-U1 shipped launcher needs 1 node x 8 GPUs x 80GB minimum for full-parameter fine-tuning at shipped scale | `training/README.md` hardware table | Verified — official doc; this is the shipped convenience default, not this audit's own measured floor (see next row) |
| SenseNova-U1 real single-H20 GPU memory footprint for load+forward+backward smoke peaks at ~59.5GB of 96GB | `runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json` (`gpu_memory_bytes.after_generation_backward.max_allocated=59478750720`), real GPU run on H20, 2026-09-16 | Verified — directly measured, not assumed |
| SenseNova-U1 source is pinned to a specific commit (not "main") | `git log -1`/`git remote -v` on `/dockerdata/t230-sensenova/SenseNova-U1`, the editable-installed package's own source tree: `f97964a6e54b0abf92aa2db849af4e942bb2ff08`, 2026-09-02 19:36:57 +0800, clean tree | Verified — direct git inspection, 2026-09-16 |
| SenseNova-U1-8B-MoT-SFT checkpoint is pinned to a specific HF revision, fully hashed, and executed from verified local SSD | `configs/admission/posttraining-startpoints/checkpoint-hashes.json` (214 files, 35,217,355,798 bytes, sha256 per file) + `configs/admission/posttraining-startpoints/storage-preflight.json` (`filesystem_class=local`, `filesystem_type=xfs`, `status=pass`), revision `846ff1352e3a4e900d064740cddfc163b115646f` | Verified — directly downloaded, hashed, and preflight-checked, 2026-09-16 |
| Three SenseNova-U1-SFT safetensors shards are exactly 16 bytes but are not corrupted downloads | Cross-checked shard byte content (valid minimal empty-safetensors header) against `model.safetensors.index.json`'s `weight_map`: zero tensors are assigned to shards 00002/00003/00004 of 16 | Verified — direct byte-level + index-file cross-check, 2026-09-16 |
| The SFT checkpoint itself (not T230's final-MoT checkpoint) loads on a real H20 and both a pure-understanding and a pure-generation forward+backward pass succeed, with gradients confined to the correct owned parameter group in each case | `runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json`: `understanding_loss=9.830007553100586` (grad only in `shared_backbone`), `generation_loss=5.4360198974609375` (grad only in `generation_private`), `status=pass` | Verified — real GPU execution, 2026-09-16 |
| Constructing an AdamW optimizer + CosineAnnealingLR scheduler + resume-metadata dict over the loaded SFT checkpoint does not mutate its weights | `runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json`: `fp_before_opt == fp_after_opt` (both `79efdf9257c805710d0ccd92f4fed56512f6fd5e05db5cbba6965a4116b646b9`), `weights_unmutated_by_optimizer_construction=true` | Verified — sha256 fingerprint match, 2026-09-16 |
| `NEOChatModel.forward()`/`batch_chat()` are `NotImplementedError` stubs, and mixed understanding+generation token batches are unsupported in this pinned commit | Direct read of `modeling_neo_chat.py`/`modeling_qwen3.py` at the pinned commit: `forward()`/`batch_chat()` raise `NotImplementedError` as their first line; `Qwen3Attention.forward()`/`Qwen3DecoderLayer.forward()` raise `NotImplementedError("...issue #207...")` for mixed batches | Verified — direct source read, 2026-09-16 |
| A `generation_private`-only T270 training subspace on SenseNova-U1-SFT plausibly fits 2xH20 (192GB) with optimizer-state sharding; full-parameter fine-tuning of both `shared_backbone`+`generation_private` does not fit 2xH20 without further sharding/offload | Derived from measured meta-device parameter counts (`shared_backbone=9,348,413,952`, `generation_private=8,186,358,272`) x bf16-weight/fp32-AdamW-state arithmetic; reproduced in `reports/T250/training-interface-audit.md` | **Derived/hedged** — arithmetic projection from real measured parameter counts, not itself a live 2-GPU run (which this task's <=2-GPU/<=4-GPU-hour envelope permits but does not require for a non-training measurement) |
| Show-o2 checkpoint is Apache-2.0, checkpoint hash as recorded | T210's accepted admission (`configs/admission/showo2/`, `reports/T210/environment-checkpoint-smoke.md`) | Verified — reused prior accepted evidence, not re-derived |
| Show-o2 both task paths execute with coherent output on H20 | `reports/T210/task-path-smoke.md` (real GPU smoke, T210) | Verified — reused prior accepted evidence; NOT re-run in this task |
| Show-o2 released checkpoint = Stage-2 SFT-equivalent | Pipeline-structure inference (only 2 stages exist, no RL stage) + T210 functional smoke | **Hedged** — no explicit checkpoint-card statement; stated as moderate confidence |
| Show-o2 resume restores weights only, not optimizer/scheduler state | `train_stage_one.py` resume block, lines ~264-320 (direct source read) | Verified — code-level |
| UniDDT has no recorded license (no LICENSE file, no HF license tag) | Direct `curl` 404 on `LICENSE`/`LICENSE.md`; HF Hub API response inspected | Verified — direct HTTP/API check |
| UniDDT's Duality post-training is not preference/RL | `MCG-NJU/UniDDT/README.md` "Architecture and training" section + arXiv 2606.16255 abstract — no reward-model/RLHF/DPO terminology found | Verified by absence — no contradicting evidence found in either official source |
| UniDDT released checkpoint's exact stage (post-Joint vs. post-Duality) | README prose vs. arXiv figure caption — inconsistent | **Hedged / unresolved** — recorded as unknown, not guessed |
| UniDDT resume is full-state via PyTorch Lightning `ckpt_path` | `main.py` CLI wiring + PyTorch Lightning's documented default `Trainer.fit(ckpt_path=...)` behavior | **Partially verified** — framework-level default confirmed; UniDDT's own trainer-subclass overrides not independently checked |
| This task consumed real, bounded GPU time (not zero) in its 2026-09-16 revision, and no optimizer step / persistent update occurred | `runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json` + cjob log timestamps (`17:38:36`-`17:38:59` CST, 23s wall clock; `total_seconds=18.261597156524658` measured internally); reported as ~0.01 GPU-hours against the 4-hour cap; `optimizer_steps=0` in `metrics.json` | Verified — self-evident from the commands actually run and their logged output |
| Repository test suite passes on this branch | `.venv/bin/python -m pytest -q` → `156 passed` | Verified — actually executed, see final validator output reported to the orchestrator |
| Repository-state check passes | `.venv/bin/python -m comppareto.repo_state.cli --root .` → `task_tree=pass tasks=32`, `run_manifests=pass manifests=<n>`, `research_state=pass` | Verified — actually executed |

## Claims explicitly NOT made

- This task does **not** claim UniDDT is technically inferior to the other
  two candidates — its architecture, resume support, and non-RL final stage
  are all comparably strong. It is excluded solely on the license gate, which
  is stated plainly rather than conflated with a capability judgment.
- This task does **not** claim SenseNova-U1's shipped launcher fits this
  audit's 2-GPU envelope — it explicitly does not, as shipped for
  full-parameter fine-tuning at the shipped sequence/image scale. The
  2026-09-16 revision's own measured single-H20 smoke and derived
  `generation_private`-only 2xH20 topology are a separate, real measurement
  for a narrower subspace, not a claim that the shipped launcher's default
  itself fits 2 GPUs.
- This task does **not** claim training readiness has been demonstrated by a
  live training run for any candidate — no persistent parameter update,
  optimizer step, or training-loop execution occurred in this task for any
  candidate. The 2026-09-16 revision DID execute a real forward+backward
  smoke (loss computation and gradient computation only, no `.step()`) on
  the pinned SenseNova-U1-SFT checkpoint; this is explicitly evidence of
  load/forward/backward-loss functionality, not of training readiness in
  the optimizer-state/resume/dataset-scale sense.
