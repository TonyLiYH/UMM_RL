# T250 — First report (before GPU work or large downloads)

Per the task file's "First report" requirement, this is published before any GPU
command or new large-asset download. **No GPU work and no large download are
currently planned for this task** (see "Planned execution" below); this report
still records the required fields so the plan is auditable before anything
changes.

## Candidate IDs

| Candidate | Official ID(s) audited | Source |
|---|---|---|
| A | `sensenova/SenseNova-U1-8B-MoT-SFT` (HF), `OpenSenseNova/SenseNova-U1` (GitHub, training code under `training/`) | arXiv 2605.12500 |
| B | `showlab/show-o2-1.5B` (HF), `showlab/Show-o` subdir `show-o2/` (GitHub) — **reusing T210's accepted admission** | arXiv 2506.15564; T210 accepted 2026-09-01 |
| C | `MCG-NJU/UniDDT` (HF and GitHub, both `vlm_uniddt_512.ckpt` / `vlm_uniddt_1024.ckpt`) | arXiv 2606.16255 |

`SenseNova-U1.5-*` is explicitly out of scope, consistent with T230's exclusion
rule ("U1.5 remains excluded until its announced training pipeline is public"):
the `OpenSenseNova/SenseNova-U1` README's own changelog (`[2026.08.20]`) states
U1.5's "full training pipeline, from SFT and RL to MOPD" is still only "being
prepared" for open-source release, so no reproducible U1.5 training path exists
yet. Only the original `SenseNova-U1-8B-MoT-SFT` checkpoint named in the task
frontmatter is audited.

## Expected stage evidence (what I will look for, and where)

- **SenseNova-U1**: the official GitHub README states explicitly (`README.md`,
  "Models" section): *"SFT models ... are trained via Understanding Warmup,
  Generation Pre-training, Unified Mid-training, and Unified SFT, with final
  models obtained after Multi-Expert RL and OPD training."* The arXiv paper's
  Table 2 / Section 3.4 names five stages (Stage 1 Understanding Warmup, Stage 2
  Generation Pre-Training (3 phases), Stage 3 Unified Mid-Training, Stage 4
  Unified Supervised Fine-Tuning, Stage 5 "Post Training for T2I Generation" —
  explicit Flow-GRPO reinforcement learning with OCR/style/aesthetic
  (HPSv3-preference) reward models). This is direct, official, stage-named,
  non-filename evidence that `-SFT` = post-Stage-4, pre-Stage-5 (pre-RL).
- **Show-o2**: reuse T210's accepted admission for license/checkpoint identity.
  T210 did not establish an evidence-backed stage label for the released
  checkpoint (open item carried into this task). The official `show-o2/README.md`
  documents exactly two training stages (Stage-1 pretraining-style, Stage-2
  LLaVA-OneVision/DenseFusion-style instruction-tuning) plus an optional
  downstream mixed-modality fine-tune; no RL/preference stage exists anywhere in
  the official Show-o2 code or docs. I will state the released checkpoint's
  stage as "SFT-equivalent (Stage-2), inferred from pipeline structure and T210's
  own functional smoke evidence, not from an explicit stage statement on the
  model card" — an explicitly hedged label, not a filename-based guess.
- **UniDDT**: the GitHub README (`README.md`, "Architecture and training")
  documents three stages (Warmup, Joint training, Duality post-training) and
  states the released checkpoint's config nulls out the warmup/joint
  initialization paths "because the released checkpoint already carries those
  weights" — consistent with, but not an explicit confirmation of, having
  completed duality post-training. Duality post-training is caption-likelihood
  maximization over the diffusion decoder's own intermediate sampling states
  (self-distillation), not preference/RL — no reward model, no RLHF/DPO
  terminology appears anywhere in the README or the arXiv abstract. I will
  record this stage label as hedged ("most likely post-duality, not explicitly
  confirmed; not a preference/RL stage either way").

## Resume-interface inspection plan (static, source-level, no GPU)

- **SenseNova-U1**: read `training/sensenovalm/checkpoint/checkpoint_manager.py`
  in full (already partially read: `try_load_internevo_ckpt` explicitly calls
  `load_model_checkpoint`, `load_optimizer_checkpoint`, `load_scheduler`, and
  `load_context`/`TrainState`, gated by `ckpt.load_optimizer` and
  `content=("model","sampler","optimizer")`). Confirm no silent no-op branches
  before writing the final claim.
- **Show-o2**: read `train_stage_one.py`'s resume block (already read: lines
  ~264-320 restore **model weights only** via `model.load_state_dict(...)` and
  reconstruct `global_step`/`first_epoch` from the `checkpoint-<step>` directory
  name; a **fresh** `AdamW` optimizer and a **fresh** LR scheduler are created
  after resume — optimizer momentum/variance and scheduler warmup state are
  **not** restored). This is a materially weaker resume than SenseNova-U1's.
- **UniDDT**: `main.py` uses `lightning.pytorch.cli.LightningCLI` with a
  `ckpt_path` config key (`is_resume = self._get(self.config_init, "ckpt_path",
  ...)`); PyTorch Lightning's own `Trainer.fit(ckpt_path=...)` contract restores
  model, optimizer, LR scheduler, and epoch/global-step/loop state by default.
  I have not yet read UniDDT's own `JointTrainer`/`DualTrainer` classes for any
  override that would break this default — will note as "framework-level
  guarantee, not directly verified in UniDDT's own trainer subclasses" unless a
  quick read resolves it.

## Reusable assets from prior admissions

- **Show-o2**: reuse T210's accepted checkpoint identity, hash, license
  (Apache-2.0), and `configs/admission/showo2/parameter-block-registry.yaml`
  wholesale — no need to re-derive parameter counts. T210's GPU smoke
  (`reports/T210/task-path-smoke.md`) already demonstrated both task paths
  (`inference_mmu.py`, `inference_t2i.py`) run to completion on H20 hardware
  with coherent output — the strongest empirical evidence available for any of
  the three candidates. T210 did **not** exercise the training/resume path
  (inference only), so this task independently audits Show-o2's training code
  statically (see above).
- **SenseNova-U1**: `docs/parameter_breakdown.md` in the official repo already
  publishes an exact, reproducible shared/private parameter split for
  `SenseNova-U1-8B-MoT` (`shared` 1.245B/7.09%, `understanding_transformer`
  8.121B/46.27%, `generation_transformer` 8.186B/46.64%, total 17.552B) via a
  documented inspection script — reusable directly as the parameter-block
  registry basis; no local computation needed.
- **UniDDT**: no prior admission evidence exists in this repo (T220 is `ready`,
  not `accepted`, and is being worked by a separate, concurrent agent in the
  main checkout — not referenced or depended on here).

## Planned downloads

**None planned.** No model weights will be downloaded for this task. All
evidence above comes from official public documentation, model-card metadata,
and official source files fetched read-only over HTTPS (GitHub raw content,
Hugging Face Hub API, arXiv HTML). If static evidence proves insufficient for
any required audit dimension, the plan is to fall back to a **weights-free**
static source read (cloning/reading code, not downloading checkpoint tensors)
before considering any GPU smoke — and only within the stated envelope (≤2 H20
GPUs, ≤4 GPU-hours, local SSD only) if truly necessary.

## Memory and runtime estimate

Zero GPU memory and zero GPU wall-clock are budgeted unless a static-evidence
gap forces a minimal load/resume-interface smoke (execution stage 6). If that
becomes necessary: Show-o2 (1.5B, ~22GB fp32 checkpoint) fits comfortably on a
single H20 (96GB) for a load/forward smoke, matching T210's own measured
footprint (~13.9-14.9GB allocated). SenseNova-U1 (17.6B total, dense+MoT) would
need materially more headroom for anything beyond a weights-only load smoke;
its own shipped launcher defaults to 8×H20 80GB+ for full training, which is
outside this task's 2-GPU envelope (irrelevant to this audit, since no training
is authorized, but relevant to record for T270 planning). UniDDT's released
checkpoint is 22.6GB fp32 EMA; a load-only smoke would likely fit on one H20.
CPU wall-clock for the source/documentation audit itself: already substantially
complete via read-only HTTPS fetches; remaining work is writing the required
deliverable files and running the local validation/test commands.

## Exact commands

Documentation/source audit (already executed, read-only, no repository writes):

```bash
curl -sS "https://raw.githubusercontent.com/OpenSenseNova/SenseNova-U1/main/README.md"
curl -sS "https://raw.githubusercontent.com/OpenSenseNova/SenseNova-U1/main/training/README.md"
curl -sS "https://raw.githubusercontent.com/OpenSenseNova/SenseNova-U1/main/training/sensenovalm/checkpoint/checkpoint_manager.py"
curl -sS "https://raw.githubusercontent.com/OpenSenseNova/SenseNova-U1/main/docs/parameter_breakdown.md"
curl -sS -L "https://ar5iv.labs.arxiv.org/html/2605.12500"
curl -sS "https://raw.githubusercontent.com/showlab/Show-o/main/show-o2/README.md"
curl -sS "https://raw.githubusercontent.com/showlab/Show-o/main/show-o2/train_stage_one.py"
curl -sS "https://raw.githubusercontent.com/showlab/Show-o/main/show-o2/train_showo2_1.5b_stage1.sh"
curl -sS "https://raw.githubusercontent.com/MCG-NJU/UniDDT/main/README.md"
curl -sS "https://raw.githubusercontent.com/MCG-NJU/UniDDT/main/main.py"
curl -sS -L "https://ar5iv.labs.arxiv.org/html/2606.16255"
curl -sS "https://huggingface.co/api/models/sensenova/SenseNova-U1-8B-MoT-SFT"
curl -sS "https://huggingface.co/api/models/MCG-NJU/UniDDT"
```

Repository-side commands to run before submission (per
`tasks/contracts/T250.acceptance.yaml`):

```bash
.venv/bin/python scripts/model_storage_preflight.py --path <local-ssd-path> \
  --minimum-free-bytes 5000000000 \
  --output configs/admission/posttraining-startpoints/storage-preflight.json
.venv/bin/python -m comppareto.repo_state.cli --root .
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
.venv/bin/python scripts/verify_manifest_artifacts.py \
  --manifest runs/admission-posttraining-startpoints-v1/manifest.json \
  --output /tmp/T250-artifact-verification.json
git diff --check origin/main...HEAD
bash scripts/validate_task_submission.sh T250
```

## Open items carried into later reports

1. SenseNova-U1: the exact expansion of "OPD" (grouped with RL as a
   post-SFT/post-RL stage in the README's changelog) is not spelled out in any
   text I could fetch; treated as "a further preference-related post-training
   step, name uncertain" rather than guessed.
2. UniDDT: no `LICENSE`/`LICENSE.md` file exists in the GitHub repository root
   (confirmed 404 on both), and the Hugging Face model API response carries no
   `license:` tag — this is a real, code/API-confirmed gap against the decision
   rule's licensing requirement, not an assumption.
3. UniDDT: whether the single released checkpoint has completed "duality
   post-training" is not explicitly stated either way in the README; the
   arXiv abstract's Figure 1 caption names only "Warmup" and "Joint training"
   (the README's third stage, "Duality post-training", is described in prose
   elsewhere in the README but is not named in the figure caption I could
   fetch) — recorded as a minor evidentiary inconsistency between the two
   official sources, not resolved by assuming either reading.

## Revision addendum — 2026-09-16 (real GPU work actually performed)

2026-09-16 local review found the static-only plan above insufficient for
SenseNova-U1-8B-MoT-SFT specifically (see the task file's "Local review
requirements") and required real GPU evidence. This section records exactly
what was executed, superseding the "no GPU work is currently planned" framing
above for this one candidate; Show-o2 and UniDDT's static audits are
unaffected and unchanged.

**Source pin.** `github.com/OpenSenseNova/SenseNova-U1@f97964a6e54b0abf92aa2db849af4e942bb2ff08`
(2026-09-02 19:36:57 +0800) — the commit actually checked out for the
editable-installed `sensenova_u1` package imported by the smoke script at
runtime (not a separately-cloned read-only checkout), clean tree, remote
confirmed.

**Checkpoint download and hashing.** `sensenova/SenseNova-U1-8B-MoT-SFT` @ HF
revision `846ff1352e3a4e900d064740cddfc163b115646f` downloaded to
container-local SSD (`/dockerdata/t250-sensenova-sft/checkpoint`,
`filesystem_class=local`/`xfs`, confirmed by
`configs/admission/posttraining-startpoints/storage-preflight.json`, run with
`HF_HOME`/`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set). 214 files,
35,217,355,798 bytes (~33GB) total. Every file's sha256 + byte size recorded
in `configs/admission/posttraining-startpoints/checkpoint-hashes.json`. Three
`model.safetensors` shards (00002/00003/00004 of 16) are exactly 16 bytes;
cross-checked against `model.safetensors.index.json`'s `weight_map` and
confirmed zero tensors are assigned to those three shard filenames — a
benign upstream sharding artifact, not corruption.

**GPU smoke (single H20, GPU0, via `cjob`).** Loaded
`NEOChatModel.from_pretrained(CKPT, config=config, torch_dtype=torch.bfloat16)`
directly from the local checkpoint path (`load_seconds=4.914`). Ran one
pure-understanding forward+backward smoke
(`understanding_loss=9.830007553100586`, gradients confined to the
`shared_backbone` group) and one pure-generation forward+backward smoke
(`generation_loss=5.4360198974609375`, gradients confined to the
`generation_private` group), using only officially-implemented (non-stubbed)
entry points — full detail and the architectural finding that motivated this
exact split (`NotImplementedError` on the top-level `forward()` and on any
mixed-token batch) are in `reports/T250/training-interface-audit.md`.
Constructed a real `AdamW` optimizer (3 named parameter groups) +
`CosineAnnealingLR` scheduler + resume-metadata dict without ever calling
`.step()`; a sha256 fingerprint over sampled parameters was identical before
and after construction, proving no weight mutation. Peak GPU memory across
the whole session: `max_allocated=59,478,750,720` bytes (~59.5GB) on one
96GB H20. Exact per-group parameter counts obtained via a zero-cost
`torch.device("meta")` model construction. Full numeric evidence:
`runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json`.

**Wall-clock / GPU-hours.** cjob wall time for the smoke itself: 23 seconds
(`17:38:36`-`17:38:59` CST, `[cjob] START`/`END` timestamps), of which the
Python process measured `total_seconds=18.261597156524658` internally. No
other GPU-attached step was run for this task (checkpoint download/hashing
were CPU/network/disk-only). Reported conservatively as **0.01 GPU-hours**
(rounding the measured wall time up), against the 4-GPU-hour cap — see
`runs/admission-posttraining-startpoints-v1/metrics.json`.

**Exact commands** (in addition to the ones already listed above):

```bash
# storage preflight against the real container-local SSD checkpoint path
export PYTHONPATH=<worktree>/src
export HF_HOME=/dockerdata/t250-sensenova-sft/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/model_storage_preflight.py \
  --path /dockerdata/t250-sensenova-sft/checkpoint \
  --minimum-free-bytes 40000000000 \
  --output configs/admission/posttraining-startpoints/storage-preflight.json

# checkpoint download (HF snapshot_download, star_proxy, offline env vars set)
# checkpoint hashing (sha256 + size for all 214 files)
# GPU smoke (see reports/T250/training-interface-audit.md for the script's
# forward/backward/optimizer-construction logic), launched via:
CUDA_VISIBLE_DEVICES=0 /dockerdata/t230-sensenova/venv/bin/python t250-smoke.py
```

No optimizer `.step()`, no persistent parameter update, and no dataset-scale
run occurred at any point in this addendum.
