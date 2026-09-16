# T710 First Report — CoRL / Janus-Pro-1B admission and GPU smoke plan

Remote executor, 2026-09-16T13:15Z. Branch
`agent/T710-corl-admission-gpu-smoke`, HEAD at time of writing `60d57af`
(review-history bullet + status:running). This report is the mandatory gate
required before any large download or GPU execution begins; no GPU compute
and no bulk asset download has happened yet. Full detail lives in the
companion files committed alongside this report:
`configs/corl/admission/source-lock.yaml`,
`configs/corl/admission/environment-lock.md`,
`configs/corl/admission/discrepancy-lock.yaml`.

## 1. Upstream identification

- **Paper**: "Co-Reinforcement Learning for Unified Multimodal Understanding
  and Generation" (CoRL), arXiv:2505.17534v3, NeurIPS 2025. Full text
  extracted locally via `pymupdf` from the arXiv PDF (24 pages); sections
  1-4 (introduction, related work, methodology, experiments) and the
  abstract/conclusion were read in full and are the basis of this report's
  claims.
- **Code**: `https://github.com/mm-vl/ULM-R1`, pinned at commit
  `0c92629f9b307a32bb286ae3562809e941d1bb0b` on `main`. Apache-2.0 licensed
  (verified via the repo's `LICENSE` file, fetched over
  `raw.githubusercontent.com`, full Apache-2.0 text). Read in full, read-only,
  from a shallow scratch clone at `/tmp/corl_audit` (not committed to this
  branch): `README.md`, `setup.py`, `pyproject.toml`, `Install.md`,
  `corl/scripts/corl_unified.sh`, `corl/scripts/zero3.json`,
  `corl/open_r1/grpo_janus_unify.py` (220 lines, full),
  `corl/open_r1/trainer/grpo_trainer_unified.py` (850 lines, full),
  `corl/open_r1/rewards/__init__.py`, `corl/open_r1/rewards/r_base.py` (192
  lines, full), `corl/open_r1/rewards/r_t2i.py` (601 lines, full), and the
  relevant sections of `janus/models/modeling_vlm.py`.
- This is distinct from `wendell0218/Janus-Pro-R1`, which is the target of
  the separate, concurrently-running sibling task T720
  (`agent/T720-janus-pro-r1-stack-smoke`); the two were not conflated.

Not disambiguated further in this pass: `corl/open_r1/rewards/r_utils.py`
(415 lines; contains `MCQAnswerExtractor`, `safe_string_equal`,
`soft_jaccard`, `token_level_max_match_similarity`, `BertScoreWrapper`) was
referenced but not fully read line-by-line; the vendored `bert_score/` and
`open_clip/` subpackages and the `eval/`/`ttrl/` directories were not
inspected (out of scope for the bounded RL-core smoke).

## 2. Assets and pinning

| Asset | Source | Revision | License | Approx size |
|---|---|---|---|---|
| Model | `deepseek-ai/Janus-Pro-1B` (HF Hub) | `960ab33191f61342a4c60ae74d8dc356a39fafcb` | MIT | ~4.2 GB |
| Dataset | `mm-vl/x2x_rft_22k` (HF Hub) | `f52833ce01b5657294bed87f23f27d04b92838b9` | derived from COCO 2017 (CC BY 4.0); repo Apache-2.0 | ~11 GB full (22,479 rows); T710 only streams a ≤32-record micro-split |
| Aux model | `sentence-transformers/all-mpnet-base-v2` | to be pinned at download time | Apache-2.0 | ~420 MB |

All three are ungated and publicly resolvable without authentication;
confirmed via `git ls-remote`/HF Hub API metadata lookups (not yet
downloaded). Neither the model nor the aux model approaches GitHub's 100MB
single-blob push limit as a concern — these are HF Hub assets, not committed
to git; only small config/report/manifest files go into this repo.

All assets will be materialized under `/dockerdata/t710-corl/` inside the
H20-FoldUMM container, confirmed to be genuine local XFS-on-NVMe storage
(`df -hT /dockerdata` → `xfs`, `/dev/mapper/gpu-gpu_volume`, 8.9T available),
not a ceph/dop-fuse network mount. Full verification detail, including the
raw `df`/`lsblk` output, is in `configs/corl/admission/environment-lock.md`.

## 3. GPU topology and expected cost

- Target: H20-FoldUMM, 8x NVIDIA H20 96GB.
- `task_flag=gpu_model_train_split_llm218029E40DDAA49`,
  `instance_id=8b1d81c89f83d4d4019f93a9170a251d`.
- **GPU index 0 reserved exclusively for T710** (0 MiB used, 0% util at time
  of check) — GPUs 1-7 show the documented benign `train2.py` placeholder
  (325 MiB, 100% util each); not touched. This avoids collision with T720,
  which may run concurrently on a different GPU index of the same
  container.
- T710's smoke is **single-GPU**, not the official `corl_unified.sh`'s
  `torchrun --nproc_per_node=8`; batch size / grad-accum will be set
  independently and small (per the ≤32-record, ≤8-step envelope), not
  inherited from the 8-GPU launch script.
- Expected cost: Janus-Pro-1B is small (1.5B backbone) and the smoke is
  capped at 8 optimizer steps over ≤32 unique records on a single H20;
  expect well under 1 GPU-hour for the smoke itself. Environment build
  (venv, flash-attn compile) and asset download are the larger time costs,
  estimated at 0.5-1.5 GPU-attached-container-hours combined (the container
  is billed/occupied regardless of whether the GPU is actively computing).
  This sits far inside the resource envelope's 16 H20-equivalent GPU-hour
  cap.
- A dedicated venv will be built at `/root/venvs/corl` inside the container
  (plain `python3 -m venv`, since no `conda` binary is present — Python
  3.10.12 confirmed), following `Install.md`'s dependency set
  (`pyproject.toml` pins: `torch==2.6.0`, `transformers>=4.49.0`,
  `deepspeed==0.16.4`, `trl==0.18.1`, `timm==1.0.14`,
  `open_clip_torch==2.31.0`, `datasets==3.2.0`). A pre-existing,
  unrelated `/root/venvs/janus_pro` venv was found in the container with a
  broken `huggingface-hub` pin; it is not reused or repaired, to avoid
  collision with other concurrently-running tasks on this container.

## 4. SCFG discrepancy (operational note)

The operator briefing specified
`config_h20_llm2.json` for `taiji_client exec` into H20-FoldUMM. This
project's own `taiji_gpu_start/SERVER_STATUS.md` and
`taiji_gpu_start/scripts/connect_foldumm.sh` both specify
`config_h20_foldumm.json` instead, for the exact `task_flag`/`instance_id`
given. Both JSON files share an identical `Token`/`business_flag` and differ
only in an `envs` block, so this is very likely inconsequential for `exec`,
but `config_h20_foldumm.json` (the project-documented, verified-working
value) was used rather than the briefing's value. Full detail in
`configs/corl/admission/environment-lock.md`.

## 5. Paper-versus-code discrepancy table (mandatory check #5)

Full structured table in `configs/corl/admission/discrepancy-lock.yaml`
(12 entries, D1-D12). Summary of the notable findings:

- **Match** (D1, D2, D3, D4, D6 design-level, D8, D9, D10 pending numeric
  verification): core GRPO objective, stage-1 beta=0/no-KL, the Eq.5 unified
  reward composition with lambda=0.8 (hardcoded but numerically matches the
  paper's own reported value), single unified advantage normalization,
  text-image matching reward's self-referential design, format reward,
  frozen-vision/trainable-LLM-only parameter split, and image-token causal
  shift all check out against the paper's stated design.
- **default_mismatch** (D5, D7): the cycle-consistency reward's *shipped
  defaults* (self-model captioning + jaccard/bertscore + MSE) differ from
  the paper's *prose description of its own method* (BLIP + SPICE + LPIPS),
  even though LPIPS/BLIP code paths exist and are selectable via CLI flags.
  The paper's own Table 5 ablation shows LPIPS is its best-performing visual
  metric, yet the code does not default to it. Also, the OE accuracy reward
  gives soft_jaccard partial credit on non-exact matches, while the paper's
  prose describes strictly binary 0/1 accuracy rewards for both MC and OE.
  Neither is a crash or silent-corruption bug — both are legitimate,
  code-supported alternate configurations — so per the task's "repair only
  locally demonstrated correctness defects" instruction, these are left as
  `upstream_exact` behavior; only D5's image metric will additionally be
  exercised as an explicit `corrected_candidate` variant
  (`--image_cs_metrics lpips`) to match the paper's own reported-best
  config, since that is a one-flag, zero-code-change change.
- **defect** (D11): `corl_unified.sh` never sets `--model_ckpt_dir` or
  `--dataset_cache_dir`; both default to a literal `"XXX/..."` placeholder.
  Under the official default reward set, this makes
  `T2ICycleConsistencyReward.load_external_model` try to load
  `all-mpnet-base-v2` from a nonexistent `"XXX/checkpoint//all-mpnet-base-v2"`
  path — i.e., running the public launch script completely unmodified would
  crash before the first optimizer step. This is recorded as a genuine
  upstream defect (packaging omission, not a logic bug) and is the one
  place T710 applies an explicit-switch repair: supplying a real local
  `--model_ckpt_dir` under `/dockerdata/t710-corl/`. `upstream_exact` mode
  (unmodified placeholder) is expected to fail at reward-construction time;
  that failure is itself a valid, recordable outcome.
- **open_item** (D12): stage-1 batch-size accounting ("batch size of 16" in
  the paper's prose) does not cleanly reconcile against
  `corl_unified.sh`'s `per_device_train_batch_size=1 * grad_accum=4 *
  nproc_per_node=8` under any single obvious convention tried; not
  independently re-derived against TRL's `GRPOConfig` semantics in this
  pass. Does not block T710, since the smoke sets its own single-GPU batch
  size directly rather than inheriting the 8-GPU launch script's config.

## 6. Correctness tests / mandatory-check plan for the GPU smoke stage

All 8 mandatory implementation checks will get an explicit recorded outcome
in `runs/corl-admission-v1/`:

1. **`requires_grad` / optimizer membership enumeration** — dump
   `named_parameters()` with `requires_grad` flags before optimizer
   construction; cross-check against `init_trainable_parameters`'s expected
   freeze set (`vision_model`, `aligner`, `gen_vision_model`, `gen_aligner`,
   `gen_head`, `gen_embed` frozen; `language_model.*` trainable).
2. **Frozen-head gradient transmission** — run one T2I backward pass with
   `retain_graph`/hook instrumentation on `gen_head`'s output to confirm a
   nonzero gradient reaches `language_model` through the frozen `gen_head`
   forward path, even though `gen_head.weight.grad` itself stays `None`
   (frozen).
3. **Image-token next-token alignment** — construct one batch, assert
   `argmax(logits[:, t])`/loss position `t` predicts `input_ids[:, t+1]` for
   a held-out image-token index within the 576-token VQ block, matching
   `_get_per_token_logps`'s shift-by-one convention.
4. **Batched MC/OE reward dispatch** — feed a mixed batch with both
   `qa_type='MC'` and `qa_type` != `'MC'` records through
   `common_qa_accuracy_reward`, record the per-example reward values and
   confirm dispatch correctness (including the soft_jaccard partial-credit
   OE path noted in D7).
5. **Paper-vs-code discrepancy table** — done (this report +
   `discrepancy-lock.yaml`, D1-D12).
6. **Reward ranges / zero-variance groups / effective tokens / clipping** —
   record min/max/mean/std per reward function across all sampled groups in
   the smoke, flag any group with `std==0` (degenerate GRPO advantage,
   divide-by-zero risk), record effective (non-padding) token counts and
   any observed logit/probability clipping.
7. **Reference immutability when enabled** — since the official default
   `beta=0.0` never instantiates `ref_model` (making this check vacuous
   under `upstream_exact`), run a second, explicitly-labeled
   `corrected_candidate` variant with `beta>0` to actually instantiate
   `ref_model`, hash its state_dict before/after the optimizer steps, and
   confirm zero drift.
8. **Separate upstream-exact / corrected-candidate outputs** — directory
   convention: `runs/corl-admission-v1/upstream_exact/` and
   `runs/corl-admission-v1/corrected_candidate/`, never mixed.

## 7. Commands (planned, to be run from `/dockerdata/t710-corl/` inside the
   container, long-running ones via `cjob.sh`)

```bash
# venv build
python3 -m venv /root/venvs/corl
/root/venvs/corl/bin/pip install -e /dockerdata/t710-corl/ULM-R1  # after clone
# (exact flash-attn / torch build commands recorded in environment-lock.md
# once executed, with resolved versions)

# asset download (huggingface_hub, revision-pinned, streamed for the dataset)
huggingface-cli download deepseek-ai/Janus-Pro-1B --revision 960ab33191f61342a4c60ae74d8dc356a39fafcb ...
# x2x_rft_22k: streamed via datasets.load_dataset(..., streaming=True), first
# <=32 unique rows materialized to configs/corl/admission/micro-split.jsonl

# GPU smoke (single GPU, GPU index 0)
CUDA_VISIBLE_DEVICES=0 /root/venvs/corl/bin/python -m comppareto.adapters.corl.smoke ...
```

Exact final commands, with real stdout/stderr capture pointers, will be
recorded in `runs/corl-admission-v1/notes.md` once executed.

## 8. Next steps

Proceed to venv build and asset download/hash-pinning under
`/dockerdata/t710-corl/`, then the bounded GPU smoke, per this plan.
