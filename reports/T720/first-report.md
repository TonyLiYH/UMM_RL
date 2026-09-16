# T720 first report — asset inventory, environment split, plan

Status at time of writing: CPU-side audit complete, vendoring complete,
admission locks written. No large downloads (InternVL2.5-8B) or GPU
execution have happened yet. This report is the explicit pre-execution gate
required by the task file's "First report" section and is being committed
before any of that work proceeds.

## 1. Asset inventory (full detail in `configs/janus-pro-r1/admission/source-lock.yaml`)

| Asset | Source | Pinned revision | License | Size |
|---|---|---|---|---|
| Code | `github.com/wendell0218/Janus-Pro-R1` | `0e40b3aa291cb15770f69affc956602c217490af` | **none declared** (open item) | vendored 134 files / 3.3MB |
| Base checkpoint | `deepseek-ai/Janus-Pro-7B` | `5c3eb3fb2a3b61094328465ba61fcd4272090d67` | MIT (code) + DeepSeek Model License (weights) | 14,840,868,118 bytes (2 shards), already present + hash-verified on H20-FoldUMM local cache |
| Released R1 checkpoint (reference only, not trained from) | `midbee/Janus-Pro-R1-7B` | `ed62b8093c6babbd0a773d27dc4f1234bb0652fa` | apache-2.0 | not downloaded |
| SFT/RL bounded data | in-repo `janus-sft/data/t2i_examples/`, `janus-rl/data/t2i_examples.txt` | pinned with the code commit above | repo default (no LICENSE) | ~1.7MB, already vendored |
| Full released dataset (admission check only) | `midbee/Janus-Pro-R1-Data` | `dc4c00a8a175820e4d917b5cf540fc8fc96dc4e1` | none declared (open item) | 1 shard of 524 downloaded+hash-verified (875,432,539 bytes), full dataset 344,369,538,709 bytes not downloaded |
| Reward model | `OpenGVLab/InternVL2_5-8B` | `e9e4c0dc1db56bfab10458671519b7fa3dd29463` | MIT + Apache-2.0 (backbone) | 16,155,492,897 bytes, **not yet downloaded — this report gates that download** |

Two open licensing items (Janus-Pro-R1 code has no LICENSE file;
Janus-Pro-R1-Data has no license tag) are carried forward verbatim into
`reports/T720/claim-check.md` and are not resolved by this report — they
are facts about the upstream project, not something T720 can fix.

## 2. Environment split

Two separately pinned venvs, both built fresh under
`/dockerdata/t720-janus-pro-r1/venvs/{sft,rl}` on H20-FoldUMM local SSD
(the pre-existing `/root/venvs/janus_pro` is broken — `huggingface-hub`
version conflict breaks `import transformers` — and is not reused):

- **SFT** (`configs/janus-pro-r1/admission/environment-sft-lock.md`):
  upstream's `requirements-sft.txt` pins neither `torch` nor
  `transformers`; T720 pins `torch==2.5.1+cu121` /
  `transformers==4.50.0` (matching the RL side) plus the small set of
  packages actually imported by the exercised code path (`einops`, `timm`,
  `easydict`, `attrdict`, `pillow`, `sentencepiece`, `pyyaml`,
  `datasets==2.16.1`). `deepspeed`/`albumentations`/`basicsr`/etc. from the
  broad upstream requirements file are skipped — not imported anywhere on
  the model/trainer path (verified by source-level import scan).
- **RL** (`configs/janus-pro-r1/admission/environment-rl-lock.md`):
  upstream's `requirements-rl.txt` is fully version-pinned; T720 installs
  the subset actually imported by `grpo_t2i.py` / `internvl_img.py` /
  `mydataset.py` and the vendored `internvl/` package
  (`torch==2.5.1+cu121`, `transformers==4.50.0`, `tokenizers==0.21.2`,
  `einops==0.8.1`, `numpy==2.3.1`, `sentencepiece==0.2.0`,
  `Pillow==11.3.0`, `timm==1.0.16`, `decord==0.6.0`, `flash_attn==2.7.4.post1`).
  `apex` and `petrel_client` are skipped by design: both are guarded
  `try/except ImportError` in the vendored source with working pure-PyTorch
  / local-PIL fallbacks (verified by direct `grep` of every hit — all
  inside guard blocks, none of them bare imports). `fastchat` is skipped:
  the vendored `internvl/conversation.py` is a self-contained copy with
  only a courtesy comment mentioning fastchat, no actual import.
  `vllm`/`deepspeed`/`accelerate` multi-process launch is skipped: T720's
  GRPO smoke is a custom single-GPU loop calling upstream's
  `InternVLReward.evaluate()` and the policy's `.generate()` directly, not
  `trl.GRPOTrainer.train()`.

## 3. Data format

- SFT `t2i_examples` label files: one JSON object per line —
  `{promptid, prompt, data: [{id, img_path, reason, prob, state}, ...]}`.
  `task_type` is selected per-sample by which labeled sub-task the trainer
  is drawing from (0 = text-to-image generation targets, 1 =
  image-text-consistency self-evaluation targets built from `state`/`prob`,
  2 = regeneration/reflection pairs). This is upstream's own format,
  unmodified.
- GRPO prompt source `t2i_examples.txt`: one free-text prompt per line,
  consumed by upstream's own `MyT2IDataset` unmodified.

## 4. Reward-model plan

Offline, in-process `InternVLReward` (upstream class, unmodified logic)
loading `OpenGVLab/InternVL2_5-8B@e9e4c0dc1db56bfab10458671519b7fa3dd29463`
from a local verified path (env-var driven, replacing the author's
hardcoded internal cluster path). Prompt template and yes/no-logit scoring
formula are upstream's exactly:
`'<image>\n Does this image match the description "{prompt}", please
directly respond with yes or no.'` ->
`score = P(yes) / (P(yes) + P(no))` from the first generated token's
softmax. This satisfies the frozen protocol's "pinned official offline
configuration" (no network call, no service endpoint).

## 5. Commands (planned, to be executed after this report lands)

All GPU commands run inside the Taiji container on H20-FoldUMM, GPU index 1
only (GPU0 is running the concurrent T710 job — must not collide), via
`script -qec "taiji_client exec ..." /dev/null` fake-PTY wrapping and
`cjob.sh` for anything long-running:

1. Build `/dockerdata/t720-janus-pro-r1/venvs/sft` and `.../rl` (pip
   install from the pinned subsets above, using the container's local
   wheel cache / star_proxy where a wheel must be fetched).
2. Download `OpenGVLab/InternVL2_5-8B` to local SSD
   (`/dockerdata/t720-janus-pro-r1/models/InternVL2_5-8B/`), verify all 4
   safetensors shard hashes against `source-lock.yaml`.
3. Download the 1 pinned `midbee/Janus-Pro-R1-Data` parquet shard to local
   SSD for the admission loader check, verify its hash.
4. Run `scripts/model_storage_preflight.py` against the local SSD path used
   for all of the above -> `configs/janus-pro-r1/admission/storage-preflight.json`.
5. Run the artifact hash checks that produce
   `configs/janus-pro-r1/admission/artifact-verification.json`.
6. Run the SFT optimizer smoke (`src/comppareto/adapters/janus_pro_r1/`
   adapter, ≤4 steps, single GPU, upstream `train_setup`/`run_step`
   sequence) on GPU1.
7. Run the GRPO smoke (≤4 steps, single GPU, custom rollout+advantage+loss
   loop calling upstream `InternVLReward`) on GPU1.
8. Write `runs/janus-pro-r1-stack-v1/{manifest.json,metrics.json,notes.md}`
   from the real numbers produced by steps 6-7.

## 6. GPU topology

Single H20 GPU (index 1) on the H20-FoldUMM Taiji container
(`task_flag gpu_model_train_split_llm218029E40DDAA49`,
`instance_id 8b1d81c89f83d4d4019f93a9170a251d`). GPU0 on the same container
is running the concurrent, independent T710 job — T720 must not touch GPU0.
GPUs 1-7 on this container run a benign `train2.py` GPU-placeholder process
(~325MB/100%util) that auto-yields once real usage on that GPU exceeds
~1200MB; it is not killed or otherwise interfered with.

## 7. Expected cost

Rough envelope estimate against the ≤12 GPU-hour budget:

- Env build + InternVL2.5-8B download + hash verification: CPU/IO-bound,
  ~0.2-0.5 GPU-hours of GPU-attached wall clock reserved defensively
  (downloads themselves do not occupy the GPU compute, but the container
  allocation is charged for the reservation window).
- SFT smoke (≤4 optimizer steps, ~7B trainable params, bf16, single GPU,
  no FSDP sharding so full optimizer state resident): expected well under
  1 GPU-hour for 4 steps plus one checkpoint save/reload cycle.
- GRPO smoke (≤4 optimizer steps, `num_generations` reduced from
  upstream's 8 to a small bounded value for the smoke, policy model +
  InternVL2.5-8B reward model co-resident on one 96GB H20): rollout
  generation dominates; expected 1-3 GPU-hours for 4 steps including reward
  scoring.
- Total expected: well under the 12 GPU-hour envelope; actual measured
  `resources.gpu_hours` will be recorded in
  `runs/janus-pro-r1-stack-v1/metrics.json` once the smokes run.

Memory feasibility (not yet empirically measured, estimated here and to be
confirmed): policy model bf16 weights ~14.84GB + AdamW bf16 m/v states for
the trainable subset (language_model + gen_embed + gen_head + gen_aligner +
aligner, dominated by the 7B-parameter language_model) + activations, plus
InternVL2.5-8B bf16 ~16GB resident for reward scoring — expected to fit
within a single 96GB H20 but this is the primary technical risk for the
GRPO smoke and will be reported honestly (including as a `[FAIL]`/blocked
item in `failure-ledger.md` if it does not fit) rather than assumed.
