# T250 — Training interface audit (static, source-level, no GPU)

Per execution stage 4: official training/resume paths inspected statically
from source. Static source evidence was judged sufficient for Show-o2 and
UniDDT's resume-restoration granularity. For SenseNova-U1-8B-MoT-SFT
specifically, 2026-09-16 local review (see the task file's "Local review
requirements") required real GPU evidence rather than static/documentation
evidence alone; that GPU work is now done and is reported inline in
Candidate A's section below (superseding the original static-only framing for
this candidate). See `first-report.md`'s revision addendum for the exact
commands and `decision-matrix.md` for how each dimension is weighted.

## Candidate A — SenseNova-U1 (`training/` subtree of `OpenSenseNova/SenseNova-U1`)

**Revision addendum (2026-09-16) — real GPU evidence, not static-only.**
Source pinned to `github.com/OpenSenseNova/SenseNova-U1@f97964a6e54b0abf92aa2db849af4e942bb2ff08`
(the commit actually checked out for the editable-installed `sensenova_u1`
package imported at runtime, confirmed via `git log -1`/`git remote -v`,
clean tree). The `-SFT` checkpoint itself (`sensenova/SenseNova-U1-8B-MoT-SFT`
@ HF revision `846ff1352e3a4e900d064740cddfc163b115646f`) was downloaded to
container-local SSD (`/dockerdata`, `filesystem_class=local`/`xfs`, confirmed
by `configs/admission/posttraining-startpoints/storage-preflight.json`),
hashed (214 files, sha256 + size, see
`configs/admission/posttraining-startpoints/checkpoint-hashes.json`), and
loaded directly via `NEOChatConfig.from_pretrained(CKPT)` +
`NEOChatModel.from_pretrained(CKPT, config=config, torch_dtype=torch.bfloat16)`
on a single H20 GPU (`load_seconds=4.914`). This is the SFT checkpoint itself,
not T230's separately-audited final-MoT checkpoint.

Reading `modeling_neo_chat.py` and `modeling_qwen3.py` in full revealed that
`NEOChatModel.forward()` and `NEOChatModel.batch_chat()` both raise
`NotImplementedError` as their literal first line in this pinned commit (the
remainder of `forward()`'s body, a CE-loss/vision-splicing block, is
unreachable dead code), and that `Qwen3Attention.forward()` /
`Qwen3DecoderLayer.forward()` both raise
`NotImplementedError("...see issue #207...")` whenever a batch mixes
understanding and generation tokens — dispatching cleanly to a
`forward_und`-only or `forward_gen`-only path otherwise. This means "one
pure-understanding and one pure-generation forward/loss smoke" (as required
by local-review item 3) is the *only* implemented path in this revision, not
a simplification chosen by this audit; a genuinely mixed und/gen batch
through the released top-level `forward()` is not currently executable at
all. This also refines the "sequential-task-batch support at one frozen
shared version" claim below: batches must be pure-understanding OR
pure-generation per forward call, not interleaved within one call.

Two smokes were executed on the loaded SFT checkpoint, using only
officially-implemented, non-stubbed entry points (mirroring exactly the
shapes/calling-conventions of the officially-live `_build_t2i_text_inputs`,
`_build_t2i_image_indexes`, `_t2i_predict_v`, `patchify`, and `extract_feature`
helpers that `t2i_generate()` itself calls):

- **Pure-understanding**: built `input_ids`/`indexes`/`attention_mask` via
  `model._build_t2i_text_inputs(tokenizer, query)`, called
  `model.language_model(..., labels=input_ids, use_cache=False)`, and called
  `.backward()` on the resulting cross-entropy loss.
  `understanding_loss=9.830007553100586`. Gradients landed only on the
  `shared_backbone` parameter group (`generation_private` and
  `vision_shared_understanding` both had zero gradient), confirming the
  understanding path touches exactly its owned parameters.
- **Pure-generation**: built a synthetic 64x64 latent, patchified it,
  extracted flow-matching image embeddings via `extract_feature(...,
  gen_model=True)`, added timestep embeddings from
  `fm_modules["timestep_embedder"]`, predicted velocity via
  `model._t2i_predict_v(...)`, and took an MSE loss against a random target
  before calling `.backward()`. `generation_loss=5.4360198974609375`.
  Gradients landed only on the `generation_private` group (zero elsewhere),
  confirming clean separation in the opposite direction.

Both results, plus the optimizer/scheduler/resume-metadata construction
described next, are recorded verbatim in
`runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json`
(`status: "pass"`, no exceptions on first attempt).

**Optimizer/scheduler/resume construction without mutating weights
(local-review item 4).** Three named parameter groups were built
(`shared_backbone`, `generation_private`, `vision_shared_understanding`),
wrapped in a real `torch.optim.AdamW(param_groups, lr=1e-5)` plus a
`torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)`, alongside
a `resume_metadata` dict (`global_step`, `epoch`,
`optimizer_state_dict_keys`, `scheduler_state_dict`, `rng_state_cpu_len`,
`rng_state_cuda_len`). No `.step()` was ever called on either the optimizer
or the scheduler. A sha256 fingerprint over six sampled parameter tensors
(one per architectural region: shared embedding, a shared-backbone
`self_attn.q_proj`, a generation-private `q_proj_mot_gen`, a
generation-private `mlp_mot_gen.gate_proj`, and two vision-embedding tensors)
was taken immediately before and after optimizer+scheduler construction:
both fingerprints are identical
(`79efdf9257c805710d0ccd92f4fed56512f6fd5e05db5cbba6965a4116b646b9`),
proving construction alone did not mutate weights. This is expected —
`torch.optim.AdamW` lazily allocates its `m`/`v` moment buffers only on the
first `.step()` — but it is now verified empirically on this exact
checkpoint's exact parameter tensors, not merely asserted from PyTorch's
documented behavior.

**Real measured GPU memory footprint (local-review item 5), replacing the
unverified "8x80GB shipped default" as the basis for feasibility.** On one
H20 (`gpu_total_memory_bytes=102010781696`, ~95GB usable of 96GB nominal):
weights alone after load = `35,107,999,744` bytes allocated (~35.1GB, matches
17.55B bf16 params x 2 bytes). After building the optimizer+scheduler:
`max_allocated=37,597,319,168`. After the understanding backward:
`max_allocated=53,877,337,600`. After the generation backward (the smoke's
peak): `allocated=51,594,826,752`, **`max_allocated=59,478,750,720`**
(~59.5GB), `reserved=60,089,696,256` (~60.1GB) — comfortably within a single
96GB H20, and well within this audit's <=2-GPU envelope. This is real,
measured evidence that the SFT checkpoint's two task paths execute on the
declared hardware; it is not evidence of a training run (no `.step()`, no
real dataset, no activation scaling from longer sequences/larger batches).

Separately, exact per-group parameter counts were obtained at zero
cost by constructing `NEOChatModel(config)` under `torch.device("meta")`
(shapes only, no real weights/compute):
`shared_backbone=9,348,413,952`, `generation_private=8,186,358,272`,
`vision_shared_understanding=17,568,768`, `total=17,552,340,992`. This
reconciles with the docs-sourced split
(`docs/parameter_breakdown.md`: shared 1.245B + understanding_transformer
8.121B = 9.366B vs. measured shared_backbone + vision_shared_understanding
= 9.366B; generation_transformer 8.186B matches generation_private exactly;
total 17.552B matches exactly). Using these exact counts: full bf16 weights
= 35.1GB total (18.7GB shared_backbone + 16.4GB generation_private +
0.035GB vision); AdamW fp32 optimizer state (2 copies x 4 bytes/param) for
`generation_private` alone = ~65.5GB, for `shared_backbone` alone = ~74.8GB.
For T270's most likely trainable subspace (`generation_private`-only, since
that is the group the generation-only smoke shows can be updated without
touching the understanding path): weights (35.1GB, all resident, since
`forward_und` still needs `shared_backbone`) + grads (~16.4GB bf16) +
AdamW state (~65.5GB) = ~117GB, which **exceeds one 96GB H20 but plausibly
fits 2xH20 (192GB) with optimizer-state sharding (ZeRO-1/2) or an 8-bit
optimizer**, leaving headroom for activations at modest batch/sequence
sizes. Full-parameter fine-tuning of both `shared_backbone` +
`generation_private` together would need ~140.3GB of AdamW state alone
(74.8+65.5GB), which does **not** fit 2xH20 (192GB) once weights/grads/
activations are added, without further sharding/offload — this is the
smallest-feasible-H20-topology measurement/derivation local review
requested, replacing the unverified assumption that the shipped 8x80GB
launcher default was itself evidence of infeasibility on 2 GPUs for the
narrower subspace T270 is actually expected to train.

Exact commands executed for this addendum are recorded in
`reports/T250/first-report.md`'s revision addendum.

**Trainer/optimizer**: custom framework derived from InternEvo
(`sensenovalm/core`), `HybridZeroOptimizer` (ZeRO-1) as the optimizer wrapper;
Intern Sequence Parallel (ISP) for sequence/tensor parallelism; weight-sharding
via `wp_size` (FSDP-like); pipeline parallel disabled by default
(`pp_size=1`) in shipped configs.

**Resume mechanism** — read `sensenovalm/checkpoint/checkpoint_manager.py`
(`try_load_internevo_ckpt`) directly:
- Calls `load_model_checkpoint(...)`, `load_optimizer_checkpoint(...)`,
  `load_scheduler(...)`, and restores `TrainState`/`load_context` (step
  counters, dataloader position for the "streaming-resumable data loading
  with checkpoint state" the training README advertises).
- Optimizer-state restoration is gated by `ckpt.load_optimizer` (a config
  flag defaulting to enabled in shipped configs) and an explicit
  `content=("model","sampler","optimizer")` tuple passed at the call site —
  i.e., **weights + optimizer state + sampler/dataloader position** are all
  restored when resuming, not merely the model weights.
- **Verdict: full-state resume (model, optimizer, scheduler, step/dataloader
  counters).** This is the strongest resume guarantee among the three
  candidates and is confirmed at the code level, not inferred from framework
  reputation alone.

**Configuration/data interfaces**: env-var-driven `torchrun` launchers
(`shell/train_u1/{8B,A3B,U1.5_8B}.sh`); dataset described via a meta JSON
(`mm_data_path`) listing `{root, annotation, repeat_time, task}` per dataset;
five `type_id` task categories (`mm_t2i`, `mm_it2i`, `mm_interleave_gen`,
`mm_interleaved`, `multimodal`) with **per-task loss bucketing already
implemented** or logging — directly supports "sequential-task-batch support
at one frozen shared version" as required by the audit dimensions, since
understanding and generation batches can be mixed within one `mm_data_path`
meta JSON and monitored independently. **Revision addendum (2026-09-16):**
"mixed within one meta JSON" means alternating pure-understanding and
pure-generation *batches* fed as separate forward calls, not a single batch
whose tokens mix und/gen — the pinned commit's `Qwen3Attention.forward()`/
`Qwen3DecoderLayer.forward()` raise `NotImplementedError` for a genuinely
mixed-token batch (see the Revision addendum above), so "sequential" here is
load-bearing: task batches must be sequenced, not co-mingled within one
forward call.

**Minimum/recommended GPU memory**: shipped `8B.sh` launcher requires 1 node
× 8 GPUs × 80GB HBM minimum (`wp=8 × tp=1 × pp=1 = 8` ranks at
`seq_len=28672`, `num_imgs=144`) for full-parameter fine-tuning at the
launcher's shipped sequence/image scale — this remains the shipped
*convenience* default, not this audit's own measured floor. Superseded by
the real single-H20 measurement and per-group derivation in the "Revision
addendum" above: this audit's own smoke peaked at ~59.5GB on ONE H20, and a
`generation_private`-only T270 subspace is derived to plausibly fit 2xH20
with optimizer-state sharding, while full-parameter fine-tuning of both
groups together does not fit 2xH20 without further sharding/offload. This
does not block the *audit* (no training is run here) but replaces the prior
material planning fact for T270 with measured/derived numbers instead of the
shipped default alone.

**Shared/private/routed parameter classification**: `docs/parameter_breakdown.md`
in the official repo publishes an exact, reproducible split for the MoT
architecture: `shared` group (`language_model.model.embed_tokens.weight` +
`language_model.lm_head.weight`) = 1.245B params (7.09%);
`understanding_transformer` = 8.121B (46.27%); `generation_transformer` =
8.186B (46.64%); total 17.552B params. Pathway totals: understanding path
≈9.366B (53.36%), generation path ≈9.431B (53.73%). This is the most
precisely quantified and independently reproducible parameter-block registry
of the three candidates (published with a documented inspection script).

**Missing dependencies**: `flash-attn` build extra required for shipped
configs (`use_flash_attn=True`); Apex / `grouped_gemm` / `deeplink_ext`
accelerator kernels are configuration-specific and must be built separately.
No missing tokenizer/VAE/reward-model dependency identified for the SFT
checkpoint's own forward/training path (reward models are only needed for
Stage 5 RL, which is explicitly out of scope for this audit and for T270's
starting point).

**License**: Apache-2.0 for the training code and (per the inference-repo
license file referenced from the training README) the released weights;
some files under `sensenovavl/model/sensenovavl_moe_chat/` are MIT-licensed
(derived from InternVL, attribution headers present) — a permissive,
compatible combination.

## Candidate B — Show-o2-1.5B (`showlab/Show-o`, `show-o2/` subdir)

**Trainer/optimizer**: HuggingFace `accelerate`-based training
(`accelerate launch`, DeepSpeed ZeRO-2 config for the downstream
mixed-modality path); standard `AdamW` optimizer instantiated directly in
`train_stage_one.py` (no custom optimizer wrapper).

**Resume mechanism** — read `train_stage_one.py` resume block directly
(lines ~264-320):
- On resume, the script scans the output directory for `checkpoint-<step>`
  subdirectories, picks the latest, and restores **model weights only** via
  `model.load_state_dict(...)` (or the `accelerate`-wrapped equivalent).
- `global_step` and `first_epoch` are **reconstructed from the checkpoint
  directory name** (parsing the trailing integer), not read back from any
  serialized optimizer/scheduler state file.
- A **fresh** `AdamW` optimizer and a **fresh** LR scheduler (cosine/warmup)
  are instantiated after the weight-load — optimizer momentum/variance
  buffers and scheduler warmup/decay position are **not** restored.
- **Verdict: weights-only resume.** This is a materially weaker guarantee
  than SenseNova-U1's (no optimizer/scheduler state carried across a
  resume), and must be stated plainly per the task's honesty mandate — it is
  a genuine limitation of this candidate as a post-training starting point
  if long-horizon resumable training is required, though it does not by
  itself disqualify Show-o2 as a *starting point* for a **fresh** joint
  post-training run (which would initialize a new optimizer/scheduler
  regardless of the source checkpoint's own resume fidelity).

**Configuration/data interfaces**: YAML configs under `configs/`
(`showo2_1.5b_demo_*.yaml` for inference; `train_showo2_1.5b_stage{1,2}.sh`
launchers for training); Stage-1 data is jsonl `{path, prompt}` pairs; Stage-2
data follows LLaVA-OneVision/DenseFusion annotation conventions.
`frozen_params` config key allows freezing named submodules
(`image_embedder_und`, `und_trans`, `showo`, `position_embedding` shown in
the downstream mixed-modality example) — directly usable for shared/private
parameter partitioning in a joint post-training design, though this is a
manual allow-list, not an automatically-derived registry like SenseNova-U1's.

**Minimum/recommended GPU memory**: T210's own measured smoke used a single
H20 with ~13.9-14.9GB allocated for inference; the shipped training launcher
targets one node × 8 GPUs, but (unlike SenseNova-U1) this is a convenience
default for the `accelerate` multi-GPU config, not a hard topology
requirement — the 1.5B model is far smaller and could plausibly run training
on fewer/smaller GPUs, though this has not been verified for training
specifically (only inference was smoke-tested by T210).

**Shared/private/routed parameter classification**: reuse
`configs/admission/showo2/parameter-block-registry.yaml` from T210's
admission wholesale (no need to re-derive).

**Missing dependencies**: T210 already identified and fixed a
`wandb==0.17.0` pin requirement and a `wandb.util.generate_id` removal issue;
also flagged `CompVis/stable-diffusion-safety-checker` as an external
dependency with an open ("More information needed") license status — carried
forward as a pre-existing open item, not new to this audit.

**License**: Apache-2.0 (confirmed by T210's accepted admission).

## Candidate C — UniDDT (`MCG-NJU/UniDDT`)

**Trainer/optimizer**: PyTorch Lightning (`lightning.pytorch.cli.LightningCLI`
in `main.py`); optimizer/scheduler configuration is delegated to Lightning's
standard `configure_optimizers()` hook pattern (framework-managed, not a
custom wrapper).

**Resume mechanism**:
- `main.py` reads a `ckpt_path` key from the CLI config
  (`is_resume = self._get(self.config_init, "ckpt_path", ...)`) and passes it
  through to Lightning's `Trainer.fit(ckpt_path=...)` call path.
- PyTorch Lightning's documented default behavior for `ckpt_path` restores
  model weights, optimizer state, LR scheduler state, and
  epoch/global-step/loop position — a full-state resume guarantee **at the
  framework level**.
- **Caveat (explicitly flagged, not glossed over):** I read `main.py`'s
  top-level wiring but did **not** locate and fully read UniDDT's own
  `Trainer`/`LightningModule` subclasses (e.g. any custom
  `on_load_checkpoint`/`on_save_checkpoint` override) to rule out a
  project-specific override that could silently narrow this guarantee (for
  example, EMA-weight handling or the dual diffusion-decoder/LLM-backbone
  split could plausibly need custom checkpoint hooks that I have not
  verified). Recorded as "framework-level guarantee, not independently
  verified against UniDDT's own trainer subclasses" — an honest partial
  finding rather than an inferred pass.

**Configuration/data interfaces**: Lightning CLI YAML configs; three-stage
pipeline (Warmup, Joint training, Duality post-training) with per-stage
config files (based on README's "Architecture and training" section).

**Minimum/recommended GPU memory**: not independently determined in this
audit (README does not appear to publish an explicit minimum-hardware table
comparable to SenseNova-U1's or a measured smoke comparable to Show-o2's/
T210's); recorded as an open item, not fabricated.

**Shared/private parameter classification**: architecturally well-defined
(shared Noisy-ViT + Qwen3-VL-4B LLM backbone vs. private
`dit_decoder.py`/`jit_decoder.py` diffusion decoder), but no published exact
parameter-count breakdown comparable to SenseNova-U1's
`docs/parameter_breakdown.md` was found — the split is qualitatively
auditable from the code/architecture but not quantitatively pre-published.

**License: UNRESOLVED / MISSING.** No `LICENSE` or `LICENSE.md` file exists
in the `MCG-NJU/UniDDT` GitHub repository root (confirmed via direct `curl`
requests returning HTTP 404 for both filenames), and the Hugging Face Hub
API response for the model (`huggingface.co/api/models/MCG-NJU/UniDDT`)
carries no `license:` tag. This is a code/API-confirmed gap, not an
assumption, and independently disqualifies UniDDT under the decision rule's
"permits the intended research use under recorded licenses" criterion
regardless of its otherwise-competitive resume/architecture properties. See
`failure-ledger.md`.
