# Janus-Pro-R1 RL (GRPO) environment lock (T720)

Separately pinned from the SFT environment above, per the frozen protocol
("Keep SFT and RL environments separately pinned when required").

## Upstream declared requirements

`vendor/janus-pro-r1/requirements-rl.txt` (pinned verbatim upstream, not
edited; 32 fully-versioned entries): `torch==2.5.1+cu121`,
`torchvision==0.20.1+cu121`, `transformers==4.50.0`, `tokenizers==0.21.2`,
`accelerate==1.4.0`, `trl==0.16.0`, `vllm==0.7.2`, `deepspeed==0.15.4`,
`apex==0.9.10dev`, `flash_attn==2.7.4.post1`, `peft==0.10.0`,
`liger_kernel==0.5.3`, `fastchat==0.1.0`, `petrel_client==0.0.1`,
`decord==0.6.0`, `datasets==3.5.0`, `numpy==2.3.1`, `sentencepiece==0.2.0`,
`Pillow==11.3.0`, `wandb==0.21.0`, `huggingface_hub==0.30.2`, others.

## What the T720 GRPO smoke actually installs

Built fresh under `/dockerdata/t720-janus-pro-r1/venvs/rl` on H20-FoldUMM
(local SSD), a strict subset of the above, based on an explicit source-level
audit of what `janus-rl/src/open_r1/{grpo_t2i.py, internvl_img.py,
mydataset.py}` and the vendored `internvl/` model package actually import
on the rollout -> reward -> loss -> backward -> optimizer path T720
exercises:

`torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `transformers==4.50.0`,
`tokenizers==0.21.2`, `einops==0.8.1`, `numpy==2.3.1`,
`sentencepiece==0.2.0`, `Pillow==11.3.0`, `timm==1.0.16`,
`decord==0.6.0` (imported by `internvl_img.py:4` for `VideoReader`, even
though only the image path is exercised — kept because the import is
unconditional at module load time), `flash_attn==2.7.4.post1` (InternVL's
`from_pretrained(..., use_flash_attn=True)` call requires it importable;
if the container's flash-attn ABI build fails, see the documented fallback
below).

## Deliberately NOT installed, with the exact upstream fallback that makes
## this safe (verified by reading the vendored source, not assumed)

- **`apex`** — every import site in the vendored `internvl/model/` package
  is wrapped `try: from apex.normalization import FusedRMSNorm ... except
  ImportError: from torch.nn import LayerNorm as ...` / a pure-PyTorch
  `InternRMSNorm` fallback class is defined right next to the guarded
  import. Confirmed via `grep -rn "apex" vendor/janus-pro-r1/janus-rl/src/open_r1/`
  — every hit is inside a `try/except ImportError` block, no hard
  dependency.
- **`petrel_client`** — same pattern: guarded `try: from petrel_client.client
  import Client except ImportError: ...` falls back to local `PIL.Image.open`
  loading in the dataset/preprocessing utilities. Also confirmed via PyPI:
  neither `petrel-client` nor `petrel_client` resolves on public PyPI
  (`https://pypi.org/pypi/petrel-client/json` and
  `.../petrel_client/json` both 404) — it is Shanghai AI Lab's internal Ceph
  client, unreachable/unneeded outside their cluster, and the fallback path
  is what a public reproduction is expected to use.
- **`fastchat`** — only referenced by an optional flash-attn monkeypatch
  module (`internvl/patch/...`) that `grpo_t2i.py`'s exercised import chain
  never calls; not imported by `internvl_img.py`, `mydataset.py`, or
  `grpo_t2i.py`'s top-level statements.
- **`vllm`, `deepspeed`, `accelerate` (multi-process launch)** — these
  back the full distributed `GRPOTrainer`/HF-`Trainer` path
  (`recipes/t2i_generation/grpo.yml` sets `num_processes: 8`,
  `use_vllm: false` in the shipped recipe but the `GRPOConfig`/`TrlParser`
  plumbing still imports `accelerate` transitively via `trl`). T720's
  bounded smoke uses a **custom single-GPU loop** that calls upstream's own
  `InternVLReward.evaluate()` and the policy model's `.generate()` directly
  for rollout, and computes the GRPO advantage/policy-gradient loss by hand
  over that one batch — it does not invoke `trl.GRPOTrainer.train()` or any
  `accelerate`/`deepspeed` process launcher. This is the single largest
  documented scope reduction in T720 and is called out again in
  `reports/T720/first-report.md` and `reports/T720/claim-check.md`.

## Reward model path (shared detail, RL env only)

`internvl_img.py`'s `InternVLReward.__init__` hardcodes a path string
(`/mnt/prev_nas/refine_draw_RL/models/models/OpenGVLab/InternVL2_5-8B`) —
an internal cluster path from the original authors, not portable. T720's
adapter parameterizes this as an environment-driven path
(`MODEL_ROOT`-relative) pointing at our own locally verified download of
`OpenGVLab/InternVL2_5-8B@e9e4c0dc1db56bfab10458671519b7fa3dd29463`
(see `configs/janus-pro-r1/admission/source-lock.yaml`), otherwise calling
`InternVLReward.evaluate()` and its prompt template/scoring formula
unmodified. Note: `recipes/t2i_generation/grpo.yml` sets `internvl_tp: 26b`
as its default recipe value (implying InternVL2.5-26B), but the
`InternVLReward` class actually instantiated by `internvl_img.py` is
hardcoded to the 8B checkpoint regardless of that recipe field — T720 uses
the 8B model as literally coded (the smaller, actually-invoked configuration),
not the 26B value the YAML comment/default suggests.
