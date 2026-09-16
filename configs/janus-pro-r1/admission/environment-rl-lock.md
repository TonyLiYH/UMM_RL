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

`torch==2.6.0`, `torchvision==0.21.0`, `transformers==4.50.0`,
`tokenizers==0.21.2`, `einops==0.8.1`, `numpy==2.2.6`,
`sentencepiece==0.2.0`, `Pillow==11.3.0`, `timm==1.0.16`,
`decord==0.6.0` (imported by `internvl_img.py:4` for `VideoReader`, even
though only the image path is exercised — kept because the import is
unconditional at module load time), `easydict`, `attrdict==2.0.1`
(both required by `janus-rl/src/open_r1/models/{projector.py,
modeling_vlm.py}` — the same vendored `models` package `JanusLLamaModel`
subclasses from, and the same `models` copy used on the SFT side; see
"attrdict / Python 3.10 compatibility shim" below and
`environment-sft-lock.md`'s identical section — this was a real gap in an
earlier draft of this document that omitted both packages, caught by
actually running the model-construction code rather than assuming the
import list from a partial source scan), `scipy==1.15.3` (see its own
section below), and `peft==0.10.0` (pinned to upstream's own
`requirements-rl.txt` value): `internvl_img.py`'s `from internvl.model.internvl_chat
import InternVLChatModel` transitively executes
`internvl/model/internvl_chat/modeling_internvl_chat.py`'s own `from peft
import LoraConfig, get_peft_model` at import time (unconditional, not
inside a lazy function) even though this smoke's `InternVLReward` never
actually applies a LoRA adapter -- another case, like `deepspeed` on the
SFT side, of a heavy dependency being a real *import-time* requirement
despite being functionally unused on the code path this smoke exercises.
Caught by an actual `ModuleNotFoundError: No module named 'peft'` on the
first real GPU invocation of `grpo_smoke.py`, not by the earlier source
scan (which only grepped the files `grpo_t2i.py`/`internvl_img.py`/
`mydataset.py` import directly, not their own transitive imports).
`peft==0.10.0` pulls in `accelerate` (resolved to `1.15.0`, not upstream's
pinned `1.4.0`) as its own transitive dependency; this is unrelated to, and
does not contradict, the "`accelerate` deliberately not installed for
multi-process launch" note further below -- `accelerate` here is merely
importable as a side effect of `peft`'s own requirements, and this smoke
never calls into `accelerate`'s distributed-launch machinery.

`numpy` is pinned to `2.2.6`, not upstream's `requirements-rl.txt` value
`2.3.1`: this cp310 venv (matching the `torch==2.6.0` cp310 wheel already
installed) has no `numpy==2.3.1` wheel available — `numpy>=2.3.0` requires
Python>=3.11. `torch`/`torchvision` themselves already pull in
`numpy==2.2.6` as a transitive dependency in this venv; pinning explicitly
to that already-resolved version (rather than upstream's cp310-incompatible
2.3.1) is the deliberate, minimal deviation. Caught by actually reading a
build log rather than trusting its `exit=0` — the first attempt's combined
`pip install transformers==... numpy==2.3.1 ...` command failed entirely on
the numpy resolution and silently skipped installing `transformers` and
every package listed after it in that command, but the script's own final
`echo ... DONE` still ran and returned an unrelated non-zero only because a
later, separate verification step then correctly failed on
`ModuleNotFoundError: No module named 'transformers'`.

## attrdict / Python 3.10 compatibility shim

Identical issue and identical fix to the SFT venv (see
`environment-sft-lock.md`'s section of the same name) — the same vendored
`models/{projector.py,modeling_vlm.py}` package is used by
`janus-rl/src/open_r1/llama.py`'s `JanusLLamaModel(MultiModalityCausalLM)`.
Verified end-to-end in the RL venv:
`attrdict shim OK: .../venvs/rl/lib/python3.10/site-packages/attrdict/__init__.py`,
`easydict OK: .../venvs/rl/lib/python3.10/site-packages/easydict/__init__.py`,
followed by successful `torch`/`torchvision`/`transformers`/`einops`/
`timm`/`sentencepiece`/`PIL`/`decord`/`numpy` imports. `easydict` was
installed from the shared local wheel cache; `attrdict` is not present in
that cache (only `easydict` is) and was installed via the `star_proxy`
PyPI proxy instead — safe for this specific package since it is small,
pure-Python, with no native/compiled extensions.

Torch is bumped from upstream's `2.5.1+cu121` pin to `2.6.0` (cu12,
cxx11abi=False, cp310) to match a prebuilt wheel already present in the
container's local wheel cache (`/apdcephfs_cq7/.../tmp/vllm-wheels/`),
avoiding a from-source build. Verified compatible with
`transformers==4.50.0` (no version ceiling on torch in transformers'
own `setup.py` for that release).

## scipy: required for an unrelated reason (package-init side effect, not Janus code)

`scipy==1.15.3` (shared local wheel cache, cp310 build) is also installed
in this venv, for the identical reason documented in
`environment-sft-lock.md`'s section of the same purpose: invoking
`python -m comppareto.adapters.janus_pro_r1.grpo_smoke` forces Python to
import the top-level `src/comppareto/__init__.py` first (outside T720's
`allowed_paths`), which unconditionally imports `.quadratic`, which imports
`scipy.optimize.minimize` — nothing to do with the Janus-Pro-R1 GRPO code
itself. Caught by an actual `ModuleNotFoundError` on the SFT side first,
then pre-emptively fixed here before the first `grpo_smoke.py` invocation
(same `comppareto.__init__` import chain applies identically).

`flash_attn` **is a genuine hard requirement for the policy model**,
contradicting an earlier draft of this document (see the corrected section
immediately below); it remains true, and unaffected, that InternVL's own
attention path degrades gracefully without it.

## flash_attn: corrected -- required for the policy model too, not just optional for InternVL

An earlier version of this document asserted `flash_attn` was "deliberately
not installed" for the entire RL venv, based on checking only
`InternVLReward`'s own `InternVisionModel`/`InternLM2` guarded-import
fallback (still accurate for the reward model in isolation, see below).
That check never covered the vendored `janus-rl/src/open_r1/models/
modeling_vlm.py` -- a *separate* copy of the same `models` package used on
the SFT side, imported by `JanusLLamaModel(MultiModalityCausalLM)`'s own
`__init__` -- which hardcodes `language_config._attn_implementation =
'flash_attention_2'` **unconditionally**, identically to the SFT-side
finding in `environment-sft-lock.md`. Caught by an actual `ImportError:
... the package flash_attn seems to be not installed` when
`JanusLLamaModel.from_pretrained(...)` was invoked for the first time on
GPU -- not by a more careful re-read of the source. Fixed by installing the
identical wheel used on the SFT side into this venv too: both venvs pin
`torch==2.6.0+cu124` (cp310, cxx11abi=False), so the same
`flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
from the shared local wheel cache applies unchanged, installed by direct
wheel-file path (same case-sensitivity workaround as the SFT side).
Verified with a real GPU forward pass on GPU1: `flash_attention_2 minimal
LlamaForCausalLM forward OK (RL venv), logits finite, shape (1, 8, 32)`.

The InternVL-side finding from the original draft is still correct and
kept below unmodified: InternVL's own `InternVisionModel`
(`modeling_intern_vit.py:24-30`) and `InternLM2` (`modeling_internlm2.py:49-62`)
both guard the `flash_attn` import in `try/except ImportError`, setting
`has_flash_attn = False` on failure, and
`modeling_internvl_chat.py:62-64` computes
`use_flash_attn = use_flash_attn if has_flash_attn else False` — i.e. the
`use_flash_attn=True` argument `InternVLReward.__init__` passes to
`InternVLChatModel.from_pretrained` would have been automatically
downgraded to eager attention had `flash_attn` not been importable, with a
printed warning, not an exception -- this describes the fallback path in
isolation and remains accurate as a statement about the code's robustness.
In practice, because `flash_attn` ended up installed in this venv anyway
(a real requirement for the policy model, see above, using the exact same
ABI-matched wheel), InternVL2.5-8B also runs on real flash attention for
this smoke rather than the eager fallback -- a side effect of fixing the
policy-model requirement, not a separate deliberate choice, and not a
correctness concern either way since the fallback is upstream's own
documented, exception-free behavior.

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
