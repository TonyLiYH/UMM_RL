# T250 — Checkpoint stage evidence

Per execution stage 3 of the task file: stage labels below are evidence-backed
(direct quotes / code citations), not inferred from checkpoint filenames alone.
Where evidence is incomplete, the label is stated as hedged/unknown rather than
guessed.

## Candidate A — SenseNova-U1-8B-MoT-SFT

**Stage label: SFT — post Stage 4 (Unified Supervised Fine-Tuning), pre Stage 5
(Post Training for T2I Generation, which is explicit RL). High confidence.**

Evidence:

1. Official GitHub README (`OpenSenseNova/SenseNova-U1/README.md`), "Models"
   section, verbatim: *"SFT models ... are trained via Understanding Warmup,
   Generation Pre-training, Unified Mid-training, and Unified SFT, with final
   models obtained after Multi-Expert RL and OPD training."* This is a direct
   statement, by the checkpoint's own publisher, that the `-SFT`-suffixed
   checkpoint line sits strictly before the RL/OPD stage. It is the strongest,
   least-ambiguous evidence available across all three candidates.
2. arXiv 2605.12500 (fetched via ar5iv HTML), "Training Procedure" section,
   names five stages explicitly:
   - Stage 1: Understanding Warmup
   - Stage 2: Generation Pre-Training (3 phases)
   - Stage 3: Unified Mid-Training
   - Stage 4: Unified Supervised Fine-Tuning
   - Stage 5: "Post Training for T2I Generation" — explicit reinforcement
     learning via **Flow-GRPO**, with three named reward models: an OCR-IoU
     text-rendering reward, a VLM-judge style-following reward, and an
     HPSv3 aesthetic/preference reward.
   Stage 5 is unambiguously a preference/RL stage (reward models + RL
   algorithm named explicitly), and the README confirms the `-SFT` checkpoint
   precedes it.
3. Corroborating changelog entry in the same README (`[2026.??.??]` training
   code release note): "Release the full-parameter fine-tuning training code
   ... for SenseNova-U1" — confirms that Stage 4 (Unified SFT) is the stage the
   released *training code* targets, consistent with the checkpoint being the
   Stage-4 output artifact.

Residual open item: the README's changelog also references "OPD" training as
grouped with RL in the post-SFT description, but no fetched document spells
out what OPD stands for. This does not change the stage label (OPD is named
as occurring *after* SFT, same as RL, so it does not weaken the "-SFT precedes
preference/RL" conclusion) — recorded as an open terminology gap only.

**Revision addendum (2026-09-16):** this stage label applies to the exact
checkpoint loaded and smoke-tested in this revision —
`sensenova/SenseNova-U1-8B-MoT-SFT` @ HF revision
`846ff1352e3a4e900d064740cddfc163b115646f`, downloaded to and executed from
container-local SSD, 214 files hashed (sha256+size in
`configs/admission/posttraining-startpoints/checkpoint-hashes.json`), loaded
directly via `NEOChatModel.from_pretrained` and exercised with a real
pure-understanding forward+backward pass
(`understanding_loss=9.830007553100586`). The stage label above is therefore
now backed by both the official README/paper statement (documentation
evidence) and a real, successful load+forward+backward on the pinned
checkpoint itself (execution evidence) — not documentation evidence alone.
See `reports/T250/training-interface-audit.md`'s Revision addendum for full
detail.

## Candidate B — Show-o2-1.5B (reusing T210's accepted admission)

**Stage label: SFT-equivalent, consistent with completion of Stage-2
(LLaVA-OneVision/DenseFusion-style instruction tuning). Moderate confidence —
inferred from pipeline structure and functional behavior, not from an
explicit checkpoint-card stage statement.**

Evidence:

1. Official `show-o2/README.md` "Training" section documents exactly two
   supervised stages for the base checkpoint line:
   - Stage-1: pretraining-style training on jsonl image-text pairs
     (`train_showo2_1.5b_stage1.sh`)
   - Stage-2: instruction-tuning-style SFT following LLaVA-OneVision /
     DenseFusion data conventions (`train_showo2_1.5b_stage2.sh`)
   plus an optional, clearly-separate "downstream fine-tuning for
   mixed-modality generation" path (`train_mixed_modality_simple.py`) that is
   NOT part of the base `showlab/show-o2-1.5B` checkpoint's lineage (it
   produces a differently-named derivative, e.g.
   `showo2-qwen2-5-1.5b-downstream-mixed-modality-432x432`).
2. No RL, DPO, RLHF, or preference-reward terminology appears anywhere in the
   Show-o2 README, training scripts, or arXiv abstract (2506.15564) — the
   official pipeline has no preference/RL stage at all for this candidate.
3. T210's own accepted GPU smoke (`reports/T210/task-path-smoke.md`) ran both
   `inference_mmu.py` (multimodal understanding) and `inference_t2i.py`
   (text-to-image generation) against the pinned checkpoint and observed
   coherent, correct outputs on both paths — functional evidence consistent
   with a completed instruction-tuned (SFT) checkpoint, though this is
   behavioral corroboration, not a textual stage statement.

Explicit hedge (per the task's honesty mandate): no model card, README
paragraph, or release note for `showlab/show-o2-1.5B` states outright "this
checkpoint is the Stage-2 SFT output." The label above is the best-supported
conclusion available, combining pipeline-structure evidence (only two stages
exist, no RL stage exists) with T210's functional smoke evidence, but it
should not be repeated as if it were an explicit publisher statement.

## Candidate C — UniDDT

**Stage label: hedged/unknown between "post-Joint-training" and
"post-Duality-post-training"; in either case NOT a preference/RL stage.**

Evidence:

1. Official GitHub README (`MCG-NJU/UniDDT/README.md`), "Architecture and
   training" section, names three stages: Warmup, Joint training, Duality
   post-training. The released checkpoint's shipped config disables the
   warmup/joint re-initialization switches, with the README's own
   explanation that this is "because the released checkpoint already carries
   those weights" — consistent with, but not an explicit confirmation of,
   having gone through Duality post-training specifically (it is equally
   consistent with a checkpoint saved right after Joint training, before
   Duality post-training, since both would already "carry" warmup/joint
   weights).
2. arXiv 2606.16255 (fetched via ar5iv), abstract and Figure 1 caption name
   only "Warmup" and "Joint training" as the pipeline; "Duality
   post-training" appears in the README's prose but is not named in the
   figure caption I could fetch from the abstract page — a minor
   inconsistency between the two official sources that I have not resolved
   either way (recorded as an open item rather than guessed).
3. Regardless of which of the two candidate stages applies, **neither is a
   preference/RL stage**: the README describes Duality post-training as
   caption-likelihood maximization over the diffusion decoder's own
   intermediate sampling states (a self-distillation / consistency-style
   refinement restricted to the diffusion decoder), with no reward model, no
   human/AI preference signal, and no RLHF/DPO terminology anywhere in the
   README or arXiv abstract. So even under the more-advanced reading, UniDDT's
   released checkpoint would still satisfy the decision rule's "precedes the
   target preference/RL stage" criterion (trivially — it has no such stage in
   its pipeline at all).

This stage-label ambiguity does **not** by itself disqualify UniDDT under the
decision rule (see `decision-matrix.md`); UniDDT is excluded for a separate,
independently-sufficient reason (missing license — see below and
`failure-ledger.md`).
