# Vendored source: Janus-Pro-R1

- Upstream repository: https://github.com/wendell0218/Janus-Pro-R1
- Pinned commit: `0e40b3aa291cb15770f69affc956602c217490af` (HEAD of `main` as of 2026-09-16;
  confirmed via `git ls-remote https://github.com/wendell0218/Janus-Pro-R1.git HEAD main`)
- Paper: Pan et al., "Unlocking Aha Moments via Reinforcement Learning: Advancing
  Collaborative Visual Comprehension and Generation", arXiv:2506.01480
- License: **no `LICENSE` file exists anywhere in the pinned commit** (checked
  `LICENSE`, `LICENSE.md`, `LICENSE.txt`, and per-subfolder variants — all 404 on
  `raw.githubusercontent.com`). Absent an explicit license grant, the code defaults
  to standard copyright (all rights reserved by the authors) under GitHub's terms of
  service; the repository is publicly readable but not proven redistributable beyond
  fair-use research inspection/citation. This vendored copy is retained for reproducible,
  auditable *local* execution only (task T720's smoke), not redistribution, and this
  open point is carried into `reports/T720/claim-check.md`.
  Individual released artifacts referenced by the repo carry their own explicit
  licenses: the `deepseek-ai/Janus-Pro-7B` checkpoint is MIT (code) + DeepSeek Model
  License (weights); the `midbee/Janus-Pro-R1-7B` checkpoint is Apache-2.0; the
  `midbee/Janus-Pro-R1-Data` dataset has no license tag on its HF card (also carried
  as an open point); `OpenGVLab/InternVL2_5-8B` is MIT (+ Apache-2.0 for its
  `internlm2_5-7b-chat` language backbone).
- What was vendored: `janus-sft/`, `janus-rl/`, `inference/` in full (134 files, 3.3MB),
  excluding `.git/` and the top-level `assets/` banner-image directory (51MB of
  marketing images, not used by any training/inference code path).
- What was NOT modified: file contents are byte-identical to the pinned commit.
  Any CompPareto-side adaptation lives in `src/comppareto/adapters/janus_pro_r1/`
  and imports from this vendored tree rather than editing it in place, so the
  upstream-exact code and the adapted smoke harness stay separable.
