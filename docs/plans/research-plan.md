# Research plan

## Objective

Establish whether compensation-aware shared updates improve joint post-training of image understanding and image generation, and whether the effect transfers across unified-model architectures without pretraining.

## Claim ladder

| Claim | Evidence required | Gate |
|---|---|---|
| C0a: deterministic algebraic contracts are correct | Fixed synthetic smoke tests | T1a |
| C0b: solver and approximations match independent references | Random overlap families, direct solve, and approximation curves | T1b |
| C1: compensation-aware quantities better describe realized conflict | Held-out predicted-vs-realized update study | D0 |
| C2: CompPareto improves the worst task over strong joint-training baselines | Three-seed Show-o2 pilot | E1 |
| C3: the effect is not architecture-specific | UniDDT and SenseNova-U1/UniAR transfer | E2 |
| C4: the method supports heterogeneous native post-training | Understanding DPO/OPD plus generation Flow-DPO/GRPO | E3 |

Claims are promoted only in order. C2 is not pursued at scale if C1 fails.

## Stage T1a — deterministic algebraic smoke

The executable T1a gate covers exact private elimination at a fixed step, legal block selectors, deterministic trust-region attainable gain, conditional rescaling of a two-task negotiation, a two-gradient common-descent case, and rejection of non-positive private curvature. It is a smoke test, not the full solver claim.

## Stage T1b — solver and approximation validation

Construct deterministic and stochastic two-task quadratics with:

- known shared/private Hessian blocks;
- disjoint, partially overlapping, and fully shared block graphs;
- controllable gradient angle, scale ratio, coupling rank, and private damping;
- both PSD and deliberately indefinite local curvature.

Tests:

1. Schur-complement solution versus direct joint minimization: relative error below `1e-3`.
2. Loss rescaling by `1e-3` to `1e3`: normalized solution invariant within numerical tolerance.
3. Convex-hull certificate versus brute-force common-descent feasibility.
4. Trust-region rejection on indefinite counterexamples.
5. Approximation error for exact CG, one-/three-/five-step unrolling, diagonal, and low-rank variants.

## Stage E0 — model and evaluation reproduction

For Show-o2-1.5B, reproduce one understanding path and one generation path before modifying training. Record checkpoint hash, library versions, prompts, sampler parameters, and metric code revision.

Model admission rules:

- public checkpoint and executable inference;
- public access to the relevant shared and private parameters;
- legal ability to modify and evaluate locally;
- no undisclosed pretrained component required by the selected task path;
- one understanding and one generation metric reproducible within a tolerance frozen before comparison.

## Stage D0 — conflict cartography and predictive validity

Use paired image–text content so that understanding and generation batches share semantic factors. For each candidate shared block, log:

- task gradient norms and norm ratios;
- raw gradient cosine and sign agreement;
- stop-gradient post-adaptation gradient;
- unrolled hypergradient;
- estimated curvature coupling and convex-hull distance;
- predicted task loss changes;
- realized fresh-batch changes after the shared step and private response.

Split diagnostic steps into calibration and held-out evaluation sets. Compare prediction error, sign precision/recall for “both tasks improve,” Brier score, and Spearman rank correlation. The target is at least `0.5` held-out Spearman and a statistically supported improvement over raw cosine; final thresholds are frozen after a variance-only dry run.

The split unit is a tuple of `(base-checkpoint snapshot, optimizer-step window, semantic batch group)`, not an individual image or prompt. All records from one tuple remain in one split to prevent near-duplicate gradients leaking between calibration and test.

Predictive baselines are:

1. raw cosine as a descriptive conflict signal, not a loss-change predictor;
2. raw first-order Taylor prediction \(g_i^\top d\);
3. raw damped second-order prediction \(g_i^\top d+\tfrac12d^\top H_i d\);
4. post-adaptation stop-gradient prediction;
5. equal-compute finite unroll;
6. implicit/Schur prediction on the diagnostic subset.

D0 passes only if compensation-aware prediction improves over the strongest raw Taylor baseline, not merely over cosine.

## Stage E1 — Show-o2 controlled pilot

### Tasks

- Understanding: caption/VQA autoregressive cross-entropy.
- Generation: text-to-image flow matching.

### Factors

- shared depth: 25%, 50%, 100% of eligible backbone blocks;
- compensation estimator: none, stop-gradient, unroll-1, unroll-3, implicit/CG diagnostic;
- three fixed data-order seeds for exploration, followed by the power-selected confirmatory seed count (default minimum five);
- matched examples/tokens, optimizer steps, and wall-clock accounting.

### Primary outcome

For task metric \(m_i\), set \(\sigma_i=+1\) when higher is better and \(\sigma_i=-1\) when lower is better. Define direction-corrected normalized gain relative to the base and single-task oracle:

\[
G_i=\frac{\sigma_i(m_i-m_i^{base})}
{\max\left(|m_i^{ST}-m_i^{base}|,\kappa_i\right)}.
\]

Report worst-task gain \(\min_iG_i\), fraction of single-task gain retained, Pareto hypervolume, and negative-transfer rate over preregistered submetrics.

Here \(\kappa_i\) is twice the E0 repeated-evaluation standard deviation in metric units. If single-task headroom is below \(\kappa_i\), the ratio is marked unstable and the headline result uses absolute non-inferiority/improvement for that task instead.

Three seeds are exploratory. A variance-only dry run determines confirmatory seed count through a preregistered power calculation, with five seeds as the default minimum. The confidence interval is hierarchical over training seed and evaluation example; an example-only bootstrap is never used to claim training robustness.

Provisional success criterion: the confirmatory 95% interval of worst-task gain is positive (or meets the preregistered absolute margin for low-headroom tasks), at least 70% of each stable single-task gain is retained, and hypervolume exceeds the best budget-matched scalarization by at least 5%.

## Stage E2 — architecture transfer

1. **UniDDT:** primary deep-sharing validation because the NoisyViT + LLM path is shared while the diffusion decoder is private.
2. **SenseNova-U1:** native pixel-space/MoT stress test using the released full-parameter fine-tuning path.
3. **UniAR:** homogeneous AR-objective boundary control. The prediction is a smaller benefit if objective heterogeneity is a major mechanism.
4. **SenseNova-U1.5:** add only after its announced SFT/RL/MOPD training pipeline is public and reproducible; it is not a dependency for the first claims.

E2 passes if the worst-task result is positive on at least two architecture families and no reported model is hidden because it failed.

Each model has an independent admission record under `runs/admission-<model>/` containing:

- official repository URL and exact commit;
- model/checkpoint identifier and hash;
- license and redistribution constraints;
- train/evaluation entry points actually exercised;
- required external tokenizers, autoencoders, reward services, and missing assets;
- one understanding and one generation smoke-test result;
- reproduction tolerance and pass/fail decision.

A failed admission is reported and excludes that model from method comparison; it does not permit replacing the model after seeing CompPareto results. The replacement order is preregistered as UniDDT, SenseNova-U1, UniAR, then JanusFlow.

## Stage E3 — heterogeneous preference and reward updates

After E1 passes, combine an understanding-side preference objective (DPO or OPD) with a generation-side Flow-DPO or GRPO objective. The operational value function uses finite native optimizer steps because an exact private minimizer is not meaningful for online rollouts.

Measure reward overoptimization, KL drift, gradient estimator variance, acceptance rate, and method overhead in addition to task metrics.

## Capability-aligned evaluation

| Capability | Understanding | Generation |
|---|---|---|
| Counting | TallyQA, GQA-count | GenEval-count |
| Spatial relation | VSR, GQA-spatial | GenEval-position, T2I-CompBench spatial |
| OCR/text | TextVQA, OCR-VQA | text rendering, DPG/WISE text slices |
| Attribute/composition | GQA, SEEDBench slices | GenEval and T2I-CompBench slices |

General understanding adds MME, MMMU, MMStar, and AI2D. Generation adds DPG-Bench, HPSv2, ImageReward/PickScore, with metric versions and prompts pinned.

## Compute plan for up to 64 H20 GPUs

Compute is a scheduling constraint, not the research contribution.

- Wave 0: 4–8 GPUs for reproduction and diagnostic instrumentation.
- Wave 1: up to 24 GPUs for three parallel Show-o2 seeds plus evaluation workers.
- Wave 2: 16–32 GPUs per deep-sharing model, with at most two scale runs active concurrently.
- Reserve 8 GPUs for generation evaluation/reward services and failed-run diagnosis.

Before scale-up, one short job must measure peak memory, samples/sec, communication overhead, and estimator overhead. Batch size is then selected from measured capacity, not guessed from GPU count.

## Stopping rules

- Stop after D0 if compensation-aware signals do not outperform raw diagnostics.
- Stop a model family if its official reproduction gate fails; record the failure and continue with another predeclared model.
- Do not run E3 until E1 passes.
- Reframe the paper as a diagnostic result if the certificate is predictive but the optimizer does not beat scalarization.
