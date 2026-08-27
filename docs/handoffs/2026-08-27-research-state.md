# Research state handoff — 2026-08-27

This note is the handoff entry point for the next local planner or remote executor. It summarizes the project state after the first theory audit and the follow-up discussion on data, model choice, and a stronger mathematical target. It was prepared from authorized `main` revision `e1a2e2e`.

The authoritative execution state remains in [`tasks/README.md`](../../tasks/README.md). This note does not authorize a planned task or replace a task contract.

## 1. Research question and current position

The scientific problem is joint post-training of a unified multimodal model when tasks:

- use heterogeneous losses and native optimizers;
- update only partially overlapping parameter blocks;
- retain task-private modules and optimizer state that can compensate after a shared update;
- may draw data from different or model-dependent distributions.

The first validation scope is image understanding plus image generation. Pretraining is out of scope. The current method route is **compensation-aware Pareto post-training**: estimate each task's loss after a fixed private response, lift the resulting task-local signal into the global shared coordinates, and negotiate a shared update.

The current defensible contribution is an optimizer-state-aware UMM method, diagnostic protocol, and compute-matched empirical package. A generic multi-objective bilevel formulation or generic Pareto-stationarity result is not by itself novel.

## 2. Evidence and claim status

| Item | Status | Interpretation |
|---|---|---|
| Shared/private/optimizer-state formulation | Established definition | Distinguishes the operational finite response from an exact regularized best response |
| Schur-complement compensation model | Proven under local PSD and positive private-curvature assumptions | Correct local result; not a global neural-network guarantee |
| Conditional loss-scale invariance | Proven after the correction in `docs/theory/formulation.md` | The attainable gain scales by definition when the trust set is fixed; mixed Adam metrics and fixed absolute epsilons break the claim |
| Deterministic common-descent certificate | Proven for exact same-state gradients | Does not directly certify stochastic, biased, stale, or distribution-shifted estimates |
| T1a deterministic smoke | Verified | Algebraic contracts pass; this is not the full T1b solver gate |
| T1b independent solver/approximation validation | In progress | T110, T120, and T130 are currently authorized |
| Public-model admission | In progress | T210 Show-o2 audit is currently authorized; no joint training is authorized |
| Compensation-aware predictive validity | Hypothesis | Must pass D0 against raw first- and second-order Taylor baselines |
| Joint multimodal performance gain | Hypothesis | No empirical improvement is currently claimed |
| Stochastic safety or convergence theorem | Open theorem target | Existing MOBL results make a generic convergence theorem insufficiently novel |

## 3. Theory audit: what remains unresolved

The current formulation is internally coherent at a fixed state, but it omits four error sources that matter in real post-training:

1. **Finite private adaptation:** a finite native optimizer trajectory need not approximate the exact regularized best response.
2. **Curvature approximation:** diagonal, low-rank, Gauss–Newton, Fisher, and truncated-CG surrogates introduce different errors.
3. **Finite samples:** the Pareto direction is a nonlinear function of stochastic gradients and is generally biased even when every task gradient estimator is unbiased.
4. **Distribution shift:** task data or rollout distributions can depend on time, policy, sampler, model state, or the negotiated update.

Static selectors \(P_i\) are also only a first model. Mixture-of-Transformers or routed experts can make overlap input- and time-dependent, which should be treated as a later stress test rather than silently absorbed into the static proof.

## 4. Recommended upgraded mathematical object

Introduce the task/data-indexed finite-response value

\[
F_{i,t}^{K}(\theta;q_{i,t})=
\mathbb E_{z\sim q_{i,t}}
\left[\ell_i\!\left(P_i\theta,
A_i^K(P_i\theta;\phi_{i,t},\omega_{i,t}),z\right)\right].
\]

For an estimated global task signal \(\widehat h_i\), explicitly decompose

\[
\widehat h_i-h_i
=e_i^{\mathrm{inner}}+e_i^{\mathrm{curv}}
+e_i^{\mathrm{sample}}+e_i^{\mathrm{shift}}.
\]

The preferred theorem target is a **high-probability robust common-descent certificate**, not another generic convergence theorem. A representative target is

\[
F_i(\theta+\eta d)-F_i(\theta)
\le
-\eta\!\left(\gamma_i-\epsilon_i\lVert d\rVert_{M^{-1}}\right)
+\frac{L_i\eta^2}{2}\lVert d\rVert^2,
\]

where \(\gamma_i=-\widehat h_i^\top d\) is the estimated margin and \(\epsilon_i\) upper-bounds the four error terms with specified probability. If every robust margin remains positive and the step satisfies the smoothness bound, every selected task decreases with the stated confidence.

This target is useful only if the bounds are computable or calibratable. A theorem with unknown constants that cannot affect acceptance, sample allocation, or step size should not be the headline result.

### Theory routes, ranked

| Route | Role | Novelty risk | Decision |
|---|---|---|---|
| Finite-response robust Pareto certificate | Main theory route | Medium | Keep |
| Overlap-graph oracle/sample complexity | Stronger optional theory | Medium-high technical risk | Keep as stretch |
| Adaptive task/capability sampling that minimizes certificate width | Data-aware extension | Medium | Merge into the main route after the certificate is defined |
| Decision-dependent/rollout distribution dynamics | E3 extension | High | Park until E1 passes |
| Generic MOBL Pareto convergence | Background/baseline theorem | Very high novelty risk | Do not claim as the main contribution |
| Optimal transport or viability language without a computable algorithm | Mathematical analogy only | High | Do not lead with it |

The most promising mathematical imports are: a posteriori inexact-hypergradient bounds, stochastic multi-gradient bias analysis, decision-dependent stochastic optimization, concentration-based robust constraints, and optimal sample allocation. Viability/safe-set language can clarify the goal, but it should enter only when it produces an executable acceptance or projection rule.

## 5. Mountain-Car-style validation principle

The lesson from [Chebyshev Policies and the Mountain Car Problem](https://arxiv.org/abs/2605.22305) is methodological: build a problem with an exact oracle and report regret to that oracle before relying on a large-model score. The paper analytically solves Mountain Car and exposes a large gap between common RL algorithms and the optimum; its arXiv record currently labels it an ICML 2026 spotlight/oral.

The analogous CompPareto benchmark should have two levels:

1. **Linear-Gaussian post-adaptation oracle.** A shared latent scene generates an image and text target; a shared encoder and two private heads admit an exact post-adaptation Pareto set, exact hypergradients, and regret to the oracle.
2. **Controlled scene-graph renderer.** CLEVR-like factors provide counting, spatial, attribute, and composition slices with exact programs and controlled task/data overlap.

The analytic benchmark must expose raw-gradient conflict, private compensation, partial overlap, finite-response error, and distribution/sample allocation as independently controllable variables. It is not just another random quadratic test; T1b already covers algebraic quadratics.

## 6. Data plan

Data is part of the optimization problem rather than a neutral input. Every dataset manifest should record source, license, capability labels, task role, split unit, hashes, and decontamination checks.

Recommended ladder:

1. linear-Gaussian synthetic data for exact theory;
2. CLEVR/scene-graph controlled data for causal capability slices;
3. natural paired data, using a COCO-like image-caption universe for joint sampling and GQA/TextVQA-style held-out stress tests;
4. preference or online rollout data only in E3.

D0 must separate data-scheduling effects from update-geometry effects. At minimum compare paired semantic batches, independent task batches, proportional sampling, and temperature-balanced sampling under the same estimator and compute budget.

The future data-aware extension may optimize sample counts by task and capability to reduce the widest robust certificate, subject to minimum coverage and bounded divergence from a frozen reference mixture. This is a theorem/algorithm target, not an accepted method yet.

## 7. Model ladder

| Model | Intended role | Current decision |
|---|---|---|
| Show-o2-1.5B | First engineering and D0/E1 path | T210 ready for admission audit |
| UniDDT | Deep shared semantic-path validation | Next admission after Show-o2 |
| SenseNova-U1 | Dynamic/routed architecture stress test | Later admission; static-overlap assumptions must be audited |
| UniAR | More homogeneous AR-objective boundary control | Later admission |
| JanusFlow | Predeclared fallback | Use only after an admission failure under the frozen replacement order |

SenseNova-U1.5 is not an initial dependency until its relevant post-training pipeline is publicly reproducible. Compute availability—up to 64 H20 GPUs—is sufficient for post-training, but no scale run should precede the theory/synthetic, model-admission, and D0 gates.

## 8. Closest mathematical literature and novelty boundary

All entries below were re-opened at their primary paper or proceedings page on 2026-08-27.

| Work | What it contributes | Consequence for this project | Verification |
|---|---|---|---|
| [Multi-Objective Bilevel Learning](https://arxiv.org/abs/2511.07824) | Deterministic/stochastic finite-time Pareto-stationarity through WC-MHGD | Generic MOBL convergence is already occupied; compare against its normalized Chebyshev route | verified via arXiv |
| [Analyzing Inexact Hypergradients for Bilevel Learning](https://arxiv.org/abs/2301.04764) | A priori and computable a posteriori hypergradient error bounds | Best starting point for \(e^{\mathrm{inner}}+e^{\mathrm{curv}}\) | verified via arXiv; journal acceptance noted there |
| [The stochastic multi-gradient algorithm](https://arxiv.org/abs/1907.04472) | Shows that the stochastic multi-gradient direction can be biased even with unbiased task gradients | A minibatch MGDA certificate cannot be treated as deterministic | verified via arXiv |
| [Bilevel Optimization with Coupled Decision-Dependent Distributions](https://proceedings.mlr.press/v202/lu23a.html) | Bilevel optimization where upper/lower data distributions depend on the other level's decision | Direct prior-work threat and tool for data-aware formulation | verified via PMLR/ICML 2023 |
| [Decision-Dependent Stochastic Optimization: The Role of Distribution Dynamics](https://arxiv.org/abs/2503.07324) | Models nonlinear feedback between decisions and evolving distributions | Relevant to later on-policy/rollout shift, not needed for the first static-data theorem | verified via arXiv |

## 9. Experiment sequence

The recommended order after the currently authorized work is:

```text
T1b independent references + Show-o2 admission
  -> analytic post-adaptation oracle
  -> finite-response/error-bound calibration
  -> controlled scene-graph benchmark and data-manifest freeze
  -> D0 predictive-validity study
  -> E1 Show-o2 pilot
  -> E2 architecture transfer
  -> E3 preference/RL with distribution shift
```

The first three steps can run mostly on CPU or a small GPU allocation. No method should proceed to E1 merely because the analytic benchmark is favorable; D0 remains the gate that tests whether compensation-aware quantities predict realized large-model changes.

## 10. Immediate agent handoff

The following tasks are already authorized on `main` and can be assigned immediately:

| Task | Branch | First deliverable | GPU policy |
|---|---|---|---|
| T110 | `agent/T110-overlap-family` | Family dimensions, seeds, tolerances, expected CPU cost | CPU |
| T120 | `agent/T120-independent-kkt-reference` | Independent solver choice and residual checks | CPU |
| T130 | `agent/T130-indefinite-trust-region` | Counterexample families and rejection criteria | CPU |
| T210 | `agent/T210-showo2-admission` | Official revision/license/component/training-interface audit | No expensive GPU work before first report |

Each executor must pull `git@github.com:alexlovecoding/UMM_RL.git`, branch from the latest authorized `main`, obey the task's `allowed_paths`, and return `awaiting_review`; only the local review side can accept work.

Suggested new tasks for a future local planner—**not yet authorized by this note**—are:

1. formalize and audit the finite-response robust Pareto theorem target;
2. implement the linear-Gaussian exact-oracle benchmark;
3. freeze the dataset/capability manifest and sampling controls;
4. prepare a model/data admission matrix extending T210;
5. map routed/dynamic parameter overlap before SenseNova-U1 experiments.

## 11. Non-negotiable review questions

- Is the private response a legitimate part of deployment/training, or only extra compute given to the proposed method?
- Are the same \(A_i^K\), data, persistent updates, and tuning budgets available to the strongest baselines?
- Can every claimed safety margin be computed or empirically calibrated?
- Does the method beat raw Taylor prediction in D0 and tuned scalarization in E1?
- Are gains from data balance rather than update geometry?
- Does partial overlap provide measurable benefit beyond general MOBL?
- Are all failed seeds, model admissions, and capability slices retained?

## 12. Recommended next local decision

Do not rename or broaden the project yet. Keep “CompPareto” provisional and complete the current T1b/T210 work. In parallel, the local planning side should convert the first two suggested tasks above into formal task contracts. The main theory contribution should be promoted only if the robust finite-response certificate is both nontrivial relative to MOBL and operationally tighter than a vacuous worst-case bound.
