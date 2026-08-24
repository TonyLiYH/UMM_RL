# Related work and novelty audit

Checked on 2026-08-24. Links point to primary papers or official repositories.

## 1. Unified multimodal architectures

| Model | Coupling pattern | Public training status relevant to this project | Role here |
|---|---|---|---|
| [Show-o2](https://arxiv.org/abs/2506.15564) / [code](https://github.com/showlab/Show-o) | Shared Transformer with autoregressive and flow heads | Training code and 1.5B/7B checkpoints public | First executable pilot |
| [UniDDT](https://arxiv.org/abs/2606.16255) / [code](https://github.com/MCG-NJU/UniDDT) | Shared NoisyViT + LLM semantic path, private diffusion decoder | Checkpoints/configuration and training implementation public | Main deep-sharing validation after reproduction audit |
| [SenseNova-U1](https://arxiv.org/abs/2605.12500) / [code](https://github.com/OpenSenseNova/SenseNova-U1) | Native pixel-space unification with mixture of Transformers | U1 full-parameter fine-tuning entry released; U1.5 SFT/RL/MOPD pipeline announced but not yet the initial dependency | Cross-architecture stress test using released U1 path |
| [UniAR](https://arxiv.org/abs/2606.18249) / [code](https://github.com/ShareLab-SII/UniAR) | One AR backbone and shared discrete visual tokenizer; diffusion decoder reconstructs images | AR/GRPO training public; visual-decoder training remains unreleased | More homogeneous-objective boundary control |
| [Representation Forcing](https://arxiv.org/abs/2605.31604) | Predict representation tokens before pixel diffusion in one backbone | Paper establishes deeper semantic–pixel alignment; not an initial execution dependency | Architectural motivation |
| [JanusFlow](https://arxiv.org/abs/2411.07975) / [Janus code](https://github.com/deepseek-ai/Janus) | AR understanding plus rectified flow with partly decoupled visual paths | Public implementation | Secondary baseline/adaptor candidate |
| [Lumina-DiMOO](https://arxiv.org/abs/2510.06308) | All-discrete diffusion across modalities | Availability must be audited before use | Unified-objective boundary hypothesis |

The latest release relevant to the long-term model suite is SenseNova-U1.5-8B-MoT (announced 2026-08-20). Its full SFT/RL/MOPD training pipeline is stated as forthcoming in the official repository, so the first plan depends only on the released U1 full-parameter path.

## 2. Evidence that cross-task transfer and conflict are real

- [Transferability of visual understanding and generation](https://arxiv.org/abs/2607.04423) reports that transfer strength depends strongly on architecture and shared visual pathways.
- [A diagnostic study of DPO in unified models](https://arxiv.org/abs/2603.17044) finds near-orthogonal task gradients but large magnitude imbalance, cautioning against treating negative cosine as the only conflict mechanism.
- [UNO](https://arxiv.org/abs/2605.05781) uses understanding-oriented supervision to improve generation.
- [UniMRG](https://arxiv.org/abs/2601.21406) uses generated intermediate representations to improve understanding and generation.
- [R3](https://arxiv.org/abs/2602.15772), [UniReasoner](https://arxiv.org/abs/2605.04040), and [UniCorn](https://arxiv.org/abs/2601.03193) construct generate–understand feedback loops rather than a general shared-update optimizer.
- Qwen-Image-2.0-RL multi-teacher on-policy distillation provides a strong capability-integration baseline, but it assumes expert trajectories and distillation rather than directly solving partial-overlap Pareto post-training.

## 3. Multi-task optimization

Required baselines include [MGDA](https://arxiv.org/abs/1810.04650), [PCGrad](https://arxiv.org/abs/2001.06782), [CAGrad](https://arxiv.org/abs/2110.14048), [Nash-MTL](https://arxiv.org/abs/2202.01017), [FAMO](https://arxiv.org/abs/2306.03792), and [ConFIG](https://arxiv.org/abs/2408.11104).

[In Defense of Unitary Scalarization](https://arxiv.org/abs/2201.04122) shows that tuned scalarization is often a strong competitor. It therefore receives a budget-matched hyperparameter search and is not treated as a weak default.

The closest novelty threat is [ConsMTL](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf), which exploits task-specific parameters to improve consistency of shared representations. [Recon](https://arxiv.org/abs/2302.11289) instead turns conflict-prone shared layers into task-specific ones.

## 4. Bilevel, meta-learning, and personalized-learning threats

The shared/private value-function idea is not new by itself:

- [Multi-Objective Meta Learning](https://proceedings.neurips.cc/paper/2021/file/b23975176653284f1f7356ba5539cfcb-Paper.pdf) formulates meta-learning as multi-objective bilevel optimization.
- [FORUM](https://arxiv.org/abs/2401.09257) uses a value-function reformulation and first-order multi-gradient method for multi-objective bilevel optimization.
- [gMOBA](https://arxiv.org/abs/2406.05455) targets Pareto stationarity in multi-objective bilevel problems.
- [Multi-Objective Bilevel Learning / WC-MHGD](https://arxiv.org/abs/2511.07824) provides deterministic and stochastic finite-time Pareto-stationarity guarantees and preference-guided Pareto exploration.
- [Partially Personalized Federated Learning](https://arxiv.org/abs/2305.18285) explicitly splits global and client-private parameters and studies local updates.
- [HyperFormer](https://arxiv.org/abs/2106.04489) generates task-specific adapters from a shared hypernetwork, while [VL-Adapter](https://openaccess.thecvf.com/content/CVPR2022/papers/Sung_VL-Adapter_Parameter-Efficient_Transfer_Learning_for_Vision-and-Language_Tasks_CVPR_2022_paper.pdf) compares separate, half-shared, and fully shared adapters for vision-language tasks. [Progressive Task-Specific Adaptation](https://arxiv.org/abs/2509.19602) moves from shared early adapters to task-specific later adapters. These methods choose or generate the shared/private parameterization; they do not by themselves estimate the post-private-response shared update.

### Formal comparison

| Method family | Inner/private object | Multiple upper objectives | Finite native optimizer state | Partial active shared blocks | Loss-unit normalization | Common-descent/Pareto result | UMM heterogeneous post-training |
|---|---|---:|---:|---:|---:|---:|---:|
| MGDA/PCGrad/CAGrad | None | Yes | No | Usually no | Method-dependent | Yes/partial | No |
| ConsMTL | Task-specific parameter feedback | Yes | No explicit native inner trajectory | Shared representation focus | No retained-solo-gain ratio | Consistency objective | No |
| MOML | Task adaptation | Yes | Gradient-based meta inner loop | Encodable | No | Multi-objective meta solution | No |
| FORUM/gMOBA | Lower-level optimizer/value function | Yes | Approximation algorithms, not arbitrary task-native state | Encodable | No retained-solo-gain ratio | Pareto stationarity | No |
| WC-MHGD | Lower optimum coupled to upper objectives | Yes | Stochastic hypergradient oracle | Encodable | Weighted Chebyshev preference | Finite-time Pareto stationarity | No |
| Partially personalized FL | Per-client private parameters/local steps | Global aggregate is primary | Yes | Global/private split | No | Convergence/client benefits | No |
| HyperFormer/VL-Adapter/progressive adapters | Task-conditioned or partly shared adapters | Joint task training | Adapter updates | Architectural shared/private pattern | No | No general Pareto certificate | Mostly language/VL/dense vision |
| CompPareto proposal | One task-indexed private response per objective | Yes | Explicit \((\phi_i,\omega_i,A_i^K)\) | Explicit overlap graph | Conditional retained-solo-gain ratio | Deterministic certificate; stochastic diagnostic only | Yes |

The last row is **not yet a proof of mathematical novelty**. Task-indexed private variables can be concatenated into one block-diagonal lower variable, and selectors can be encoded as masks in a general bilevel formulation. Therefore the current defensible contribution is a problem-specific algorithmic and empirical package:

1. operationalizing optimizer-state-aware finite native responses for heterogeneous UMM tasks;
2. negotiating normalized retained single-task gain over an explicit overlap graph;
3. testing whether compensation-aware geometry predicts realized multimodal interference better than raw local models;
4. providing compute-matched protocols for private-response estimators.

A claim of new optimization theory requires an additional result not inherited from general MOBLO—for example, a strictly cheaper block-overlap oracle-complexity result or a finite-step error bound tied to the overlap graph. Until such a result is proved, the project must not market the formulation alone as a new general theory.

### Retained gain versus Chebyshev and Nash negotiation

The retained-gain objective is best understood as a **specialized normalized Chebyshev max-min objective**, not an unrelated new bargaining principle. Its task normalization point is the improvement each task can attain alone inside the same local trust region. Weighted Chebyshev methods instead take externally specified preferences/reference scaling, while Nash-MTL uses a bargaining solution derived from task utilities. The research question is whether the single-task attainable-gain normalization is better calibrated for heterogeneous UMM losses; this must be tested against WC-MHGD/normalized Chebyshev and Nash-style negotiation using the same compensated gradients and the same \(A_i^K\).

## 5. Expert merging and distillation

Independent expert training followed by model merging or distillation is a practical alternative. Baselines should cover task arithmetic/TIES-style merging, a trust-region merge such as [TATR](https://arxiv.org/abs/2501.15065), and a Qwen-Image-style multi-teacher on-policy distillation path when expert trajectories are available.

These approaches answer “how to combine trained experts.” CompPareto targets “how to negotiate a shared update while each task retains its native training and private response.”

## 6. Novelty boundary

The project is not novel if implemented as loss weighting plus PCGrad, nor merely as MGDA over standard bilevel hypergradients. Its defensible systems/method contribution requires all of the following:

1. an explicit finite-step or implicit private-response value function;
2. parameter-block overlap rather than an assumed fully shared vector;
3. conditionally loss-scale-invariant retained single-task gain;
4. a measurable common-descent/safety certificate;
5. evidence that compensation-aware diagnostics are more predictive than raw gradients.

The decisive comparison is not CompPareto versus raw MGDA. It is: **under equal compute and the identical finite private response \(A_i^K\), does optimizer-state-aware compensation plus attainable-gain normalization predict or optimize joint UMM changes better than general MOBLO/MGDA, normalized Chebyshev, and Nash negotiation?**

## 7. Reviewer questions that must remain visible

- Is private adaptation a real capability of the deployed model, or an artificial extra compute advantage?
- Does the benefit remain after equalizing total task-specific and shared optimizer steps?
- Is the Schur-complement approximation useful outside a PSD local surrogate?
- Does the method outperform a scalarization tuned with the same total search budget?
- Are gains due to task-balanced data rather than update geometry?
- Does “partial overlap” matter empirically, or can a representation-level method explain the same result more simply?
