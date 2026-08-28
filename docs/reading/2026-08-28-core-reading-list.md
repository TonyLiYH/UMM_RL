# Core reading list: unified multimodal Pareto post-training

Updated: 2026-08-28

This is a personal reading TODO for the six papers most relevant to the current CompPareto research question. Paper existence, current version, and venue/status statements were checked against primary paper pages on 2026-08-28. A paper's own claims remain claims to audit rather than accepted project facts.

## Completion rule

Mark a paper complete after reading its problem formulation, core method or theorem, experimental protocol, and limitations, and after writing a short note using the template at the end of this file.

## Recommended order

- [ ] **1. Transferability Between Understanding and Generation in Unified Multimodal Models**
  - Paper: [arXiv:2607.04423](https://arxiv.org/abs/2607.04423)
  - Status: ECCV 2026 acceptance is stated on the arXiv record.
  - Why read: Direct empirical study of when understanding and generation transfer across unified multimodal architectures.
  - Focus: shared-backbone structure, unified visual encoders, capability-level transfer, and the distinction between conflict and positive transfer.
  - Question for CompPareto: Should task interaction be conditioned on architecture, capability, and parameter block instead of represented by one global conflict score?

- [ ] **2. Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters**
  - Paper: [CVPR 2025 open-access paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf)
  - Status: CVPR 2025 proceedings paper.
  - Why read: The closest established method using both shared and task-specific parameters to mitigate multi-task conflict.
  - Focus: upper-level gradient aggregation, lower-level gradient alignment, compute accounting, and the semantics of task-specific adaptation.
  - Question for CompPareto: What remains distinct after matching ConsMTL's private-parameter updates, data, and compute budget?

- [ ] **3. Multi-Objective Bilevel Learning**
  - Paper: [arXiv:2511.07824](https://arxiv.org/abs/2511.07824)
  - Status: 2025 arXiv preprint; treat venue and peer-review status separately unless independently verified.
  - Why read: The closest general optimization formulation, with deterministic and stochastic finite-time Pareto-stationarity results through WC-MHGD.
  - Focus: lower-level solution assumptions, stochastic hypergradient oracle, weighted Chebyshev objective, oracle complexity, and Pareto-front exploration.
  - Question for CompPareto: Which result genuinely requires finite task-native optimizer trajectories, optimizer state, or an explicit overlap graph and is therefore not inherited from general MOBL?

- [ ] **4. Analyzing Inexact Hypergradients for Bilevel Learning**
  - Paper: [arXiv:2301.04764](https://arxiv.org/abs/2301.04764)
  - Status: The arXiv record states acceptance in the *IMA Journal of Applied Mathematics*.
  - Why read: Provides a unified treatment of approximate implicit and unrolled hypergradients, including a priori and computable a posteriori error bounds.
  - Focus: residual-based bounds, lower-level inexactness, differentiation error, computable constants, and conditions required by the analysis.
  - Question for CompPareto: Can the bound be adapted to a finite optimizer-state response and propagated into a simultaneous multi-task descent margin?

- [ ] **5. Efficient Hessian-Free Methods for Multi-Objective Bilevel Optimization with Nonconvex Lower Level**
  - Paper: [arXiv:2608.12704](https://arxiv.org/abs/2608.12704)
  - Status: August 2026 arXiv preprint; recent and not treated as settled peer-reviewed evidence.
  - Why read: A direct novelty threat covering nonconvex lower levels, Moreau-envelope reformulation, single-loop Hessian-free optimization, and stochastic MOBL.
  - Focus: Moreau-envelope assumptions, smooth weighted Tchebycheff scalarization, convergence measure, estimator cost, and experimental scope.
  - Question for CompPareto: Does our target theorem add a computable finite-response certificate or graph-localized complexity result beyond nonconvex Hessian-free MOBL?

- [ ] **6. Show-o2: Improved Native Unified Multimodal Models**
  - Paper: [arXiv:2506.15564](https://arxiv.org/abs/2506.15564)
  - Code: [official Show-o repository](https://github.com/showlab/Show-o)
  - Status: NeurIPS 2025 is stated on the arXiv record; code and checkpoints are linked there.
  - Why read: The first planned experimental substrate, combining autoregressive understanding with flow-matching generation in a native unified model.
  - Focus: exact parameter sharing, language and flow heads, visual pathways, two-stage training, released training interfaces, and metric reproduction.
  - Question for CompPareto: Which parameters and optimizer states are truly shared or private, and which finite private response can be executed reversibly and fairly?

## Per-paper note template

Create a note under `docs/reading/notes/` with these fields:

```text
# <paper title>

- Read date:
- Version/revision:
- Research question:
- Core method or theorem:
- Assumptions:
- Evidence and experimental scope:
- Main limitation:
- Relation to CompPareto:
- Novelty threat or opportunity:
- Baseline, theorem, or experiment to import:
- Open questions:
```

## Expected synthesis after all six

After completing the list, write a one-page synthesis answering:

1. Which task interactions are architecture- or capability-dependent?
2. What does ConsMTL already obtain from task-specific parameters?
3. What parts of CompPareto reduce to general MOBL or normalized Chebyshev optimization?
4. Which finite-response error terms can be bounded from observed residuals?
5. What additional result survives the 2026 nonconvex Hessian-free MOBL novelty threat?
6. What exact shared/private block map and reversible response does Show-o2 support?
