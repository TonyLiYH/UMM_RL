# T1a synthetic algebraic smoke

- Date: 2026-08-24
- Command: `bash scripts/run_t1.sh`
- Configuration: `configs/t1_synthetic.json`
- Result: pass
- Tests: 13 passed
- Exact-elimination absolute error: `2.7755575615628914e-17`
- Conditional-rescaling maximum error: `5.684341886080802e-14`
- Common-descent maximum directional derivative: `-0.5`
- Indefinite private curvature: rejected as required
- Trust-region attainable gain: `0.875` (expected `0.875`)
- Independent task-rescaling negotiation solution error: `1.8762769116165146e-14`
- Selector validation: illegal mixing/repeated selectors rejected
- Code revision under test: `a6416361c3837128a9218ddb66cfc09b5256433b` with `dirty=false`

Scope: deterministic algebraic smoke only. It does not cover random overlap families, independent direct solvers, overall-indefinite trust-region acceptance, or approximation error curves. No GPU, model checkpoint, or multimodal dataset was used.
