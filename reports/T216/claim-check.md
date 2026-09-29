# Claim Check — T216

Status of each claim after the K=1 sweep (`runs/alternating-showo2-v1/`, manifest `status: pass`).

## Verdict summary

**The gate claim is NOT supported.** "A negotiator supports the gate" is a K=3
result per the task contract. This run is K=1 only. K=1 is a mandatory gate that
must be passed *before* K=3; passing it is necessary, not sufficient.

## Claims and verdicts

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | All four negotiators (raw_sum, normalized_sum, pcgrad, mgda) produce finite directions and finite losses on both tasks at K=1 | **SUPPORTED** | `metrics.json` → `numerical.nonfinite_mandatory_rows: 0` of `total_rows: 60`; all 62 raw rows have `finite: true` |
| 2 | The `persistent_updates` count is 0 — no protocol step leaves a permanent mark on the model | **SUPPORTED** | `metrics.json` → `persistent_updates: 0`, `snapshot_restore.failed: 0`; byte-exact restore asserted with `atol=0.0, rtol=0.0` |
| 3 | Every mandatory row has its P1 control baseline | **SUPPORTED** | `metrics.json` → `attribution.mandatory_rows_missing_control: 0` |
| 4 | All four protocols ran to completion | **SUPPORTED** | `metrics.json` → `protocols.{simultaneous,private_only_control,shared_then_private,private_then_shared_commit}_measured: true` |
| 5 | The RNG snapshot/restore mechanism is byte-exact | **SUPPORTED** (unit-level) | `tests/adapters/showo2_alternating/` — full suite `331 passed`; the GPU run's `persistent_updates: 0` is consistent with it |
| 6 | The attribution identity `delta_total = delta_private + delta_controlled` holds | **HOLDS, BUT IS NOT EVIDENCE** | Recomputed over all 60 mandatory rows: 0 violations. However this is an *algebraic identity* of how the three quantities are defined (`delta_controlled = loss_after - control_loss`, `delta_private = control_loss - loss_before`), so it cannot fail and cannot validate the attribution. It is reported as a consistency check only. |
| 7 | Some ordering (SP or PS) changes the outcome relative to simultaneous update | **NOT SUPPORTED at K=1** | P0: 0/24 rows with non-zero `delta_controlled`. P2: 8/12 und rows at -0.03125. P3: 3/6 und rows at -0.03125. gen: 0/30. All differences are 0–1 bf16 ULP (ULP at 7.59375 is 0.03125). See the precision caveat below. |
| 8 | Any negotiator is preferable to the others | **NOT SUPPORTED** | All four negotiators fall in the same 0/-1-ULP band; no seed replication exists at K=1, so no variance estimate and no ranking |
| 9 | **Gate claim: a negotiator supports the gate** | **NOT SUPPORTED — requires K=3** | Contract: K=3 only if the K=1 gate passes. K=1 passed as a validity gate; the gate decision itself is not yet measurable |

## Claims this task explicitly does NOT make

- That any negotiator "supports the gate" — gate passage requires K=3, not K=1.
- That SP or PS ordering shows "faster convergence" — this is a K=1 diagnostic, not a convergence study.
- That any specific direction is preferred — all four are measured neutrally and are
  indistinguishable at K=1 within bf16 readout resolution.
- That the all-zero P0 result means the shared gradient has no effect — the step-scale
  grid puts the shared displacement below bf16 parameter resolution, so the null result
  is confounded with precision (see `result-summary.md`).
- That T215's failures are "fixed" in general — T216 only documents that the three
  specific mechanisms (create_graph, RNG re-seed, finite-difference) were not used here.
- Anything at the checkpoint or dataset level: 0 persistent updates means the model was
  restored bit-exactly between every row, so this run produced no trained weights.

## Evidence locations

| Claim | Evidence |
|-------|----------|
| 1, 3, 4 | `runs/alternating-showo2-v1/metrics.json` |
| 1, 6, 7, 8 | `runs/alternating-showo2-v1/raw_rows.json` (62 rows: 24 P0, 24 P2, 12 P3, 2 P1) |
| 2, 5 | `metrics.json` + `tests/adapters/showo2_alternating/test_protocols.py` |
| run provenance | `runs/alternating-showo2-v1/manifest.json` (`execution_revision: 28b8df07…`, `config_sha256: d1c7f380…`) |
| artifact hashes | `configs/alternating/showo2/artifact-verification.json` (`failed: 0`, 2/2 passed) |
