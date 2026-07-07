# Sub-Saturation Admission-Delay Floor for the Occupancy-Aware Estimators (Design)

Date: 2026-07-06
Branch: `feat/edpp-estimator-validation` (follow-up to the utilization sweep)
Related: `docs/superpowers/specs/2026-07-06-edpp-utilization-sweep-design.md` and
`campaigns/edpp-study/FINDINGS.md` "Utilization sweep" (the finding this fixes);
`docs/superpowers/specs/2026-07-02-edpp-admission-estimator-design.md` (Stage C, the estimators).

## 1. Goal & scope

The utilization sweep showed that below the saturation cliff the occupancy-aware admission estimators
(`fluid`, `rollforward`) predict **0** — their free-slot early-return (`sim/admission_estimator.go:66`
and `:84`) fires whenever a batch slot + KV are free, modelling admission as instantaneous. But the next
`FormBatch` only runs after the current in-progress decode step finishes, so realized admission delay has
a floor of ≈ one decode iteration (`T_iter`, confirmed empirically to track ITL: ~29 ms at ρ̂=0.5 rising
with batch fill). This change adds that floor so the estimators track sub-saturation admission delay.

Scope:
- Add a lower bound of one `T_iter` to the **deployable occupancy/aggregate estimators** `little`,
  `fluid`, `rollforward` (and, since they share impls, `fluid_oracle`/`rollforward_oracle`).
- **`waiting` is deliberately NOT floored.** It is the occupancy-blind strawman baseline; leaving it
  unchanged preserves the "default `--edpp-tadm-estimator waiting` → decider byte-identical" invariant
  held across Stages A–C, so no prior finding or default-behavior baseline is disturbed. The floor is a
  physics-aware refinement that the naive baseline correctly does not receive.
- Estimator code only (`sim/admission_estimator.go`) + affected unit tests + a validation re-run and
  doc update. No routing-rule, snapshot, or trace changes. `AdmissionContext.TIter` already carries the
  per-iteration time the floor needs.

Non-goal: exactness of the floor. One `T_iter` is a first-order term (it slightly over-predicts at high
load, where the measured floor is ~0.7·ITL); a calibrated/residual floor is out of scope (YAGNI — the
floor is routing-irrelevant, below).

## 2. The change

A shared lower-bound helper, applied at every return of the three floored estimators:

```go
// flooredTAdm lower-bounds an admission-delay estimate by one iteration: even with a
// free slot, a request waits for the current step to finish before the next FormBatch
// admits it (~T_iter). No-op when TIter is unavailable (≤0) or already exceeded.
func flooredTAdm(est float64, ctx AdmissionContext) float64 {
    if ctx.TIter > est {
        return ctx.TIter
    }
    return est
}
```

Applied by wrapping each `return <value>` in `littleEstimator`, `fluidEstimator`, and
`rollforwardEstimator` as `return flooredTAdm(<value>, ctx)` — including the free-slot early-returns
(which become `T_iter` instead of `0`) and the saturated/fallback returns (unchanged in practice, since
those already exceed `T_iter`). `waitingEstimator.EstimateTAdm` is left exactly as-is.

Because the floor is a lower bound, it is a **no-op above saturation** (slot-wait ≫ `T_iter`), so the
Stage C overload numbers (57×→1.16×) are unchanged; it only lifts the sub-saturation estimates from ~0
to `T_iter`.

## 3. Why this is routing-irrelevant (and therefore safe)

The floor (~30–47 ms) is ≪ τ_ttft = 2 s, so it does not move the SLO virtual queue `z_ttft` across its
threshold, and `T̂` enters the decision only as `z_ttft·(ttft_p − ttft_d)`. Adding a per-pool `T_iter`
floor to both sides shifts the difference only by the (small) decode-vs-prefill `T_iter` gap, far below
any level that changes a routing decision. This is a fidelity improvement to the estimator, not a change
to routing behavior — and `waiting` (the default driver) is untouched regardless.

## 4. Validation

- Re-run `bash campaigns/edpp-study/repro_utilization_sweep.sh`: the STABLE-band ratios for
  `fluid`/`rollforward` (and `little`) should move from `predict 0`/NaN to ≈ 1× (floor ≈ realized floor),
  while `waiting` still predicts ~0 (the strawman, unchanged). The OVERLOADED point is unchanged.
- Re-run `bash campaigns/edpp-study/repro_stage_c.sh`: confirm the heavy-overload headline
  (`waiting` 57× → `fluid`/`rollforward` ~1.16×) is unchanged (floor is a no-op there).
- Update FINDINGS "Utilization sweep" and README §7.7 with the floored result and the note that the fix
  is estimator-fidelity-only (no routing effect; `waiting` intentionally unfloored).

## 5. Testing & invariants

- **Unit (`flooredTAdm`):** `flooredTAdm(0, {TIter:100}) == 100`; `flooredTAdm(500, {TIter:100}) == 500`;
  `flooredTAdm(500, {TIter:0}) == 500` (no-op when TIter unavailable).
- **Estimator units:** a free-slot context (BatchSize < MaxBatchSize, KV free) now yields `T_iter` (not
  0) for `fluid` and `rollforward`; a deep-queue saturated context is unchanged (≫ `T_iter`); `little`
  with a tiny queue is floored to `T_iter`. Pre-existing tests that asserted `0` on the free-slot path
  MUST be updated to expect `T_iter` — this is an intended behavior change for these three estimators,
  not a regression; document it in the test and the commit.
- **`waiting` unchanged:** its existing tests must still pass byte-identically (regression guard for the
  default driver).
- **INV-6 determinism:** pure function of `ctx`, no randomness. **INV-9:** `T_iter` is deployable
  (derived from coeffs + occupancy, no `OutputTokens`) — unchanged. **Default byte-identical:** preserved
  because `waiting` is untouched.

## 6. Deliverables

1. `flooredTAdm` helper + its application to `little`/`fluid`/`rollforward` in `sim/admission_estimator.go`;
   `waiting` untouched.
2. Updated estimator unit tests (free-slot cases now expect `T_iter`; `waiting` unchanged) + a
   `flooredTAdm` unit test.
3. Re-run both repro scripts; update FINDINGS "Utilization sweep" (floored numbers, `waiting` still ~0)
   and README §7.7; confirm Stage C overload unchanged.
