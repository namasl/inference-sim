# Stage C Fix-Cluster — Correct & De-confound the Admission Estimators (Design)

Date: 2026-07-05
Branch: `feat/edpp-estimator-validation`
Related: `docs/superpowers/specs/2026-07-02-edpp-admission-estimator-design.md` (Stage C);
`campaigns/edpp-study/FINDINGS.md` "Stage C" + "Stage C CORRECTION (2026-07-05)". Root cause established
by a systematic-debugging investigation (see FINDINGS correction).

## 1. Goal & scope

Make the Stage C admission-estimator ablation **trustworthy (non-confounded) and correct**. A
root-cause investigation of `fluid`'s ~4e6× under-prediction found a *cluster* of interacting issues,
two of which invalidate the current ablation numbers. This spec fixes all six as one cohesive change
(they share `sim/admission_estimator.go`, `sim/edpp.go`, `sim/cluster/snapshot.go`, `cmd/`):

1. `fluid` formula (naive `/BatchSize`, `N_ahead=1`) → wave mean-field.
2. Deployable remaining-steps collapse (`RemainingStepsEst → 1`) → per-request + **censored lower bound**.
3. **Oracle leakage** into deployable estimators (the measurement-validity bug) → caller-controlled separation.
4. `little` inert (`AdmissionRate` 0 until first completion) → windowed admission-rate signal.
5. Prefill-pool estimators inert (only decode enriched) → symmetric prefill enrichment.
6. No CLI selector → `--edpp-tadm-estimator` flag.

Non-goals: the utilization sweep (separate follow-up); Layer-2 closed-form (separate track); Stage D.

## 2. `fluid` → wave mean-field

Replace `T_adm = 1 / (BatchSize/(RemainingStepsEst·TIter))` (= `RemainingStepsEst·TIter/BatchSize`) with:

```
if BatchSize < MaxBatchSize && FreeKVBlocks >= ReqKVNeed:  return 0
T_adm ≈ ceil((QueueDepth + 1) / BatchSize) · RemainingStepsEst · TIter
```

Rationale (root cause): the batch is **synchronized** — occupants finish thousands of steps together,
not one every `R̄/B` iterations — so the naive fluid-drain `/BatchSize` under-counts by ~`BatchSize`.
The batch instead frees slots in **waves of `BatchSize` every `≈R̄` iterations**; a request at queue
position `QueueDepth` waits `⌈(QueueDepth+1)/BatchSize⌉` waves. Reduces to `≈R̄·TIter` when the queue
fits one batch (matching `rollforward`'s order of magnitude), adds the queue-ahead the old form ignored,
and remains a closed-form mean-field (the Layer-2 birth–death first-passage counterpart), distinct from
`rollforward`'s per-request walk. Guard `BatchSize<=0 || RemainingStepsEst<=0 || TIter<=0 → 0`.

## 3. Deployable remaining-steps + censored lower bound

Two defects: `RemainingDecodeWork` is never populated (→ fallback), and the fallback
`max(N̂_out − meanStepsDone, 1)` uses a **mean** that goes negative under saturation → clamps to 1.

- **Per-request, not mean.** Compute `RemainingStepsEst` in `Decide` from the `RunningDecode` slice as
  the mean over running requests of `remaining_i` (below), not from a single aggregate that can go
  negative.
- **Censored lower bound (both parts, per §(a) decision).** A request that has already produced `k =
  StepsDone_i` tokens satisfies `o_r ≥ k`, a censored observation. So:
  - Each in-flight request's *total-length* estimate is `ô_i = max(N̂_out, StepsDone_i)`; its remaining
    is `remaining_i = max(ô_i − StepsDone_i, 1)`.
  - `N̂_out` itself is floored by observed in-flight elapsed lengths: when updating/reading the per-class
    running mean, incorporate in-flight `StepsDone` as lower bounds so `N̂_out` stops under-counting long
    survivors (a request in flight at `k` tokens contributes `≥k` to the length estimate, not 0). Keep
    the completion-time update (INV-9-safe: `StepsDone` is progress, not `OutputTokens`).
- This is deployable (no oracle read); it materially raises the collapsed estimate and is defensible as
  standard censored-data reasoning.

## 4. Oracle/deployable separation (the confound fix)

**Problem:** the deployable `rollforward`/`fluid` and their `_oracle` variants are the same impl and read
`RunningReqState.TrueRemaining` whenever ≥0; enabling `--edpp-admission-trace` populates `TrueRemaining`,
so the "deployable" predictions silently use oracle data (why `rollforward == rollforward_oracle`).

**Fix — caller-controlled context, estimator impls unchanged.** In `BuildAdmissionRecords`, compute:
- each **deployable** variant's prediction from a context whose `Running[].TrueRemaining` is set to −1
  (stripped) and whose `RemainingStepsEst` is the censored `N̂_out` estimate (§3);
- each **oracle** variant's prediction from a context whose `Running[].TrueRemaining` is populated (and
  `RemainingStepsEst` from the true mean).

The estimator keeps "prefer `TrueRemaining` if ≥0" — but a deployable call never sees a ≥0
`TrueRemaining`. Result: `rollforward` ≠ `rollforward_oracle` in general; the decomposition (`realized −
oracle` = estimator-*form* error; `oracle − deployable` = `N̂_out`-*prediction* error) is finally valid.
The **routing driver** (Decide) already uses only the deployable path with `N̂_out` — confirm it never
consumes `TrueRemaining` (INV-9); the guard from Stage C Task 6 stays.

## 5. `little` admission-rate + `MaxBatchSize`

- **Windowed admission rate.** Add a rolling admissions-per-µs signal (count of admissions over a recent
  time window) to the instance, surfaced on the snapshot as `AdmissionRate` (req/µs) when admission
  detail is on. Non-zero from the first few admissions (unlike `DispatchRate`, which is 0 until first
  *completion*). `Decide` populates `AdmissionContext.AdmissionRate` from it; `little =
  QueueDepth/AdmissionRate` becomes live.
- **`MaxBatchSize`.** Populate it in the snapshot (currently 0), so `fluid`/`rollforward` free-slot
  early-returns are correct by construction, not by accident.

## 6. Prefill-pool enrichment

Symmetric to the decode enrichment: expose the running-**prefill** state on the snapshot (per-request
prefill chunks in flight / prefill tokens remaining, and prefill `MaxBatchSize`), so the prefill
`AdmissionContext` (`ttft_p` side) has real occupancy and `fluid`/`rollforward` are not inert on the
prefill pool. Gated on admission detail (zero-cost off). The estimator math is identical (it operates
on `AdmissionContext`); only the population differs.

## 7. CLI flag

`--edpp-tadm-estimator {waiting|little|fluid|rollforward}` (run + replay), sets `EDPPConfig.TAdmEstimator`;
default `""`→`waiting` (byte-identical). Oracle names rejected by the existing `NewEDPPDecider` INV-9
guard — this flag finally exercises that guard's runtime path (a test selects an oracle name → fatal).

## 8. Validation (re-run, de-confounded)

Re-run `repro_stage_c.sh` (T1 all-local, T2 all-disagg). Now expect, per pool:
- `waiting`: still large under-prediction (unchanged strawman).
- `fluid` (wave): a sensible middle — no longer ~1e6× off; order-of-magnitude of realized.
- `rollforward` **deployable**: improved by the censored floor but **distinct from and worse than
  `rollforward_oracle`**; the `oracle − deployable` gap is the real `N̂_out` residual after censoring.
- `little`: non-zero, aggregate-accurate but transient-lagging (expected).
- prefill pool: real (non-zero) predictions; `ttft_p` now measurable.
Update FINDINGS with the de-confounded tables and explicitly retract the oracle-fed 1.29× as the
deployable number. Record the post-censoring `N̂_out` residual.

## 9. Testing & invariants

- **fluid** unit: short queue (`QueueDepth<BatchSize`) → `≈R̄·TIter`; deep queue → `⌈(Q+1)/B⌉` waves;
  free-slot → 0.
- **censored floor** unit: request with `StepsDone=k > N̂_out` → total `≥k`, remaining `≥1` (not
  collapsed); `N̂_out` floored by in-flight elapsed.
- **oracle/deployable separation** unit: for one running req with `TrueRemaining=t` and `N̂_out=n≠t`,
  the deployable prediction uses `n` (context stripped) and the oracle prediction uses `t`, and they
  differ — this is the regression test that the confound cannot recur.
- **little** unit: non-zero with a populated windowed rate; 0 when rate 0.
- **prefill context** populated (non-empty running-prefill) on a disagg run.
- **INV-9**: deployable/routing path never reads `OutputTokens` or `TrueRemaining`; guard rejects oracle
  as routing driver (now also via the CLI flag — a runtime test).
- **INV-6** determinism; **INV-13** run/replay parity of the admission trace; default `waiting`
  byte-identical (existing `TestEDPP*` unchanged).
- Full `go test ./...`; `gofmt`/`go vet` clean (golangci-lint in CI).

## 10. Deliverables

1. `fluid` wave formula (`sim/admission_estimator.go`).
2. Per-request censored remaining-steps + `N̂_out` lower-bound floor (`sim/edpp.go`, and the `N̂_out`
   update path).
3. Oracle/deployable separation in `BuildAdmissionRecords` (`sim/cluster/cluster.go`) — deployable
   contexts strip `TrueRemaining`.
4. Windowed admission-rate signal + `MaxBatchSize` population (`sim/…` instance/snapshot).
5. Prefill-pool running-state enrichment (snapshot + `Decide` prefill context).
6. `--edpp-tadm-estimator` flag (run + replay).
7. Re-run + de-confounded FINDINGS update (retract the 1.29× deployable claim; record `N̂_out` residual).
8. Tests per §9.
