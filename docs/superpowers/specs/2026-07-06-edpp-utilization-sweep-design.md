# Utilization Sweep — Decode-Pool Admission-Estimator Fidelity vs Load (Design)

Date: 2026-07-06
Branch: `feat/edpp-estimator-validation` (continues after the Stage C fix-cluster)
Related: `docs/superpowers/specs/2026-07-02-edpp-admission-estimator-design.md` (Stage C, the estimators
and T1 isolation topology this sweep parameterizes);
`campaigns/edpp-study/FINDINGS.md` "Stage C — DE-CONFOUNDED" (the single-point result this hardens);
`docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (§5.4 estimator bias — the guarantee's
one soft dependency this measurement supports). The Layer-2 analytical track is separate.

## 1. Goal & scope

The de-confounded Stage C result validates the occupancy-aware admission estimators at ONE forced-overload
operating point — which is a **non-stationary stress test**: under sustained overload the backlog grows
without bound, admission delay increases monotonically, and there is no steady-state value for the
predicted-vs-realized comparison to converge to. This sweep establishes that the estimators track realized
**decode-pool** admission delay across a range of **bounded, stationary operating points below capacity**,
where admission delay is large but finite and predicted-vs-realized has a well-defined steady-state meaning.

Scope:
- **Decode pool only.** Isolation topology T1 from Stage C: 1P1D, `--pd-decider edpp` with a huge transfer
  penalty (`--edpp-c-xfer 100s`) so every request stays LOCAL on the single decode engine — the
  predicted-vs-realized gap is purely estimator quality, no routing/balancing noise. Prefill-pool
  saturation is a separately-scoped follow-up (needs a prefill-bound workload).
- **Workload:** synth (decode-bound), the base synthetic-data-generation spec.
- **Estimators:** all four deployable (`waiting`, `little`, `fluid`, `rollforward`) + the two oracle twins,
  logged side-by-side per request (reuse the Stage C `--edpp-admission-trace` + `admission_ablation.py`).
- **Measurement-only.** No change to the decider or estimator code (`sim/edpp.go`, `sim/admission_estimator.go`
  untouched). New artifacts are a sweep repro script, an aggregation analyzer, and a FINDINGS section.

## 2. Capacity finding — locate λ*

A pre-pass raises `aggregate_rate` on the T1 topology until the system leaves the stationary regime, using
BLIS's post-hoc saturation detector (`--post-hoc-detector composite --saturation-report <path>`). λ* is the
lowest scanned rate that is NOT classified STABLE (i.e. BACKLOGGED or OVERLOADED). Method:
- Coarse scan over a fixed rate grid (provisional `0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0`; the script widens
  the top if all points are still STABLE, and refines the bottom of the grid downward if even the lowest is
  already non-STABLE). **No bisection** — λ* is known to within one grid step, which is sufficient because
  the per-point achieved utilization ρ̂ is measured directly (§4), not inferred from λ*.
- λ* is **reported, never hard-coded** — it is re-derived each run, so it survives coefficient/workload
  changes.

## 3. The ρ grid

Target achieved utilization ρ = λ/λ* on a grid **strictly below capacity, denser near ρ→1** where the
estimators are expected to diverge most:

```
ρ_targets = {0.5, 0.7, 0.85, 0.9, 0.95, 0.98}   (+ λ* itself logged as the boundary reference)
```

The offered rate for each point is `λ = ρ · λ*`. Specs are generated on the fly by the sweep script from
the base synth spec (rewriting `aggregate_rate`, mirroring `make_specs.py`), so no spec files are
committed. 5000 requests per point (ample for a decode-bound single engine below capacity to reach and hold
steady state, then drain).

## 4. Per-point run

For each ρ point, one `./blis run` on the T1 topology emitting:
- `--edpp-admission-trace <path>` — realized admission delay + all 6 estimator predictions per request×pool
  (only the `local` pool rows matter for T1's decode-pool claim).
- `--post-hoc-detector composite --saturation-report <path>` — the stationarity classification (§5).

**Achieved utilization ρ̂** is measured post-hoc from the run, not assumed equal to the target: the x-axis of
every figure/table is measured ρ̂. Primary estimator: decode-engine busy fraction (steps-executing time /
horizon) if available in the metrics; fallback: `completed_requests / (λ* · horizon)` throughput ratio. The
analyzer records which estimator it used.

## 5. Stationarity certification (two layers)

Per the requirement that the fidelity claim is about *admission delay* being bounded/stationary — not E2E
latency — certification is belt-and-suspenders:

1. **Coarse (built-in):** each swept point's `--saturation-report` must classify STABLE. A point classified
   BACKLOGGED/OVERLOADED is flagged and EXCLUDED from the fidelity curve (it is above capacity — the very
   condition this sweep exists to avoid). This is a proxy (E2E/backlog trend) but reuses shipped, parity-safe
   tooling.
2. **Direct (admission-delay drift):** `utilization_sweep.py` computes, per point, an admission-delay drift
   statistic — after discarding a warmup prefix (first `W` requests by enqueue order, default configurable),
   split the remaining realized admission delays into first-half vs second-half and report the drift ratio
   `median(second half) / median(first half)`. A stationary point has drift ≈ 1; a creeping backlog is caught
   even when E2E looks fine. This directly substantiates "bounded/stationary admission delay."

## 6. Analysis & success criteria

`campaigns/edpp-study/analyze/utilization_sweep.py` aggregates the per-point `admission_ablation.py` outputs
into one table + figure:
- **Primary:** median bias ratio (realized / predicted) per estimator vs measured ρ̂, decode (`local`) pool.
- Alongside each point: the admission-delay drift statistic (§5.2), the built-in detector verdict, and the
  realized admission-delay p50/p90 (to show it is large but finite).
- **Numerical caveat (baked in):** at low ρ the realized admission delay is near zero, so the ratio is
  numerically unstable (tiny/tiny). The analyzer reports BOTH the signed absolute error (ms) and the ratio,
  and marks the ratio as meaningful only where realized p50 exceeds a small floor (e.g. a few ms). The
  narrative reads the absolute error at low ρ and the ratio near ρ→1.

**Success criteria:**
- Every retained point is stationary (STABLE verdict AND drift ≈ 1).
- At each stationary point, `fluid`/`rollforward` hold bias ≈ 1× (well-defined steady-state), while
  `waiting`'s under-prediction **grows with ρ** (occupancy-blindness scales with running-batch occupancy) —
  a monotonic degradation curve, the paper figure.
- `oracle ≈ deployable` where N̂_out is well-estimated (the censored floor holds across load), or a
  quantified oracle–deployable gap where it does not.

## 7. Deliverables

1. `campaigns/edpp-study/repro_utilization_sweep.sh` (tracked) — capacity-find (λ*) → ρ-grid sweep on T1 →
   per-point `admission_ablation.py` → `utilization_sweep.py` aggregation. One command, deterministic (INV-6).
2. `campaigns/edpp-study/analyze/utilization_sweep.py` (tracked) — aggregation, ρ̂ measurement, admission-delay
   drift, numerical-caveat handling, JSON + optional `--plots` figure (bias vs ρ̂ per estimator).
3. FINDINGS "Utilization sweep" section — the ρ̂ table, λ*, the stationarity evidence, and a checkpoint block
   (the numbers a correct re-run must reproduce), so a future run instantly reveals harness vs estimator
   regression.
4. README §7 pointer to the sweep (extends the Stage C walkthrough).
5. Outputs under `campaigns/edpp-study/out/utilization_sweep/` (out/ gitignored).

## 8. Testing & invariants

- INV-6 determinism: the whole sweep is byte-identical across invocations (fixed seed, sorted traces). The
  λ* scan is deterministic (fixed grid).
- INV-13 parity: not re-tested here — the sweep uses `blis run` only (Stage C already covers run/replay
  parity of the admission trace). Noted, not re-exercised.
- INV-9 clean: the sweep drives routing with the default `waiting`-independent path via the c-xfer knob and
  logs deployable + oracle predictions; oracle variants are logging-only (guard already in place). No new
  control-plane read of `OutputTokens`.
- No new Go code, so no new unit tests; the analyzer gets a small self-check on a synthetic trace (known
  drift, known bias) to prove the aggregation/drift math, mirroring the other analyze/ scripts.

## 9. Known limitations (documented, paper-relevant)

- Single decode engine, synth (decode-bound) — this hardens the decode-pool claim; prefill-pool saturation
  is a separate follow-up.
- Carries Stage C's trained-physics attention-basis caveat (the estimators' T_iter uses the same coeffs).
- ρ̂ is a run-averaged utilization; within a finite run there is still an arrival transient (handled by the
  warmup discard) and an end-of-run drain (excluded by measuring over the stationary middle).
