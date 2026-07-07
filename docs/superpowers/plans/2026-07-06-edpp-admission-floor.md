# EDPP Sub-Saturation Admission-Delay Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lower-bound the occupancy-aware admission estimators (`little`/`fluid`/`rollforward` + oracles) by one `T_iter` so they track the sub-saturation admission floor instead of predicting 0, leaving the `waiting` strawman untouched.

**Architecture:** Add one pure helper `flooredTAdm(est, ctx) = max(est, ctx.TIter)` in `sim/admission_estimator.go` and wrap every return of the three floored estimators with it. `waiting` is not floored (preserves the byte-identical default driver). Because it is a lower bound, it is a no-op above saturation, so Stage C's overload numbers are unchanged. Then re-validate (sweep + Stage C) and update the docs.

**Tech Stack:** Go 1.22 (`sim/` package), Python3/bash (existing repro scripts), no new dependencies.

## Global Constraints

- Branch: `feat/edpp-estimator-validation`.
- Only `sim/admission_estimator.go` + `sim/admission_estimator_test.go` change in code; then docs (`FINDINGS.md`, `README.md`). No routing-rule, snapshot, or trace changes.
- **`waitingEstimator` MUST remain byte-identical** — do not floor it (it is the occupancy-blind strawman and the default routing driver; the Stage A–C default-byte-identical invariant depends on it).
- Floor value: exactly one `ctx.TIter` (already a field on `AdmissionContext`). No calibration, no residual/2.
- `flooredTAdm` is a no-op when `ctx.TIter <= est` (including `TIter <= 0`, i.e. unavailable).
- INV-6 determinism (pure function of `ctx`, no randomness); INV-9 unaffected (`TIter` is coeff/occupancy-derived, never reads `OutputTokens`).
- The oracle variants (`fluid_oracle`/`rollforward_oracle`) share the `fluid`/`rollforward` impls, so they inherit the floor — that is fine (logging-only).
- Run Go tests with `go test ./sim/ -run TestName`. Build with `go build -o blis main.go`.

---

### Task 1: Add `flooredTAdm` and apply to little/fluid/rollforward

**Files:**
- Modify: `sim/admission_estimator.go`
- Test: `sim/admission_estimator_test.go`

**Interfaces:**
- Consumes: `AdmissionContext.TIter float64` (existing field).
- Produces: `func flooredTAdm(est float64, ctx AdmissionContext) float64` (package-private); unchanged public `EstimateTAdm` signatures on all estimators.

- [ ] **Step 1: Write the failing tests**

Add to `sim/admission_estimator_test.go`:

```go
func TestFlooredTAdm(t *testing.T) {
	// est below the floor -> lifted to TIter
	if got := flooredTAdm(0, AdmissionContext{TIter: 100}); got != 100 {
		t.Fatalf("floor(0, TIter=100) = %v, want 100", got)
	}
	// est above the floor -> unchanged
	if got := flooredTAdm(500, AdmissionContext{TIter: 100}); got != 500 {
		t.Fatalf("floor(500, TIter=100) = %v, want 500", got)
	}
	// TIter unavailable -> no-op
	if got := flooredTAdm(500, AdmissionContext{TIter: 0}); got != 500 {
		t.Fatalf("floor(500, TIter=0) = %v, want 500", got)
	}
}

func TestFloor_FreeSlotEstimatorsReturnTIter(t *testing.T) {
	// A free-slot context: fluid and rollforward must now return TIter, not 0.
	free := AdmissionContext{BatchSize: 2, MaxBatchSize: 4, FreeKVBlocks: 100, ReqKVNeed: 10, TIter: 1000, RemainingStepsEst: 20}
	if got := (fluidEstimator{}).EstimateTAdm(free); got != 1000 {
		t.Fatalf("fluid free-slot = %v, want 1000 (one TIter floor)", got)
	}
	if got := (rollforwardEstimator{}).EstimateTAdm(free); got != 1000 {
		t.Fatalf("rollforward free-slot = %v, want 1000 (one TIter floor)", got)
	}
	// little with a tiny queue floored to TIter.
	if got := (littleEstimator{}).EstimateTAdm(AdmissionContext{QueueDepth: 0, AdmissionRate: 0.002, TIter: 1000}); got != 1000 {
		t.Fatalf("little tiny-queue = %v, want 1000 (one TIter floor)", got)
	}
	// waiting is NOT floored: still 0 when QWork is 0.
	if got := (waitingEstimator{}).EstimateTAdm(free); got != 0 {
		t.Fatalf("waiting free-slot = %v, want 0 (strawman unfloored)", got)
	}
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `go test ./sim/ -run 'TestFlooredTAdm|TestFloor_FreeSlotEstimatorsReturnTIter' -v`
Expected: FAIL — `flooredTAdm` undefined (compile error), and once defined, `fluid`/`rollforward` free-slot still return 0.

- [ ] **Step 3: Add the helper**

Add to `sim/admission_estimator.go` (after the imports/`AdmissionContext`, before the estimators):

```go
// flooredTAdm lower-bounds an admission-delay estimate by one iteration: even with a
// free slot, a request waits for the current decode step to finish before the next
// FormBatch admits it (~T_iter). No-op when TIter is unavailable (<=0) or already exceeded.
func flooredTAdm(est float64, ctx AdmissionContext) float64 {
	if ctx.TIter > est {
		return ctx.TIter
	}
	return est
}
```

- [ ] **Step 4: Apply the floor to little/fluid/rollforward (NOT waiting)**

Rewrite the three estimator methods so every return is wrapped. `waitingEstimator` is left exactly as-is.

`littleEstimator`:
```go
func (littleEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.AdmissionRate <= 0 {
		return flooredTAdm(0, ctx)
	}
	return flooredTAdm(float64(ctx.QueueDepth)/ctx.AdmissionRate, ctx)
}
```

`fluidEstimator`:
```go
func (fluidEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}
	if ctx.BatchSize <= 0 || ctx.RemainingStepsEst <= 0 || ctx.TIter <= 0 {
		return flooredTAdm(0, ctx)
	}
	waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
	return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
}
```

`rollforwardEstimator` — wrap each of its four returns identically: the free-slot `return 0` becomes `return flooredTAdm(0, ctx)`; `return float64(d.step) * ctx.TIter` becomes `return flooredTAdm(float64(d.step)*ctx.TIter, ctx)`; the wave fallback `return waves * ctx.RemainingStepsEst * ctx.TIter` becomes `return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)`; the `return float64(deps[len(deps)-1].step) * ctx.TIter` becomes `return flooredTAdm(float64(deps[len(deps)-1].step)*ctx.TIter, ctx)`; and the final `return 0` becomes `return flooredTAdm(0, ctx)`. Leave the loop/sort logic untouched.

- [ ] **Step 5: Update the two pre-existing free-slot assertions (intended behavior change)**

In `sim/admission_estimator_test.go`, the free-slot cases now return `T_iter`, not 0. This is the intended fix, not a regression.
- `TestFluidEstimator` (the `free` context, currently `if got := e.EstimateTAdm(free); got != 0 {`): change the expectation to `1000` (that context has `TIter: 1000`), and update the failure message.
- `TestFluidEstimator_WaveMeanField` (the free-slot line currently asserting `!= 0` on the `BatchSize: 2, MaxBatchSize: 4` context with `TIter: 1000`): change the expectation to `1000`.
- Leave every other assertion unchanged (the `full`/`deep`/oracle/waiting cases return values `>> 1000` or belong to `waiting`, so the floor is a no-op there).

- [ ] **Step 6: Run the full estimator test file**

Run: `go test ./sim/ -run 'Estimator|Floor|Oracle|Guard' -v`
Expected: PASS — the new floor tests pass, the two updated free-slot assertions pass at 1000, all saturated/oracle/waiting cases unchanged.

- [ ] **Step 7: Build + full sim package test**

Run: `go build -o blis main.go && go test ./sim/`
Expected: build OK; `ok  .../sim`. (gofmt the file: `gofmt -w sim/admission_estimator.go sim/admission_estimator_test.go`.)

- [ ] **Step 8: Commit**

```bash
git add sim/admission_estimator.go sim/admission_estimator_test.go
git commit -m "feat(edpp): floor occupancy-aware admission estimators by one T_iter

little/fluid/rollforward (+oracles) lower-bound their admission estimate by
ctx.TIter (the current-step→next-FormBatch wait); waiting left unfloored as the
byte-identical strawman. Lower bound ⇒ no-op above saturation."
```

---

### Task 2: Re-validate and update the docs

**Files:**
- Modify: `campaigns/edpp-study/FINDINGS.md` (replace the "PLANNED FOLLOW-UP" note in the Utilization-sweep section with the implemented result)
- Modify: `campaigns/edpp-study/README.md` (§7.7 — update the planned-follow-up line to "done")

**Interfaces:**
- Consumes: the built `blis` from Task 1 and the existing `repro_utilization_sweep.sh` / `repro_stage_c.sh`.

- [ ] **Step 1: Re-run the utilization sweep**

Run: `bash campaigns/edpp-study/repro_utilization_sweep.sh`
Then inspect: `python3 -c "import json; r=json.load(open('campaigns/edpp-study/out/utilization_sweep/sweep.json')); [print(round(p['rho_hat'],3), p['stationary_verdict'], {k:(round(v['median_ratio_real_over_pred'],2) if v.get('median_ratio_real_over_pred') else None) for k,v in p['estimators'].items()}) for p in r['points']]"`
Expected: at the STABLE points, `fluid`/`rollforward` (and `little`) ratios move from `None`/NaN toward ≈1× (floor ≈ realized ~30–47 ms); `waiting` still ≈ huge (unfloored strawman). **Report the actual numbers honestly** — if a floored estimator lands well off 1× (e.g. because the realized floor is <1 T_iter at high load), record it, don't tune.

- [ ] **Step 2: Confirm Stage C overload is unchanged**

Run: `bash campaigns/edpp-study/repro_stage_c.sh`
Then inspect `out/stage_c/t1_ablation.json`: verify the `local` pool still shows `waiting` ≈57× and `fluid`/`rollforward` ≈1.16× (the floor is a no-op at overload, where the wait ≫ T_iter). If these moved, the floor is leaking into the saturated path — stop and investigate.

- [ ] **Step 3: Update FINDINGS**

In `campaigns/edpp-study/FINDINGS.md`, in the "Utilization sweep" section, replace the "**PLANNED FOLLOW-UP (scoped, not yet implemented)**" paragraph with a "**FLOOR IMPLEMENTED (<date>, commit <hash>)**" paragraph stating: the occupancy-aware estimators now lower-bound by one `T_iter`; give the re-run STABLE-band ratios (floored ≈1× vs `waiting` unchanged) and confirm the Stage C overload headline is unchanged. Keep the mechanism explanation above it.

- [ ] **Step 4: Update README §7.7**

In `campaigns/edpp-study/README.md` §7.7, change the "Planned follow-up (scoped)" sentence to note it is **done**: the occupancy-aware estimators are floored by one `T_iter` (spec `docs/superpowers/specs/2026-07-06-edpp-admission-floor-design.md`), `waiting` unfloored, Stage C overload unchanged.

- [ ] **Step 5: Commit**

```bash
git add campaigns/edpp-study/FINDINGS.md campaigns/edpp-study/README.md
git commit -m "docs(edpp): record admission-floor result (sweep STABLE band now ~1x; overload unchanged)"
```

---

## Notes for the implementer (confirm-in-situ)

- The `rollforward` rewrite touches four return sites inside a function with a loop and a `sort` — change ONLY the return expressions (wrap in `flooredTAdm(..., ctx)`); do not alter the departure-walk logic.
- If `go test ./sim/` surfaces an unrelated pre-existing failure, note it in the report but do not fix it (out of scope).
- The sweep re-run is a multi-simulation compute step (~15–25 min), same as when it was first run.
