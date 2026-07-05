# Stage C Fix-Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct and de-confound the Stage C admission-delay estimators — fix `fluid`'s formula, the collapsed deployable remaining-steps (censored floor), the oracle-leakage measurement bug, inert `little`, inert prefill-pool, and add the CLI selector — so the fidelity ablation is trustworthy.

**Architecture:** Estimators are pure functions of `AdmissionContext` (`sim/admission_estimator.go`); the decider (`sim/edpp.go`) assembles the context and the cluster (`sim/cluster/`) logs per-request predictions. Fixes touch: the `fluid` formula; the deployable remaining-steps computation + a censored lower bound in `Decide`; snapshot population (`MaxBatchSize`, windowed `AdmissionRate`, running-prefill state); and `BuildAdmissionRecords` (strip `TrueRemaining` for deployable predictions).

**Tech Stack:** Go 1.22+ (`sim/`, `sim/cluster/`, `cmd/`), Python 3 (analysis).

## Global Constraints

- Default `--edpp-tadm-estimator` unset → `waiting` → decider byte-identical to pre-change (existing `TestEDPP*` unchanged).
- INV-9: the routing/deployable path never reads `req.OutputTokens` or `RunningReqState.TrueRemaining`; only `_oracle` logging predictions may read `TrueRemaining`. Oracle names rejected as routing driver (existing `NewEDPPDecider` guard).
- INV-6 determinism (pure estimators; admission trace sorted by request_id); INV-7 (new snapshot fields Periodic tier); INV-13 (admission trace identical run vs replay).
- Zero cost when `--edpp-admission-trace` unset (new snapshot population gated on admission detail).
- Censored floor is LOCAL to the admission remaining-steps computation in `Decide` — do NOT change the global `nHatFor` running mean (it feeds `W_d`; out of scope).
- `go test ./...`, `gofmt -l`, `go vet ./...` clean after each task (golangci-lint in CI).

---

### Task 1: `fluid` → wave mean-field

**Files:** Modify `sim/admission_estimator.go` (`fluidEstimator.EstimateTAdm`); Test `sim/admission_estimator_test.go`.

**Interfaces:** Consumes `AdmissionContext.{BatchSize, MaxBatchSize, FreeKVBlocks, ReqKVNeed, QueueDepth, RemainingStepsEst, TIter}`. No signature change.

- [ ] **Step 1: Write the failing test**

```go
func TestFluidEstimator_WaveMeanField(t *testing.T) {
	e, _ := NewAdmissionEstimator("fluid")
	// Free slot + KV → 0.
	if got := e.EstimateTAdm(AdmissionContext{BatchSize: 2, MaxBatchSize: 4, FreeKVBlocks: 100, ReqKVNeed: 10, QueueDepth: 0, RemainingStepsEst: 20, TIter: 1000}); got != 0 {
		t.Fatalf("free slot → 0, got %v", got)
	}
	// Full batch, short queue (QueueDepth+1 <= BatchSize → 1 wave): ⌈(0+1)/4⌉·20·1000 = 20000.
	full1 := AdmissionContext{BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 10, QueueDepth: 0, RemainingStepsEst: 20, TIter: 1000}
	if got := e.EstimateTAdm(full1); got < 19999 || got > 20001 {
		t.Fatalf("short-queue 1 wave → ~20000, got %v", got)
	}
	// Deep queue: QueueDepth=9, BatchSize=4 → ⌈10/4⌉=3 waves → 3·20·1000 = 60000.
	deep := AdmissionContext{BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 10, QueueDepth: 9, RemainingStepsEst: 20, TIter: 1000}
	if got := e.EstimateTAdm(deep); got < 59999 || got > 60001 {
		t.Fatalf("deep-queue 3 waves → ~60000, got %v", got)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestFluidEstimator_WaveMeanField -v`
Expected: FAIL (old form returns `RemainingStepsEst·TIter/BatchSize` = 5000 for `full1`, not 20000).

- [ ] **Step 3: Replace the fluid body**

In `sim/admission_estimator.go`, replace `fluidEstimator.EstimateTAdm`:

```go
func (fluidEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	// Admit next iteration if a slot AND enough KV already fit.
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return 0
	}
	if ctx.BatchSize <= 0 || ctx.RemainingStepsEst <= 0 || ctx.TIter <= 0 {
		return 0
	}
	// Synchronized batch: occupants finish ~R̄ steps together, so slots free in WAVES of
	// BatchSize every ~R̄ iterations. A request at queue position QueueDepth waits
	// ⌈(QueueDepth+1)/BatchSize⌉ waves. (Not the naive fluid-drain /BatchSize.)
	waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
	return waves * ctx.RemainingStepsEst * ctx.TIter
}
```
Ensure `math` is imported in the file.

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./sim/ -run 'TestFluidEstimator' -v`
Expected: PASS (update/replace any prior `TestFluidEstimator` that asserted the old `/BatchSize` value — its expectation is now the wave form; note the change in the report).

- [ ] **Step 5: Commit**

```bash
git add sim/admission_estimator.go sim/admission_estimator_test.go
git commit -m "fix(edpp): fluid uses wave mean-field (⌈(QueueDepth+1)/BatchSize⌉·R̄·TIter)"
```

---

### Task 2: Censored deployable remaining-steps + `MaxBatchSize` population

**Files:** Modify `sim/edpp.go` (the `remStepsEst` computation in `Decide`, ~line 456-469); Modify `sim/cluster/snapshot.go` (populate `MaxBatchSize`); Test `sim/edpp_test.go`.

**Interfaces:** Consumes `decSnap.RunningDecode []RunningReqState` (`StepsDone`), `d.nHatFor(class).mean()`. Produces the corrected `RemainingStepsEst` fed into `AdmissionContext` (decode side) + populated `RoutingSnapshot.MaxBatchSize`.

- [ ] **Step 1: Write the failing test**

```go
// The deployable remaining-steps must NOT collapse to 1 under saturation: with running
// requests whose StepsDone exceed a stale N̂_out, N̂_out is floored by the max in-flight
// elapsed (censored lower bound: o_r ≥ tokens already produced), so the estimate reflects
// the running occupants' scale rather than 1.
func TestDecide_CensoredRemainingSteps(t *testing.T) {
	var seen AdmissionContext
	spy := admissionSpy{onCall: func(c AdmissionContext) { seen = c }}
	d := newTestEDPPDeciderWithEstimator(t, spy)
	// N̂_out starts at its default (small/0); running requests are deep into decode.
	state := &RouterState{SelectedInstance: "d0", Snapshots: []RoutingSnapshot{{
		ID: "d0", BatchSize: 3, MaxBatchSize: 4,
		RunningDecode: []RunningReqState{{StepsDone: 2000, TrueRemaining: -1}, {StepsDone: 2200, TrueRemaining: -1}, {StepsDone: 1800, TrueRemaining: -1}},
	}}}
	d.Decide(makeReq("r1", 100, "batch"), state)
	// Censored floor: N̂_out_eff ≥ max StepsDone (2200); remaining_i = max(N̂_out_eff − StepsDone_i, 1).
	// Mean over {2200-2000, 2200-2200→1, 2200-1800} = mean(200,1,400) ≈ 200.3 — NOT 1.
	if seen.RemainingStepsEst < 100 {
		t.Fatalf("remaining-steps collapsed to ~1 (%v); censored floor should keep it ~200", seen.RemainingStepsEst)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestDecide_CensoredRemainingSteps -v`
Expected: FAIL — old code computes `max(N̂_out − meanSteps, 1)` = `max(small − 2000, 1)` = 1.

- [ ] **Step 3: Replace the remaining-steps computation**

In `sim/edpp.go` `Decide`, replace the `remStepsEst` block:

```go
	// RemainingStepsEst (deployable): per-running-request censored estimate, NOT a mean that
	// can go negative. A request that has produced StepsDone tokens has o_r ≥ StepsDone
	// (censored lower bound), so floor the class output estimate by the max in-flight elapsed.
	remStepsEst := 1.0
	if n := len(decSnap.RunningDecode); n > 0 {
		nHatOut := d.nHatFor(req.SLOClass).mean()
		var maxSteps int64
		for _, r := range decSnap.RunningDecode {
			if r.StepsDone > maxSteps {
				maxSteps = r.StepsDone
			}
		}
		nHatEff := math.Max(nHatOut, float64(maxSteps)) // censored: N̂_out ≥ longest in-flight elapsed
		var sum float64
		for _, r := range decSnap.RunningDecode {
			sum += math.Max(nHatEff-float64(r.StepsDone), 1) // per-request remaining, floored at 1
		}
		remStepsEst = sum / float64(n)
	}
```
(Removes the reliance on the never-populated `decSnap.RemainingDecodeWork`; keep the field for BC but it's no longer the primary source.)

- [ ] **Step 4: Populate `MaxBatchSize` in the snapshot**

In `sim/cluster/snapshot.go`, where `RunningDecode`/`BatchSize` refresh (the Periodic `BatchSize` tier block), set `snap.MaxBatchSize` from the instance's configured max running requests (the same value the batch-formation cap uses; confirm the accessor in situ — e.g. `inst.MaxRunningReqs()` / the field feeding `MaxRunningReqs`). Gated identically to the other admission-detail fields.

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./sim/... -run 'TestDecide_CensoredRemainingSteps|TestEDPP' -v`
Expected: PASS; `TestEDPP*` unchanged (waiting path ignores `RemainingStepsEst`).

- [ ] **Step 6: Commit**

```bash
git add sim/edpp.go sim/cluster/snapshot.go sim/edpp_test.go
git commit -m "fix(edpp): censored per-request remaining-steps (no collapse-to-1) + populate MaxBatchSize"
```

---

### Task 3: Windowed admission-rate signal for `little`

**Files:** Modify `sim/cluster/instance.go` (or wherever admissions are counted) + `sim/cluster/snapshot.go` (populate `AdmissionRate`); Modify `sim/edpp.go` (fill `AdmissionContext.AdmissionRate` from the snapshot); Test `sim/cluster/…_test.go`.

**Interfaces:** Produces `RoutingSnapshot.AdmissionRate` (req/µs) populated from a rolling admission counter (non-zero from the first admissions, unlike `DispatchRate` which is 0 until first completion). Consumed by `little`.

- [ ] **Step 1: Write the failing test**

```go
// A rolling admission-rate signal is non-zero after admissions occur, even with zero completions.
func TestAdmissionRate_NonZeroBeforeCompletions(t *testing.T) {
	inst := newTestInstanceWithAdmissionDetail(t) // helper: instance with admission detail enabled
	// Simulate 3 admissions over a 300ms window with no completions.
	inst.recordAdmission(0)
	inst.recordAdmission(100_000)
	inst.recordAdmission(200_000)
	rate := inst.WindowedAdmissionRate(300_000) // now=300ms
	if rate <= 0 {
		t.Fatalf("windowed admission rate must be > 0 after admissions with no completions, got %v", rate)
	}
}
```

(Adapt the helper/method names to the instance's existing structure — the point is a rolling
count of admissions over a recent window, expressed as req/µs.)

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/cluster/ -run TestAdmissionRate_NonZeroBeforeCompletions -v`
Expected: FAIL — no admission counter / `WindowedAdmissionRate` yet.

- [ ] **Step 3: Implement the windowed admission counter**

Add a lightweight rolling counter on the instance: record each admission timestamp (or a decaying
count), and `WindowedAdmissionRate(now) = admissionsInWindow / windowDuration` (req/µs). Populate
`snap.AdmissionRate` from it in `snapshot.go` (gated on admission detail). In `sim/edpp.go`, set
`AdmissionContext.AdmissionRate` from the selected snapshot's `AdmissionRate` (replace the
`DispatchRate/1e6` fallback source with this when present; keep DispatchRate as a secondary fallback).

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./sim/cluster/ -run TestAdmissionRate_NonZeroBeforeCompletions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/cluster/instance.go sim/cluster/snapshot.go sim/edpp.go sim/cluster/*_test.go
git commit -m "fix(edpp): windowed admission-rate signal so little is live before first completion"
```

---

### Task 4: Oracle/deployable separation in `BuildAdmissionRecords`

**Files:** Modify `sim/cluster/cluster.go` (`BuildAdmissionRecords` / its per-variant prediction loop); Test `sim/cluster/admission_trace_test.go`.

**Interfaces:** Consumes the captured `AdmissionContext` per request (Stage C Task 7) + `NewAdmissionEstimator(name)`. Produces per-request predictions where **deployable** variants see a context with `Running[].TrueRemaining = -1` and **oracle** variants see the populated one.

- [ ] **Step 1: Write the failing test**

```go
// The deployable rollforward must NOT read TrueRemaining even when it is populated (oracle mode);
// only rollforward_oracle may. With N̂_out-based remaining ≠ TrueRemaining, the two predictions differ.
func TestBuildAdmissionRecords_OracleDeployableSeparation(t *testing.T) {
	cs := newClusterWithCapturedAdmissionCtx(t, map[string]AdmissionContextCapture{
		"r1": { /* decode ctx: one running req, TrueRemaining=2 (oracle), StepsDone big so N̂_out estimate ≠ 2 */
			Decode: AdmissionContext{BatchSize: 1, MaxBatchSize: 1, FreeKVBlocks: 0, ReqKVNeed: 5, TIter: 1000,
				RemainingStepsEst: 50, Running: []RunningReqState{{StepsDone: 100, KVBlocks: 10, TrueRemaining: 2}}},
			RealizedDecodeTAdm: 2000,
		},
	})
	recs := cs.BuildAdmissionRecords()
	var r trace.AdmissionRecord
	for _, x := range recs { if x.RequestID == "r1" && x.Pool == "decode" { r = x } }
	// deployable rollforward uses RemainingStepsEst=50 → ~50000µs; oracle uses TrueRemaining=2 → ~2000µs. Distinct.
	if r.TAdmPredRollforward == r.TAdmPredRollforwardOracle {
		t.Fatalf("deployable and oracle rollforward must differ (oracle leakage); got both %v", r.TAdmPredRollforward)
	}
	if r.TAdmPredRollforwardOracle < 1999 || r.TAdmPredRollforwardOracle > 2001 {
		t.Fatalf("oracle should use TrueRemaining=2 → ~2000, got %v", r.TAdmPredRollforwardOracle)
	}
}
```
(Adapt the capture-injection helper to the actual Stage-C-Task-7 structure holding per-request contexts.)

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/cluster/ -run TestBuildAdmissionRecords_OracleDeployableSeparation -v`
Expected: FAIL — both currently read `TrueRemaining` (identical).

- [ ] **Step 3: Strip `TrueRemaining` for deployable predictions**

In `BuildAdmissionRecords`, before computing each prediction: build a **deployable** context = a copy of
the captured context with every `Running[i].TrueRemaining = -1` (and `RemainingStepsEst` = the censored
`N̂_out` estimate already in the captured context); use it for `waiting/little/fluid/rollforward`. Use the
**original** (TrueRemaining-populated) context only for `fluid_oracle/rollforward_oracle`. Deep-copy the
`Running` slice so stripping doesn't mutate the captured/original context.

```go
	stripOracle := func(c AdmissionContext) AdmissionContext {
		rc := make([]RunningReqState, len(c.Running))
		copy(rc, c.Running)
		for i := range rc { rc[i].TrueRemaining = -1 }
		c.Running = rc
		return c
	}
	// deployable = stripOracle(captured); oracle = captured (unchanged)
```

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./sim/cluster/ -run 'TestBuildAdmissionRecords' -v`
Expected: PASS (separation holds; existing admission-trace tests still green).

- [ ] **Step 5: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/admission_trace_test.go
git commit -m "fix(edpp): strip TrueRemaining for deployable predictions (de-confound the ablation)"
```

---

### Task 5: Prefill-pool running-state enrichment

**Files:** Modify `sim/routing.go` (a `RunningPrefill`/prefill-occupancy field if needed) + `sim/cluster/snapshot.go` (populate for prefill instances) + `sim/edpp.go` (fill the prefill `AdmissionContext` with occupancy); Test `sim/edpp_test.go` / a cluster test.

**Interfaces:** Produces a populated prefill-side `AdmissionContext` (`BatchSize`/`MaxBatchSize`/`RunningDecode`-analog for prefill, `RemainingStepsEst`, `QueueDepth`) so `fluid`/`rollforward` are not inert on the prefill pool.

- [ ] **Step 1: Write the failing test**

```go
// The prefill AdmissionContext must carry real occupancy so ttft_p estimators aren't inert.
func TestDecide_PrefillContextPopulated(t *testing.T) {
	var seenPrefill AdmissionContext
	spy := admissionSpyByPool{onPrefill: func(c AdmissionContext) { seenPrefill = c }}
	d := newTestEDPPDeciderWithPoolSpy(t, spy)
	state := &RouterState{ /* selected decode + a prefill snapshot with running prefill work */ }
	// (construct a prefill snapshot with BatchSize>0 / running prefill state)
	d.Decide(makeReq("r1", 4000, "batch"), state)
	if seenPrefill.BatchSize == 0 && seenPrefill.RemainingStepsEst == 0 {
		t.Fatalf("prefill context inert: %+v", seenPrefill)
	}
}
```
(Adapt to how `Decide` currently builds the prefill context and how the spy distinguishes pools.)

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestDecide_PrefillContextPopulated -v`
Expected: FAIL — prefill context currently has only `QWork`/`Mu`, no occupancy.

- [ ] **Step 3: Enrich prefill occupancy**

Add prefill running-state to the snapshot (symmetric to `RunningDecode` — e.g. `RunningPrefill` with per-request remaining prefill chunks, or at minimum `BatchSize`/`MaxBatchSize`/`QueueDepth`/`RemainingStepsEst` for the prefill pool). Populate in `snapshot.go` for prefill-role instances (gated). In `Decide`, fill the prefill `AdmissionContext` occupancy fields from `prefillSnap` (mirroring the decode side).

- [ ] **Step 4: Run to verify it passes**

Run: `go test ./sim/ -run 'TestDecide_PrefillContextPopulated|TestEDPP' -v`
Expected: PASS; `TestEDPP*` unchanged.

- [ ] **Step 5: Commit**

```bash
git add sim/routing.go sim/cluster/snapshot.go sim/edpp.go sim/edpp_test.go
git commit -m "fix(edpp): enrich prefill-pool occupancy so ttft_p estimators are live"
```

---

### Task 6: `--edpp-tadm-estimator` CLI flag

**Files:** Modify `cmd/root.go` (flag var + registration in `registerSimConfigFlags`; set `DeploymentConfig`/`EDPPConfig.TAdmEstimator`), `cmd/replay.go` (shared); Test `cmd/…_test.go` or `sim/…` for the guard.

**Interfaces:** `--edpp-tadm-estimator string` (default `""`→waiting) → `EDPPConfig.TAdmEstimator`. Deployable-only; oracle names rejected by the existing `NewEDPPDecider` guard (this exercises its runtime path).

- [ ] **Step 1: Write the failing test**

```go
// Selecting an oracle estimator as the routing driver must be rejected (INV-9 guard runtime path).
func TestNewEDPPDecider_RejectsOracleDriver(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when oracle estimator is the routing driver")
		}
	}()
	cfg := validTestEDPPConfig(t)
	cfg.TAdmEstimator = "rollforward_oracle"
	_ = NewEDPPDecider(cfg, nil, nil, nil)
}
```

- [ ] **Step 2: Run to verify it fails/passes appropriately**

Run: `go test ./sim/ -run TestNewEDPPDecider_RejectsOracleDriver -v`
Expected: PASS if the guard already panics (Stage C Task 6 added it); if it doesn't fire, that's a real gap — fix the guard. Either way this test now pins the runtime behavior.

- [ ] **Step 3: Add the flag (run + replay)**

In `cmd/root.go`: add `edppTAdmEstimator string`; register in `registerSimConfigFlags`:
`cmd.Flags().StringVar(&edppTAdmEstimator, "edpp-tadm-estimator", "", "EDPP admission-delay estimator that DRIVES routing: waiting|little|fluid|rollforward (default waiting). Oracle variants are logging-only and rejected here.")`.
Thread it into the EDPP config where `--edpp-coeffs`/`--edpp-tau-*` are wired (`DeploymentConfig` → `EDPPConfig.TAdmEstimator`), on both run and replay.

- [ ] **Step 4: Build + smoke + run guard test**

```bash
go build -o blis main.go && ./blis run --help | grep edpp-tadm-estimator
go test ./sim/ -run TestNewEDPPDecider_RejectsOracleDriver -v
```
Expected: flag present; guard test PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/root.go cmd/replay.go sim/edpp_test.go
git commit -m "feat(edpp): --edpp-tadm-estimator flag (deployable-only; oracle rejected as driver)"
```

---

### Task 7: Re-run de-confounded ablation + FINDINGS update

**Files:** Modify `campaigns/edpp-study/FINDINGS.md`; (optionally) `campaigns/edpp-study/analyze/admission_ablation.py` if the decomposition needs the now-distinct deployable/oracle columns surfaced.

**Interfaces:** none (verification + docs).

- [ ] **Step 1: Rebuild + re-run**

```bash
go build -o blis main.go
bash campaigns/edpp-study/repro_stage_c.sh
cat campaigns/edpp-study/out/stage_c/{t1,t2}_ablation.json
```

- [ ] **Step 2: Verify the de-confounded expectations**

Confirm from the JSON: `rollforward` (deployable) and `rollforward_oracle` are now **distinct** (oracle
near-exact; deployable improved by the censored floor but worse — the gap is the real `N̂_out` residual);
`fluid` (wave) is now within an order of magnitude of realized (no longer ~1e6× off); `little` non-zero;
prefill-pool predictions non-zero. If deployable==oracle still, Task 4 didn't take — stop and fix.

- [ ] **Step 3: Rewrite the Stage C FINDINGS numbers**

Replace the provisional Stage C tables with the de-confounded results: the deployable-vs-oracle gap per
pool, the `N̂_out` residual after censoring, `fluid`'s wave-form ratio, `little`'s aggregate accuracy.
Explicitly state the earlier 1.29× was oracle-fed and give the corrected deployable `rollforward` number.
Keep the utilization-sweep follow-up note.

- [ ] **Step 4: Commit**

```bash
git add campaigns/edpp-study/FINDINGS.md campaigns/edpp-study/analyze/admission_ablation.py
git commit -m "docs(edpp): de-confounded Stage C ablation results after fix-cluster"
```

---

## Self-Review Notes

- **Spec coverage:** §2 fluid → Task 1; §3 censored remaining-steps + MaxBatchSize → Task 2; §4 oracle/deployable separation → Task 4; §5 little admission-rate + MaxBatchSize → Tasks 3 (rate) & 2 (MaxBatchSize); §6 prefill enrichment → Task 5; §7 CLI flag → Task 6; §8 validation → Task 7; §9 tests distributed. All mapped.
- **Confirm-in-situ (flagged, not placeholders):** the instance's max-running-requests accessor for `MaxBatchSize` (Task 2); the admission-counter hook point on the instance (Task 3); the captured-context structure from Stage C Task 7 for the separation + its test helpers (Task 4); how `Decide` builds the prefill context + a pool-aware spy (Task 5); the EDPP-config threading site for the flag (Task 6). Each names its anchor.
- **Type consistency:** `AdmissionContext`, `RunningReqState.TrueRemaining` (−1 sentinel), `RemainingStepsEst`, `AdmissionRate`, `MaxBatchSize`, `NewAdmissionEstimator`, `EDPPConfig.TAdmEstimator`, `trace.AdmissionRecord.TAdmPred*` used consistently with Stage C.
- **Censored floor is local to `Decide`** (Task 2) — global `nHatFor` unchanged (W_d out of scope), per Global Constraints.
- **Default `waiting` byte-identical** preserved (Tasks 1/2/5 assert `TestEDPP*` unchanged; new fields ignored by waiting).
