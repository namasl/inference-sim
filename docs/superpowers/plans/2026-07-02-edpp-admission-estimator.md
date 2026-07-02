# EDPP Occupancy-Aware Admission Estimator + Fidelity Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace EDPP's occupancy-blind admission-delay estimate with a pluggable estimator (waiting / little / fluid / rollforward, plus oracle measurement variants), and validate each per-pool on minimal isolation topologies via a new admission trace.

**Architecture:** Estimators run decider-side (they need `N̂_out`), consuming an `AdmissionContext` the `EDPPDecider` assembles from its own backlog/rate state plus an enriched `RoutingSnapshot` that exposes the running batch's per-request state. The decider computes all variants' predictions for logging; only a deployable variant drives routing. Two forced-routing topologies (T1 single decode engine / T2 1P+1D always-disagg) isolate `ttft_d` and `ttft_p` against single-engine realized admission delay.

**Tech Stack:** Go 1.22+ (`sim/`, `sim/cluster/`, `sim/trace/`, `cmd/`), Python 3 + pandas.

## Global Constraints

- The estimator replaces ONLY the `qD/muDec` (decode) and `qP/muPf` (prefill) admission terms in `ttft_d`/`ttft_p` (`sim/edpp.go`); the `+compute` term (`nChunks·(T_iter+δ)`) is unchanged.
- Default estimator is `waiting` → decider byte-identical to pre-Stage-C.
- Deployable variants (`waiting|little|fluid|rollforward`) are INV-9-safe (remaining decode steps = `N̂_out − progress`, never `OutputTokens`). Oracle variants (`fluid_oracle|rollforward_oracle`) read true remaining and are **logging-only** — selecting one as the routing driver is a fatal config error.
- Remaining-steps estimate uses the routing request's class `N̂_out` for all running requests (documented approximation, adequate for the uniform microbenchmark).
- INV-6 determinism (pure estimators, trace rows sorted by `request_id`, no RNG); INV-7 (new snapshot fields Periodic tier); INV-13 (new trace columns identical run vs replay).
- New per-request/per-snapshot state is gated: no allocation/work unless an occupancy estimator is the driver OR admission logging is enabled.
- `go test ./...` and `gofmt -l`/`go vet ./...` clean after each task (golangci-lint in CI).
- Work/time in µs.

---

### Task 1: Estimator interface, `AdmissionContext`, `waiting` impl, decider seam

**Files:**
- Create: `sim/admission_estimator.go`
- Modify: `sim/edpp.go` (Decide: build context, call estimator for `ttftD`/`ttftP` admission terms; add `tadmEstimator` field + config)
- Modify: `sim/edpp_coeffs.go` or `sim/edpp.go` (EDPPConfig: add `TAdmEstimator string`)
- Test: `sim/admission_estimator_test.go`

**Interfaces:**
- Produces: `type AdmissionContext struct { QWork, Mu float64; BatchSize, MaxBatchSize int; FreeKVBlocks, ReqKVNeed int64; TIter float64; QueueDepth int; AdmissionRate float64; RemainingStepsEst float64; Running []RunningReqState }`; `type AdmissionDelayEstimator interface { EstimateTAdm(ctx AdmissionContext) float64 }`; `func NewAdmissionEstimator(name string) (AdmissionDelayEstimator, error)`; `waitingEstimator`.
- `RunningReqState` defined here: `type RunningReqState struct { StepsDone, KVBlocks, TrueRemaining int64 }` (TrueRemaining = -1 when oracle not populated). Consumed by fluid/rollforward (Tasks 4/5).

- [ ] **Step 1: Write the failing test**

```go
package sim

import "testing"

func TestWaitingEstimator_ReproducesFormula(t *testing.T) {
	e, err := NewAdmissionEstimator("waiting")
	if err != nil { t.Fatal(err) }
	// waiting: QWork/Mu (the current admission term).
	got := e.EstimateTAdm(AdmissionContext{QWork: 5000, Mu: 0.5})
	if got != 10000 { t.Fatalf("waiting = %v, want 10000", got) }
}

func TestNewAdmissionEstimator_UnknownIsError(t *testing.T) {
	if _, err := NewAdmissionEstimator("nope"); err == nil {
		t.Fatal("expected error for unknown estimator")
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run 'TestWaitingEstimator|TestNewAdmissionEstimator' -v`
Expected: FAIL — undefined `NewAdmissionEstimator`/`AdmissionContext`.

- [ ] **Step 3: Create the interface + waiting impl**

Create `sim/admission_estimator.go`:

```go
package sim

import "fmt"

// RunningReqState is one running decode request's state for the roll-forward
// estimator. TrueRemaining is the oracle remaining step count (-1 when the
// oracle is not populated); StepsDone is decode steps completed; KVBlocks held.
type RunningReqState struct {
	StepsDone     int64
	KVBlocks      int64
	TrueRemaining int64
}

// AdmissionContext bundles everything an admission-delay estimator may read for
// one pool. The EDPPDecider assembles it from its backlog/rate state and the
// (possibly enriched) selected snapshot. Times/work in µs.
type AdmissionContext struct {
	QWork             float64 // waiting-backlog work (µs)
	Mu                float64 // occupancy-aware drain rate
	BatchSize         int
	MaxBatchSize      int
	FreeKVBlocks      int64
	ReqKVNeed         int64   // KV blocks this request needs
	TIter             float64 // occupancy-aware per-iteration time (µs)
	QueueDepth        int
	AdmissionRate     float64 // req/µs admitted at this pool (for little)
	RemainingStepsEst float64 // mean estimated remaining decode steps (N̂_out-based, for fluid)
	Running           []RunningReqState // per-running-request state (for rollforward)
}

// AdmissionDelayEstimator predicts the admission delay (µs) a request would incur
// at a pool given its current state. Pure function of ctx (INV-6).
type AdmissionDelayEstimator interface {
	EstimateTAdm(ctx AdmissionContext) float64
	Name() string
}

type waitingEstimator struct{}

func (waitingEstimator) Name() string { return "waiting" }
func (waitingEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.Mu <= 0 {
		return 0
	}
	return ctx.QWork / ctx.Mu
}

// NewAdmissionEstimator returns the estimator by name. Little/fluid/rollforward
// and the oracle variants are added in later tasks.
func NewAdmissionEstimator(name string) (AdmissionDelayEstimator, error) {
	switch name {
	case "", "waiting":
		return waitingEstimator{}, nil
	default:
		return nil, fmt.Errorf("unknown admission estimator %q", name)
	}
}
```

- [ ] **Step 4: Wire the seam in `Decide` (default preserves behavior)**

In `sim/edpp.go`, add a field `tadmEstimator AdmissionDelayEstimator` to `EDPPDecider` and construct it in `NewEDPPDecider` from `cfg.TAdmEstimator` (add that string field to `EDPPConfig`, default `""`→waiting; fatal on error). Replace the two admission terms:

```go
	// was: ttftP := qP/muPf + nChunks*(...)+CXfer ;  ttftD := qD/muDec + nChunks*(...)
	tAdmP := d.tadmEstimator.EstimateTAdm(AdmissionContext{QWork: qP, Mu: muPf}) // occupancy fields added in Task 3
	tAdmD := d.tadmEstimator.EstimateTAdm(AdmissionContext{QWork: qD, Mu: muDec})
	ttftP := tAdmP + nChunks*(d.coeffs.tIterPrefill(sPfPrefill)+deltaPfChunk) + float64(d.cfg.CXferUs)
	ttftD := tAdmD + nChunks*(tBminus1+deltaPfChunk)
```

With the default `waiting` estimator, `tAdmP==qP/muPf` and `tAdmD==qD/muDec` — byte-identical.

- [ ] **Step 5: Run tests + regression**

Run: `go test ./sim/... -run 'TestWaitingEstimator|TestNewAdmissionEstimator|TestEDPP' -v`
Expected: PASS; existing `TestEDPP*` unchanged (default waiting preserves values).

- [ ] **Step 6: Commit**

```bash
git add sim/admission_estimator.go sim/edpp.go sim/edpp_coeffs.go sim/admission_estimator_test.go
git commit -m "feat(edpp): pluggable admission estimator interface + waiting impl (default byte-identical)"
```

---

### Task 2: `little` estimator + admission-rate signal

**Files:**
- Modify: `sim/admission_estimator.go` (add `littleEstimator`; register in factory)
- Test: `sim/admission_estimator_test.go`

**Interfaces:**
- Consumes: `AdmissionContext.QueueDepth`, `AdmissionContext.AdmissionRate`.
- Produces: `littleEstimator` (name `"little"`).

- [ ] **Step 1: Write the failing test**

```go
func TestLittleEstimator(t *testing.T) {
	e, _ := NewAdmissionEstimator("little")
	// T_adm ≈ L̄_q / λ_adm : QueueDepth=8 waiting, AdmissionRate=0.002 req/µs → 4000µs.
	got := e.EstimateTAdm(AdmissionContext{QueueDepth: 8, AdmissionRate: 0.002})
	if got != 4000 { t.Fatalf("little = %v, want 4000", got) }
	// Zero admission rate → 0 (avoid div by zero; no signal).
	if e.EstimateTAdm(AdmissionContext{QueueDepth: 8, AdmissionRate: 0}) != 0 {
		t.Fatal("little with zero rate must be 0")
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestLittleEstimator -v`
Expected: FAIL — unknown estimator "little".

- [ ] **Step 3: Implement**

Add to `sim/admission_estimator.go` and register `case "little"` in the factory:

```go
type littleEstimator struct{}

func (littleEstimator) Name() string { return "little" }
func (littleEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.AdmissionRate <= 0 {
		return 0
	}
	return float64(ctx.QueueDepth) / ctx.AdmissionRate
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/ -run TestLittleEstimator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/admission_estimator.go sim/admission_estimator_test.go
git commit -m "feat(edpp): little admission estimator (L̄_q/λ_adm)"
```

---

### Task 3: Snapshot enrichment + context population in Decide

**Files:**
- Modify: `sim/routing.go` (add fields to `RoutingSnapshot`)
- Modify: `sim/simulator.go` (populate the running-batch state when enabled) and/or `sim/cluster/cluster.go` (snapshot build)
- Modify: `sim/edpp.go` (`Decide` fills the occupancy fields of `AdmissionContext` from the selected snapshot + `N̂_out`; `AdmissionRate` from snapshot)
- Test: `sim/edpp_test.go` (context population) + `sim/simulator_test.go` (snapshot fields)

**Interfaces:**
- Produces: `RoutingSnapshot.RunningDecode []RunningReqState`, `RoutingSnapshot.RemainingDecodeWork float64`, `RoutingSnapshot.AdmissionRate float64`, `RoutingSnapshot.MaxBatchSize` (already present). A `Simulator` gate `recordAdmissionDetail bool` + `SetAdmissionDetail()`.
- Consumes: Task 1 `AdmissionContext`, `RunningReqState`.

- [ ] **Step 1: Write the failing test (context population)**

```go
// In sim/edpp_test.go — Decide must fill AdmissionContext occupancy fields from the
// selected decode snapshot so fluid/rollforward have inputs. We assert via a spy estimator.
func TestDecide_PopulatesAdmissionContext(t *testing.T) {
	var seen AdmissionContext
	spy := admissionSpy{onCall: func(c AdmissionContext) { seen = c }}
	d := newTestEDPPDeciderWithEstimator(t, spy) // helper: constructs decider, injects spy as tadmEstimator
	state := &RouterState{
		SelectedInstance: "d0",
		Snapshots: []RoutingSnapshot{{
			ID: "d0", BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0,
			RemainingDecodeWork: 30, AdmissionRate: 0.001,
			RunningDecode: []RunningReqState{{StepsDone: 2, KVBlocks: 5, TrueRemaining: -1}},
		}},
	}
	d.Decide(makeReq("r1", 100, "batch"), state)
	if seen.BatchSize != 4 || seen.MaxBatchSize != 4 || seen.RemainingStepsEst == 0 || seen.AdmissionRate != 0.001 {
		t.Fatalf("context not populated from snapshot: %+v", seen)
	}
	if len(seen.Running) != 1 || seen.Running[0].StepsDone != 2 {
		t.Fatalf("running state not propagated: %+v", seen.Running)
	}
}
```

(Add `admissionSpy` and `newTestEDPPDeciderWithEstimator`/`makeReq` helpers in the test file; `RemainingStepsEst` derives from `RemainingDecodeWork`/`BatchSize` or `N̂_out`.)

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestDecide_PopulatesAdmissionContext -v`
Expected: FAIL — new snapshot fields / spy undefined.

- [ ] **Step 3: Add snapshot fields**

In `sim/routing.go` `RoutingSnapshot`, add:

```go
	RemainingDecodeWork float64           // Σ estimated remaining decode steps over running decode reqs (N̂_out-based); 0 if not populated
	AdmissionRate       float64           // req/µs admitted at this instance (for the little estimator); 0 if not available
	RunningDecode       []RunningReqState // per-running-decode-request state for the roll-forward estimator; nil unless admission detail enabled
```

- [ ] **Step 4: Populate them (gated)**

In the snapshot-build path (`sim/simulator.go:363` region / `sim/cluster/cluster.go:962`), when `recordAdmissionDetail` is set, populate `RunningDecode` by iterating the running batch (`StepsDone = ProgressIndex − len(InputTokens)`, `KVBlocks` from KV accounting, `TrueRemaining = len(OutputTokens) − StepsDone` ONLY when oracle logging on else −1), and `RemainingDecodeWork`/`AdmissionRate` from running/completion counters. Add `Simulator.SetAdmissionDetail(oracle bool)`. Zero-cost when unset.

- [ ] **Step 5: Fill the context in Decide**

In `sim/edpp.go` `Decide`, extend the two `AdmissionContext` literals with the occupancy fields from the selected decode snapshot (decode side) and prefill snapshots (prefill side): `BatchSize`, `MaxBatchSize`, `FreeKVBlocks`, `ReqKVNeed` (from `a_r`/block size), `TIter: tBminus1` (decode) / `tIterPrefill(sPfPrefill)` (prefill), `QueueDepth`, `AdmissionRate`, `RemainingStepsEst` (= `RemainingDecodeWork` or `max(N̂_out − mean StepsDone, 1)`), `Running: snap.RunningDecode`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `go test ./sim/... -run 'TestDecide_PopulatesAdmissionContext|TestEDPP' -v`
Expected: PASS; default-waiting decisions still byte-identical (waiting ignores the new fields).

- [ ] **Step 7: Commit**

```bash
git add sim/routing.go sim/simulator.go sim/cluster/cluster.go sim/edpp.go sim/edpp_test.go
git commit -m "feat(edpp): enrich RoutingSnapshot with running-batch state; populate AdmissionContext"
```

---

### Task 4: `fluid` estimator

**Files:**
- Modify: `sim/admission_estimator.go` (add `fluidEstimator`; register)
- Test: `sim/admission_estimator_test.go`

**Interfaces:**
- Consumes: `AdmissionContext.{BatchSize, MaxBatchSize, FreeKVBlocks, ReqKVNeed, TIter, RemainingStepsEst}`.
- Produces: `fluidEstimator` (name `"fluid"`).

- [ ] **Step 1: Write the failing test (incl. the ~905× bug case)**

```go
func TestFluidEstimator(t *testing.T) {
	e, _ := NewAdmissionEstimator("fluid")
	// Slot + KV already free → ~0.
	free := AdmissionContext{BatchSize: 2, MaxBatchSize: 4, FreeKVBlocks: 100, ReqKVNeed: 10, TIter: 1000, RemainingStepsEst: 20}
	if got := e.EstimateTAdm(free); got != 0 { t.Fatalf("free slot must give 0, got %v", got) }
	// Full batch, zero waiting work: waiting would give 0; fluid must give a large T_adm.
	// N_ahead=1 slot needed; X̂_dep = B/(R̄·T_iter) = 4/(20·1000)=2e-4 dep/µs → T_adm=1/2e-4=5000µs.
	full := AdmissionContext{BatchSize: 4, MaxBatchSize: 4, FreeKVBlocks: 0, ReqKVNeed: 10, TIter: 1000, RemainingStepsEst: 20}
	got := e.EstimateTAdm(full)
	if got < 4999 || got > 5001 { t.Fatalf("full-batch fluid = %v, want ~5000", got) }
	// Contrast: waiting on the same full/zero-waiting state gives 0 (the bug).
	w, _ := NewAdmissionEstimator("waiting")
	if w.EstimateTAdm(full) != 0 { t.Fatal("waiting must give 0 here (documents the bug fluid fixes)") }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestFluidEstimator -v`
Expected: FAIL — unknown estimator "fluid".

- [ ] **Step 3: Implement**

Add and register `case "fluid"`:

```go
type fluidEstimator struct{}

func (fluidEstimator) Name() string { return "fluid" }
func (fluidEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	// Admit next iteration if a slot AND enough KV already fit.
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return 0
	}
	// Occupancy-conditioned departure rate X̂_dep = B / (R̄ · T_iter) departures per µs.
	if ctx.RemainingStepsEst <= 0 || ctx.TIter <= 0 || ctx.BatchSize <= 0 {
		return 0
	}
	xDep := float64(ctx.BatchSize) / (ctx.RemainingStepsEst * ctx.TIter)
	if xDep <= 0 {
		return 0
	}
	// N_ahead: at least one departure needed for a slot; add KV-driven departures if KV-bound.
	nAhead := 1.0
	return nAhead / xDep
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/ -run TestFluidEstimator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/admission_estimator.go sim/admission_estimator_test.go
git commit -m "feat(edpp): fluid (occupancy-conditioned) admission estimator"
```

---

### Task 5: `rollforward` estimator (true per-request deterministic)

**Files:**
- Modify: `sim/admission_estimator.go` (add `rollforwardEstimator`; register)
- Test: `sim/admission_estimator_test.go`

**Interfaces:**
- Consumes: `AdmissionContext.{BatchSize, MaxBatchSize, FreeKVBlocks, ReqKVNeed, TIter, RemainingStepsEst, Running}`.
- Produces: `rollforwardEstimator` (name `"rollforward"`).

- [ ] **Step 1: Write the failing test (hand departure schedule)**

```go
func TestRollforwardEstimator(t *testing.T) {
	e, _ := NewAdmissionEstimator("rollforward")
	// Batch full (2/2), no free KV. Two running reqs with remaining steps 3 and 5,
	// holding 10 and 10 KV blocks. Request needs 8 blocks.
	// Roll forward at T_iter=1000µs: after 3 iters (3000µs) req A departs → frees 10 blocks
	// AND a slot. 10 ≥ 8 and slot free → admit. T_adm = 3·1000 = 3000µs.
	ctx := AdmissionContext{
		BatchSize: 2, MaxBatchSize: 2, FreeKVBlocks: 0, ReqKVNeed: 8, TIter: 1000,
		Running: []RunningReqState{{TrueRemaining: 3, KVBlocks: 10}, {TrueRemaining: 5, KVBlocks: 10}},
		RemainingStepsEst: 4,
	}
	got := e.EstimateTAdm(ctx)
	if got < 2999 || got > 3001 { t.Fatalf("rollforward = %v, want ~3000", got) }
}

func TestRollforwardEstimator_UsesEstimateWhenNoOracle(t *testing.T) {
	e, _ := NewAdmissionEstimator("rollforward")
	// TrueRemaining=-1 (no oracle) → use RemainingStepsEst for each running req.
	ctx := AdmissionContext{
		BatchSize: 1, MaxBatchSize: 1, FreeKVBlocks: 0, ReqKVNeed: 5, TIter: 1000,
		Running: []RunningReqState{{TrueRemaining: -1, KVBlocks: 10}}, RemainingStepsEst: 4,
	}
	// Single req departs after est 4 steps → frees slot+KV → T_adm = 4000µs.
	got := e.EstimateTAdm(ctx)
	if got < 3999 || got > 4001 { t.Fatalf("rollforward(est) = %v, want ~4000", got) }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestRollforwardEstimator -v`
Expected: FAIL — unknown estimator "rollforward".

- [ ] **Step 3: Implement**

Add and register `case "rollforward"`:

```go
type rollforwardEstimator struct{}

func (rollforwardEstimator) Name() string { return "rollforward" }
func (rollforwardEstimator) EstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return 0
	}
	// Deterministic look-ahead: each running req departs after its remaining steps
	// (oracle TrueRemaining if ≥0, else the N̂_out estimate), freeing its KV. Accumulate
	// elapsed = departureStep·T_iter until a slot AND enough free KV exist.
	type dep struct{ step, kv int64 }
	deps := make([]dep, 0, len(ctx.Running))
	for _, r := range ctx.Running {
		rem := r.TrueRemaining
		if rem < 0 {
			rem = int64(ctx.RemainingStepsEst)
			if rem < 1 { rem = 1 }
		}
		deps = append(deps, dep{step: rem, kv: r.KVBlocks})
	}
	// Sort by departure step ascending (soonest first).
	sortDepsByStep(deps)
	freeSlots := ctx.MaxBatchSize - ctx.BatchSize
	freeKV := ctx.FreeKVBlocks
	for _, d := range deps {
		freeSlots++
		freeKV += d.kv
		if freeSlots >= 1 && freeKV >= ctx.ReqKVNeed {
			return float64(d.step) * ctx.TIter
		}
	}
	// Batch never frees enough within its current occupants — cap at the last departure.
	if len(deps) > 0 {
		return float64(deps[len(deps)-1].step) * ctx.TIter
	}
	return 0
}

func sortDepsByStep(d []struct{ step, kv int64 }) {
	sort.Slice(d, func(i, j int) bool { return d[i].step < d[j].step })
}
```

(Ensure `sort` imported; adjust `sortDepsByStep` to the local `dep` type — inline the `sort.Slice` if the named-type helper is awkward.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./sim/ -run TestRollforwardEstimator -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add sim/admission_estimator.go sim/admission_estimator_test.go
git commit -m "feat(edpp): rollforward (true per-request deterministic) admission estimator"
```

---

### Task 6: Oracle variants + INV-9 routing guard

**Files:**
- Modify: `sim/admission_estimator.go` (oracle wrappers; factory; `IsOracle` / deployable check)
- Modify: `sim/edpp.go` or config validation (reject oracle as routing driver)
- Test: `sim/admission_estimator_test.go`

**Interfaces:**
- Produces: estimators `"fluid_oracle"`, `"rollforward_oracle"`; `func IsDeployableEstimator(name string) bool`.

- [ ] **Step 1: Write the failing test**

```go
func TestOracleVariants(t *testing.T) {
	// Oracle variants exist and use TrueRemaining even when an estimate is also present.
	e, err := NewAdmissionEstimator("rollforward_oracle")
	if err != nil { t.Fatal(err) }
	ctx := AdmissionContext{
		BatchSize: 1, MaxBatchSize: 1, FreeKVBlocks: 0, ReqKVNeed: 5, TIter: 1000,
		Running: []RunningReqState{{TrueRemaining: 2, KVBlocks: 10}}, RemainingStepsEst: 99,
	}
	// Oracle uses TrueRemaining=2 (not est 99) → 2000µs.
	if got := e.EstimateTAdm(ctx); got < 1999 || got > 2001 {
		t.Fatalf("rollforward_oracle = %v, want ~2000 (uses TrueRemaining)", got)
	}
}

func TestDeployableGuard(t *testing.T) {
	if IsDeployableEstimator("rollforward") != true { t.Fatal("rollforward is deployable") }
	if IsDeployableEstimator("rollforward_oracle") != false { t.Fatal("oracle is NOT deployable") }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run 'TestOracleVariants|TestDeployableGuard' -v`
Expected: FAIL — unknown oracle estimators / `IsDeployableEstimator` undefined.

- [ ] **Step 3: Implement oracle variants + guard**

The oracle variants force `TrueRemaining` usage. Simplest: the deployable `rollforward`/`fluid` already prefer `TrueRemaining` when `≥0`; the oracle variants are the same estimators run against a context whose `Running[].TrueRemaining` is populated (Task 3 oracle mode) and, for `fluid`, `RemainingStepsEst` set from the true mean. So register `"fluid_oracle"`→`fluidEstimator{}`, `"rollforward_oracle"`→`rollforwardEstimator{}` but tag them oracle for the guard and for context assembly (Task 7 passes true-remaining-populated context to the oracle-named loggers). Add:

```go
func IsDeployableEstimator(name string) bool {
	switch name {
	case "", "waiting", "little", "fluid", "rollforward":
		return true
	default:
		return false // oracle variants and unknowns
	}
}
```

Register the two oracle names in `NewAdmissionEstimator` (returning the same impls). In `NewEDPPDecider`/config validation, if `cfg.TAdmEstimator` is set and `!IsDeployableEstimator(cfg.TAdmEstimator)` → `logrus.Fatalf` ("oracle admission estimators are logging-only, not routing drivers").

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./sim/ -run 'TestOracleVariants|TestDeployableGuard' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/admission_estimator.go sim/edpp.go sim/admission_estimator_test.go
git commit -m "feat(edpp): oracle admission variants (logging-only) + INV-9 routing-driver guard"
```

---

### Task 7: `local_t_adm` capture + `--edpp-admission-trace` companion trace

**Files:**
- Modify: `sim/cluster/cluster.go` (local admission-time capture via existing `OnAdmit`/`localAdmitTimes`; `BuildAdmissionRecords`)
- Modify: `sim/trace/record.go` + Create `sim/trace/admission_csv.go`
- Modify: `cmd/root.go`, `cmd/replay.go` (flag + write wiring; enable admission detail + logging)
- Test: `sim/cluster/admission_trace_test.go`, `sim/trace/admission_csv_test.go`

**Interfaces:**
- Consumes: Task 1-6 estimators (`NewAdmissionEstimator` for each of the 6 names), `AdmissionContext` assembly (Task 3), Stage A's `localAdmitTimes`/`parentRequests`.
- Produces: `trace.AdmissionRecord` (fields: `RequestID, Pool string, RealizedTAdm float64, TAdmPredWaiting, TAdmPredLittle, TAdmPredFluid, TAdmPredRollforward, TAdmPredFluidOracle, TAdmPredRollforwardOracle float64`); `WriteAdmissionCSV`; `--edpp-admission-trace` flag; `cs.EnableAdmissionTrace(coeffs)`; `cs.BuildAdmissionRecords()`.

- [ ] **Step 1: Write the failing tests**

CSV writer test (`sim/trace/admission_csv_test.go`): header + row, mirroring `work_trace_csv_test.go` (assert header prefix `request_id,pool,realized_t_adm,t_adm_pred_waiting,` and a row). Cluster builder test (`sim/cluster/admission_trace_test.go`): given a stub of captured per-request realized `t_adm` + a recorded `AdmissionContext` per request, `BuildAdmissionRecords` emits one sorted row per request with all six predictions equal to each estimator's `EstimateTAdm(ctx)`, and `local_t_adm = local_schedule − local_enqueue` for a local request.

- [ ] **Step 2: Run to verify they fail**

Run: `go test ./sim/trace/ ./sim/cluster/ -run 'Admission' -v`
Expected: FAIL — undefined types/methods.

- [ ] **Step 3: Add the record + writer**

`trace.AdmissionRecord` in `record.go`; `WriteAdmissionCSV` in `admission_csv.go` mirroring `work_trace_csv.go` (header order: `request_id, pool, realized_t_adm, t_adm_pred_waiting, t_adm_pred_little, t_adm_pred_fluid, t_adm_pred_rollforward, t_adm_pred_fluid_oracle, t_adm_pred_rollforward_oracle`; floats via `FormatFloat(...'g',-1,64)`; no sort in writer).

- [ ] **Step 4: Capture local admission + record contexts**

In `sim/cluster/cluster.go`: (a) capture the local enqueue instant so `local_t_adm` is real (extend the Stage A `localAdmitTimes` capture to also store enqueue; or read `GatewayEnqueueTime`). (b) At each routing decision, when admission-trace enabled, snapshot the assembled `AdmissionContext` (decode + prefill) keyed by request/sub-request id (so predictions can be recomputed at end for all six estimators). (c) `BuildAdmissionRecords()` walks parents + locals, computes realized `t_adm` (reuse Stage A correlation), runs all six `NewAdmissionEstimator(name).EstimateTAdm(ctx)`, emits sorted rows.

- [ ] **Step 5: Wire the flag (run + replay)**

`--edpp-admission-trace <path>` in `cmd/root.go` (registration + var), `cs.EnableAdmissionTrace(coeffs)` when set (also flips `SetAdmissionDetail(oracle=true)` so `RunningDecode`/`TrueRemaining` populate), shared write helper mirroring `writeWorkTrace`, mirrored in `cmd/replay.go` (INV-13).

- [ ] **Step 6: Build, smoke, full suite**

```bash
go build -o blis main.go && ./blis run --help | grep edpp-admission-trace
go test ./... 2>&1 | tail -20 && gofmt -l sim/ cmd/ | grep -v 'pre-existing' || true
```
Expected: flag present; tests PASS.

- [ ] **Step 7: Commit**

```bash
git add sim/cluster/cluster.go sim/trace/record.go sim/trace/admission_csv.go cmd/root.go cmd/replay.go sim/cluster/admission_trace_test.go sim/trace/admission_csv_test.go
git commit -m "feat(edpp): --edpp-admission-trace + local_t_adm capture + per-request all-variant predictions"
```

---

### Task 8: Microbenchmark repro + ablation analysis + FINDINGS

**Files:**
- Create: `campaigns/edpp-study/repro_stage_c.sh`
- Create: `campaigns/edpp-study/analyze/admission_ablation.py`
- Modify: `campaigns/edpp-study/FINDINGS.md`, `campaigns/edpp-study/README.md` (add Stage C + admission-trace columns)

**Interfaces:** none (verification + docs).

- [ ] **Step 1: Write the repro script (T1 + T2)**

`repro_stage_c.sh`: build blis; **T1** — single decode-capable instance, `--pd-decider never`, a saturating synth spec, `--edpp-admission-trace out/stage_c/t1_admission.csv`; **T2** — `--num-instances 2 --prefill-instances 1 --decode-instances 1 --pd-decider always`, same spec, `--edpp-admission-trace out/stage_c/t2_admission.csv`. Both need `--edpp-coeffs`. Then run the analysis on each. (Routing is forced in both, so the driving `--edpp-tadm-estimator` is immaterial — all six predictions are logged regardless; default `waiting` is fine as the driver.)

- [ ] **Step 2: Write the analysis**

`admission_ablation.py --admission <csv> [--out json]`: per `pool`, over completed requests, compute for each of the six prediction columns the median ratio `realized_t_adm / pred` and mean/median signed error; and the decomposition `realized − rollforward_oracle` (form error) and `rollforward_oracle − rollforward` (N̂_out error). Emit a JSON table. Guard divide-by-zero/inf/empty (mirror `work_model_validation.py`).

- [ ] **Step 3: Run end-to-end**

```bash
bash campaigns/edpp-study/repro_stage_c.sh
cat campaigns/edpp-study/out/stage_c/*_ablation.json
```
Expected: the median ratio collapses `waiting → little → fluid → rollforward`; `rollforward_oracle` ≈ 1×; `waiting` shows the large (~hundreds×) miss on the saturated pool. If `waiting`'s miss does NOT reproduce, the operating point is not saturating — increase `aggregate_rate` in the spec.

- [ ] **Step 4: Write FINDINGS + README**

Append FINDINGS "Stage C" (the ablation tables for T1 decode-pool and T2 prefill+decode-pool, the error decomposition, the reproduce command + checkpoint, and the trained-physics/single-operating-point limitations). Add the `--edpp-admission-trace` column dictionary and the Stage C repro to `README.md`.

- [ ] **Step 5: Commit**

```bash
git add campaigns/edpp-study/repro_stage_c.sh campaigns/edpp-study/analyze/admission_ablation.py campaigns/edpp-study/FINDINGS.md campaigns/edpp-study/README.md
git commit -m "docs(edpp): Stage C admission-estimator ablation repro + findings"
```

---

## Self-Review Notes

- **Spec coverage:** §2 seam → Task 1. §3 estimators: waiting T1, little T2, fluid T4, rollforward T5, oracle+guard T6. §4 enrichment → Task 3. §5 topologies + §5c logging → Task 8 (T1/T2) + Task 7 (trace). §5b local_t_adm → Task 7. §6 ablation+decomposition → Task 8. §7 invariants: INV-9 guard T6, default-waiting regression T1/T3, INV-6/13 T7. §8 tests distributed. §9 deliverables all mapped. §10 limits → Task 8 FINDINGS.
- **Confirm-in-situ (flagged, not placeholders):** exact snapshot-build site to populate `RunningDecode` (`simulator.go:363` / `cluster.go:962`); how KV blocks per running request are read; the `EDPPConfig`/`NewEDPPDecider` construction site for the estimator field + guard; the deployment coeffs var (`config.EDPPCoeffs`, per Stage B) at the admission-trace enable sites; test helpers `admissionSpy`/`newTestEDPPDeciderWithEstimator`/`makeReq` follow existing `sim/edpp_test.go` patterns.
- **Type consistency:** `AdmissionContext`, `RunningReqState`, `AdmissionDelayEstimator`, `NewAdmissionEstimator`, `IsDeployableEstimator`, `AdmissionRecord`, `EnableAdmissionTrace`, `BuildAdmissionRecords` used identically across tasks. Estimator names (`waiting|little|fluid|rollforward|fluid_oracle|rollforward_oracle`) consistent in factory (T1/2/4/5/6), guard (T6), trace columns (T7), analysis (T8).
- **Default-preserves-behavior** asserted in Task 1 and Task 3 (waiting ignores new fields → byte-identical); oracle-as-driver rejected (Task 6). No byte-identical requirement for non-waiting variants (they change decisions by design — but in T1/T2 routing is forced, so even they don't alter routing there).
