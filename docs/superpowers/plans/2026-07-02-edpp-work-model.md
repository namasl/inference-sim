# EDPP Work Model Correction + Per-Request Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the EDPP prefill/decode work formulas to equal the work the active latency model actually charges, and validate them per-request against realized trajectory work via a new `--edpp-work-trace` CSV.

**Architecture:** (1) Fix `Wp`/`Wd` in `sim/edpp_coeffs.go` and their call sites in `sim/edpp.go` — measurement of *decisions* changes, that's expected. (2) A per-request work accumulator in the per-instance `Simulator` mirrors the active latency model's per-step charge; the cluster correlates prefill/decode sub-request accumulators back to the parent (reusing `parentRequests`, like Stage A), computes closed forms with realized inputs, and emits one row per request. (3) A Python analyzer reports realized-vs-closed error.

**Tech Stack:** Go 1.22+ (`sim/`, `sim/cluster/`, `sim/trace/`, `cmd/`), Python 3 + pandas.

## Global Constraints

- Same frozen coefficients (`C0, C1, C_pf, C_attn, α`) — no refit.
- `Wp(a_p, a_r) = C_pf·a_p + C_attn·a_p·(a_r + a_p/2)` — the trained-physics basis (`+ a_p/2`), matching the active latency model; NOT §3.6's causal `− a_p/2`.
- `Wd(a_r, o) = C0·o + C1·o·(a_r + (o−1)/2)` — exact discrete decode sum.
- The accumulator MUST mirror the active latency model's per-step charge verbatim: prefill `C_pf·s_r + C_attn·s_r·(a_r + s_r/2)` with `a_r = len(InputTokens)` (full); decode `C0 + C1·ProgressIndex`.
- Zero cost when `--edpp-work-trace` unset (gate all new work/allocation on the flag).
- INV-6 determinism: work-trace CSV byte-identical across seeded runs; rows sorted by `request_id`.
- INV-13 run/replay parity: same config ⇒ identical work-trace CSV.
- Do NOT assert byte-identical synth regression — the attention term changes ~3×; the decision shift is expected and recorded, not prevented.
- `go test ./...` and `gofmt -l`/`go vet ./...` clean after every task (golangci-lint is not installed locally; run it in CI).
- All work values in µs (the coeff units).

---

### Task 1: Correct `Wp` and add `Wd` in the coeffs, update decider call sites

**Files:**
- Modify: `sim/edpp_coeffs.go` (`Wp` at line 62; add `Wd`)
- Modify: `sim/edpp.go` (`Decide` wp at line 385; `OnRoute` wp+wd at lines 563-564)
- Test: `sim/edpp_coeffs_test.go`

**Interfaces:**
- Produces: `func (c EDPPCoeffs) Wp(ap, ar int) float64`; `func (c EDPPCoeffs) Wd(ar int, o float64) float64`.
- Consumes: existing `EDPPCoeffs` fields `CPf, CAttn, C0, C1`; `d.nHatFor(class).mean()`.

- [ ] **Step 1: Write the failing coeff tests**

Add to `sim/edpp_coeffs_test.go`:

```go
func TestWp_TrainedPhysicsBasis(t *testing.T) {
	c := EDPPCoeffs{CPf: 6.0, CAttn: 0.001}
	// No cache (a_p = a_r = 1000): C_pf·1000 + C_attn·1000·(1000 + 500) = 6000 + 0.001·1000·1500 = 6000 + 1500.
	got := c.Wp(1000, 1000)
	want := 6.0*1000 + 0.001*1000*(1000+1000.0/2)
	if got != want {
		t.Fatalf("Wp(1000,1000) = %v, want %v", got, want)
	}
	// This is NOT the old (C_attn/2)·a² form: old attention would be 0.001/2·1000² = 500, not 1500.
	oldAttn := (0.001 / 2) * 1000 * 1000
	newAttn := got - 6.0*1000
	if newAttn == oldAttn {
		t.Fatalf("Wp attention term must differ from old (C_attn/2)a²=%v; got %v", oldAttn, newAttn)
	}
	// Cached prefix (a_p=200 uncached of a_r=1000): C_pf·200 + C_attn·200·(1000 + 100).
	gotc := c.Wp(200, 1000)
	wantc := 6.0*200 + 0.001*200*(1000+200.0/2)
	if gotc != wantc {
		t.Fatalf("Wp(200,1000) = %v, want %v", gotc, wantc)
	}
}

func TestWd_DiscreteDecodeSum(t *testing.T) {
	c := EDPPCoeffs{C0: 5.0, C1: 0.05}
	// Wd = Σ_{k=0}^{o-1}(C0 + C1·(a_r+k)) = C0·o + C1·o·(a_r + (o-1)/2).
	sum := func(ar int, o int) float64 {
		var s float64
		for k := 0; k < o; k++ {
			s += 5.0 + 0.05*float64(ar+k)
		}
		return s
	}
	for _, tc := range []struct{ ar, o int }{{1000, 0}, {1000, 1}, {500, 10}, {2000, 128}} {
		got := c.Wd(tc.ar, float64(tc.o))
		want := sum(tc.ar, tc.o)
		if diff := got - want; diff < -1e-6 || diff > 1e-6 {
			t.Fatalf("Wd(%d,%d) = %v, want %v (discrete sum)", tc.ar, tc.o, got, want)
		}
	}
	if c.Wd(1000, 0) != 0 {
		t.Fatalf("Wd with o=0 must be 0, got %v", c.Wd(1000, 0))
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./sim/ -run 'TestWp_TrainedPhysicsBasis|TestWd_DiscreteDecodeSum' -v`
Expected: FAIL — `Wp` takes 1 arg (compile error), `Wd` undefined.

- [ ] **Step 3: Correct `Wp` and add `Wd`**

In `sim/edpp_coeffs.go`, replace the `Wp` method:

```go
// Wp is the prefill demand of a_p uncached tokens for a prompt of full length
// a_r, in µs. It is the trajectory sum of the active (trained-physics) latency
// model's per-step prefill charge C_pf·s + C_attn·s·(a_r + s/2) — hence the
// (a_r + a_p/2) form (see docs/superpowers/specs/2026-07-01-edpp-work-model-design.md
// §2, §7). At a_p = a_r (no cache) this is C_pf·a_r + 1.5·C_attn·a_r².
func (c EDPPCoeffs) Wp(ap, ar int) float64 {
	a := float64(ap)
	r := float64(ar)
	return c.CPf*a + c.CAttn*a*(r+a/2.0)
}

// Wd is the decode demand for a prompt of length a_r generating o output tokens,
// in µs: the exact discrete per-step sum Σ_{k=0}^{o-1}(C0 + C1·(a_r+k)) =
// C0·o + C1·o·(a_r + (o-1)/2). Matches the active latency model's per-decode-step
// charge (context = ProgressIndex = a_r + k). o is the N̂_out estimate at routing.
func (c EDPPCoeffs) Wd(ar int, o float64) float64 {
	if o <= 0 {
		return 0
	}
	r := float64(ar)
	return c.C0*o + c.C1*o*(r+(o-1)/2.0)
}
```

- [ ] **Step 4: Update the decider call sites**

In `sim/edpp.go` `Decide` (line 385), pass `a_r`:

```go
	wp := d.coeffs.Wp(ap, len(req.InputTokens))
```

In `sim/edpp.go` `OnRoute` (lines 563-564), pass `a_r` and use `Wd`:

```go
	wp := d.coeffs.Wp(apTokens, len(req.InputTokens))
	wd := d.coeffs.Wd(len(req.InputTokens), d.nHatFor(req.SLOClass).mean())
```

(Leave the `NomDecodeCtx` field and `deltaBarDecode`/`selectedDecodeState` fallback as-is; they are no longer on the `W_d` path. Add a one-line comment at the old `wd` site noting `W_d` no longer uses `NomDecodeCtx`.)

- [ ] **Step 5: Run tests to verify they pass + build**

Run: `go test ./sim/ -run 'TestWp_TrainedPhysicsBasis|TestWd_DiscreteDecodeSum' -v && go build ./...`
Expected: PASS; build clean. Existing `sim` tests that call `Wp(x)` with one arg will fail to compile — update those call sites to `Wp(x, x)` (no-cache) and note them in the report; their numeric expectations may change (attention term ~3×) — update expected values to the corrected basis, do NOT revert the formula.

- [ ] **Step 6: Run the full sim package tests**

Run: `go test ./sim/... 2>&1 | tail -20`
Expected: PASS. Any failures are pre-existing tests asserting the OLD `Wp` value — update their expected numbers to the corrected basis (this is the intended behavior change), and list each in the report.

- [ ] **Step 7: Commit**

```bash
git add sim/edpp_coeffs.go sim/edpp.go sim/edpp_coeffs_test.go
git commit -m "feat(edpp): correct Wp to trained-physics basis and add discrete Wd"
```

---

### Task 2: WorkTraceRecord type + CSV writer

**Files:**
- Modify: `sim/trace/record.go` (add `WorkTraceRecord`)
- Create: `sim/trace/work_trace_csv.go`
- Test: `sim/trace/work_trace_csv_test.go`

**Interfaces:**
- Produces: `trace.WorkTraceRecord` (fields below); `func WriteWorkTraceCSV(w io.Writer, records []WorkTraceRecord) error`.

- [ ] **Step 1: Write the failing test**

Create `sim/trace/work_trace_csv_test.go`:

```go
package trace

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteWorkTraceCSV_HeaderAndRow(t *testing.T) {
	recs := []WorkTraceRecord{{
		RequestID: "r1", SLOClass: "batch", Ar: 1000, ApRealized: 1000, ORealized: 50,
		PrefillChunks: 1, CacheHitFrac: 0.0,
		RealizedPrefillWork: 1506000, RealizedDecodeWork: 313725,
		WpClosed: 1506000, WdClosed: 313725, WpClosedNoCacheOld: 506000,
	}}
	var buf bytes.Buffer
	if err := WriteWorkTraceCSV(&buf, recs); err != nil {
		t.Fatalf("WriteWorkTraceCSV: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want header + 1 row, got %d:\n%s", len(lines), buf.String())
	}
	if !strings.HasPrefix(lines[0], "request_id,slo_class,a_r,a_p_realized,o_r_realized,prefill_chunks,cache_hit_frac,") {
		t.Fatalf("unexpected header: %s", lines[0])
	}
	if !strings.Contains(lines[1], "r1,batch,1000,1000,50,1,0,") {
		t.Fatalf("unexpected row: %s", lines[1])
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/trace/ -run TestWriteWorkTraceCSV -v`
Expected: FAIL — `WorkTraceRecord` / `WriteWorkTraceCSV` undefined.

- [ ] **Step 3: Add the record struct**

In `sim/trace/record.go`, add:

```go
// WorkTraceRecord is one request's realized trajectory work vs the closed-form
// work model (Stage B validation). Times/work in µs. Emitted only under
// --edpp-work-trace. See docs/superpowers/specs/2026-07-01-edpp-work-model-design.md.
type WorkTraceRecord struct {
	RequestID     string
	SLOClass      string
	Ar            int64   // full prompt length len(InputTokens)
	ApRealized    int64   // Σ new prefill tokens actually processed (excludes cached prefix)
	ORealized     int64   // realized output length (decode steps)
	PrefillChunks int     // number of prefill steps (1 = single-chunk)
	CacheHitFrac  float64 // 1 - ApRealized/Ar

	RealizedPrefillWork float64 // Σ per-step prefill δ (active latency model basis)
	RealizedDecodeWork  float64 // Σ per-step decode δ

	WpClosed           float64 // Wp(ApRealized, Ar) — corrected closed form
	WdClosed           float64 // Wd(Ar, ORealized) — corrected closed form
	WpClosedNoCacheOld float64 // old shipped form C_pf·ApRealized + (C_attn/2)·ApRealized² (for delta reporting)
}
```

- [ ] **Step 4: Add the CSV writer**

Create `sim/trace/work_trace_csv.go`:

```go
package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

var workTraceCSVHeader = []string{
	"request_id", "slo_class", "a_r", "a_p_realized", "o_r_realized",
	"prefill_chunks", "cache_hit_frac",
	"realized_prefill_work", "realized_decode_work",
	"wp_closed", "wd_closed", "wp_closed_nocache_old",
}

// WriteWorkTraceCSV writes realized-vs-closed work records to w as CSV (header +
// one row per record). Callers pass records pre-sorted by request_id for
// deterministic output (INV-6). Consumed by --edpp-work-trace / work_model_validation.py.
func WriteWorkTraceCSV(w io.Writer, records []WorkTraceRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(workTraceCSVHeader); err != nil {
		return err
	}
	i := func(v int64) string { return strconv.FormatInt(v, 10) }
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, r.SLOClass, i(r.Ar), i(r.ApRealized), i(r.ORealized),
			strconv.Itoa(r.PrefillChunks), f(r.CacheHitFrac),
			f(r.RealizedPrefillWork), f(r.RealizedDecodeWork),
			f(r.WpClosed), f(r.WdClosed), f(r.WpClosedNoCacheOld),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `go test ./sim/trace/ -run TestWriteWorkTraceCSV -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add sim/trace/record.go sim/trace/work_trace_csv.go sim/trace/work_trace_csv_test.go
git commit -m "feat(edpp): add WorkTraceRecord and CSV writer"
```

---

### Task 3: Per-request work accumulator in the Simulator

**Files:**
- Modify: `sim/simulator.go` (struct fields near line 134; init near line 200; accumulate in `executeBatchStep` ~line 773; getter)
- Test: `sim/simulator_work_accum_test.go` (new)

**Interfaces:**
- Consumes: `EDPPCoeffs` (the accumulator needs `C0,C1,CPf,CAttn`); `req.InputTokens`, `req.NumNewTokens`, `req.ProgressIndex`, `req.OutputTokens`, `req.SLOClass`, `req.ID`.
- Produces: on `Simulator` — `recordWorkTrace bool`, `workCoeffs EDPPCoeffs`, `workAcc map[string]*reqWorkAccum`, a setter `SetWorkTrace(coeffs EDPPCoeffs)`, and getter `func (sim *Simulator) WorkAccumulators() map[string]ReqWork`. `ReqWork` is an exported snapshot struct.

- [ ] **Step 1: Write the failing test**

Create `sim/simulator_work_accum_test.go`:

```go
package sim

import "testing"

// The accumulator must sum per-step δ (active latency-model basis) to the analytic
// closed form for a single request driven through prefill then decode.
func TestWorkAccumulator_SumsToClosedForm(t *testing.T) {
	c := EDPPCoeffs{C0: 5.0, C1: 0.05, CPf: 6.0, CAttn: 0.001}
	sim := &Simulator{workAcc: map[string]*reqWorkAccum{}, recordWorkTrace: true, workCoeffs: c}

	ar := 100
	// Single-chunk prefill: one step processes all ar tokens (ProgressIndex 0 → ar).
	sim.accumulateStepWork("r1", "batch", &Request{
		ID: "r1", SLOClass: "batch",
		InputTokens: make([]int, ar), NumNewTokens: ar, ProgressIndex: 0,
	})
	// 3 decode steps at ProgressIndex ar, ar+1, ar+2.
	for k := 0; k < 3; k++ {
		sim.accumulateStepWork("r1", "batch", &Request{
			ID: "r1", SLOClass: "batch",
			InputTokens: make([]int, ar), OutputTokens: make([]int, 3),
			NumNewTokens: 1, ProgressIndex: int64(ar + k),
		})
	}
	got := sim.WorkAccumulators()["r1"]

	wantPrefill := 6.0*float64(ar) + 0.001*float64(ar)*(float64(ar)+float64(ar)/2.0) // single chunk
	if diff := got.RealizedPrefillWork - wantPrefill; diff < -1e-6 || diff > 1e-6 {
		t.Fatalf("prefill work = %v, want %v", got.RealizedPrefillWork, wantPrefill)
	}
	var wantDecode float64
	for k := 0; k < 3; k++ {
		wantDecode += 5.0 + 0.05*float64(ar+k)
	}
	if diff := got.RealizedDecodeWork - wantDecode; diff < -1e-6 || diff > 1e-6 {
		t.Fatalf("decode work = %v, want %v", got.RealizedDecodeWork, wantDecode)
	}
	if got.Ar != int64(ar) || got.ApRealized != int64(ar) || got.ORealized != 3 || got.PrefillChunks != 1 {
		t.Fatalf("bad accum meta: %+v", got)
	}
}

func TestWorkAccumulator_DisabledNoAlloc(t *testing.T) {
	sim := &Simulator{recordWorkTrace: false}
	sim.accumulateStepWork("r1", "batch", &Request{ID: "r1", InputTokens: make([]int, 10), NumNewTokens: 10})
	if sim.workAcc != nil {
		t.Fatalf("workAcc must stay nil when disabled, got %v", sim.workAcc)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestWorkAccumulator -v`
Expected: FAIL — `reqWorkAccum`, `accumulateStepWork`, `WorkAccumulators`, `recordWorkTrace`, `workCoeffs` undefined.

- [ ] **Step 3: Add the accumulator types, fields, and method**

In `sim/simulator.go`, add near the `stepRec` field (~line 137):

```go
	// Work-trace accumulator (off unless --edpp-work-trace). Sums each resident
	// request's per-step δ (active latency-model basis) into per-request totals,
	// so realized trajectory work can be compared to the closed-form W_p/W_d.
	recordWorkTrace bool
	workCoeffs      EDPPCoeffs
	workAcc         map[string]*reqWorkAccum
```

Add the types and method (new region in `sim/simulator.go`, e.g. after `executeBatchStep`):

```go
// reqWorkAccum accumulates one request's realized trajectory work.
type reqWorkAccum struct {
	slo           string
	ar            int64
	apRealized    int64
	oRealized     int64
	prefillChunks int
	prefillWork   float64
	decodeWork    float64
}

// ReqWork is an exported snapshot of a request's accumulated work (for the cluster
// builder / --edpp-work-trace).
type ReqWork struct {
	SLOClass            string
	Ar                  int64
	ApRealized          int64
	ORealized           int64
	PrefillChunks       int
	RealizedPrefillWork float64
	RealizedDecodeWork  float64
}

// SetWorkTrace enables per-request work accumulation with the given coeffs.
func (sim *Simulator) SetWorkTrace(coeffs EDPPCoeffs) {
	sim.recordWorkTrace = true
	sim.workCoeffs = coeffs
	if sim.workAcc == nil {
		sim.workAcc = make(map[string]*reqWorkAccum)
	}
}

// accumulateStepWork adds one scheduled request's per-step δ to its accumulator,
// mirroring the active latency model's charge (prefill C_pf·s + C_attn·s·(a_r+s/2)
// with a_r = full input length; decode C0 + C1·ProgressIndex). No-op when disabled.
func (sim *Simulator) accumulateStepWork(id, slo string, req *Request) {
	if !sim.recordWorkTrace {
		return
	}
	a := sim.workAcc[id]
	if a == nil {
		a = &reqWorkAccum{slo: slo, ar: util.Len64(req.InputTokens)}
		sim.workAcc[id] = a
	}
	si := util.Len64(req.InputTokens)
	if req.ProgressIndex < si {
		s := float64(req.NumNewTokens)
		a.prefillWork += sim.workCoeffs.CPf*s + sim.workCoeffs.CAttn*s*(float64(si)+s/2.0)
		a.apRealized += int64(req.NumNewTokens)
		a.prefillChunks++
	} else if len(req.OutputTokens) > 0 {
		a.decodeWork += sim.workCoeffs.C0 + sim.workCoeffs.C1*float64(req.ProgressIndex)
		a.oRealized++
	}
}

// WorkAccumulators returns a snapshot of accumulated per-request work (empty when disabled).
func (sim *Simulator) WorkAccumulators() map[string]ReqWork {
	out := make(map[string]ReqWork, len(sim.workAcc))
	for id, a := range sim.workAcc {
		out[id] = ReqWork{
			SLOClass: a.slo, Ar: a.ar, ApRealized: a.apRealized, ORealized: a.oRealized,
			PrefillChunks: a.prefillChunks,
			RealizedPrefillWork: a.prefillWork, RealizedDecodeWork: a.decodeWork,
		}
	}
	return out
}
```

- [ ] **Step 4: Hook the accumulator into `executeBatchStep`**

In `sim/simulator.go` `executeBatchStep`, right after the `sim.stepRec` block (after line 787), add (iterating the same `scheduled` slice):

```go
	if sim.recordWorkTrace {
		for _, req := range scheduled {
			sim.accumulateStepWork(req.ID, req.SLOClass, req)
		}
	}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./sim/ -run TestWorkAccumulator -v`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
git add sim/simulator.go sim/simulator_work_accum_test.go
git commit -m "feat(edpp): per-request work accumulator mirroring active latency model"
```

---

### Task 4: Cluster builder + `--edpp-work-trace` flag (run + replay)

**Files:**
- Modify: `sim/cluster/cluster.go` (add `BuildWorkTraceRecords`; expose per-instance `Simulator` access — reuse the existing instance iteration)
- Modify: `cmd/root.go` (flag var, registration, set-on-clusters, write at run end)
- Modify: `cmd/replay.go` (mirror write)
- Test: `sim/cluster/work_trace_test.go` (new)

**Interfaces:**
- Consumes: `Simulator.WorkAccumulators() map[string]ReqWork` and `Simulator.SetWorkTrace(EDPPCoeffs)` (Task 3); `trace.WorkTraceRecord` + `trace.WriteWorkTraceCSV` (Task 2); `EDPPCoeffs.Wp/Wd` (Task 1); the cluster's `parentRequests` (parent has `PrefillSubReqID`, `DecodeSubReqID`, `PrefillInstanceID`, `DecodeInstanceID`, `OriginalRequest`); the per-instance simulators (each instance exposes `.sim` — confirm the field/accessor name in situ, same one used at `cluster.go:601` for `OnAdmit`).
- Produces: `func (cs *ClusterSimulator) BuildWorkTraceRecords() []trace.WorkTraceRecord` (sorted by request_id); `func (cs *ClusterSimulator) EnableWorkTrace(EDPPCoeffs)`; CLI flag `--edpp-work-trace <path>`.

**Correlation note (critical):** for a disaggregated request the prefill sub-request ran on the prefill instance's `Simulator` keyed by `parent.PrefillSubReqID`, and the decode sub-request on the decode instance keyed by `parent.DecodeSubReqID`. Merge them into one record per parent. For a non-disaggregated (local) request there is no parent; the base request ID on its handling instance carries both prefill and decode work. Build a combined `map[instanceID]map[reqID]ReqWork` first (from every instance's `WorkAccumulators()`), then:
- For each parent in `parentRequests`: prefill work from `byInst[PrefillInstanceID][PrefillSubReqID]`, decode work from `byInst[DecodeInstanceID][DecodeSubReqID]`; `a_r`/`slo` from `parent.OriginalRequest`; `a_p_realized`/`prefill_chunks` from the prefill side; `o_realized` from the decode side.
- For each `(instance, id)` not claimed by any parent (and not a `_prefill`/`_decode` sub-request id): emit as a local request with both works from that single accumulator.

- [ ] **Step 1: Write the failing builder test**

Create `sim/cluster/work_trace_test.go`:

```go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Local (non-disagg) request: one accumulator carries both prefill and decode work,
// and the builder computes closed forms with the corrected coeffs.
func TestBuildWorkTraceRecords_Local(t *testing.T) {
	c := sim.EDPPCoeffs{C0: 5.0, C1: 0.05, CPf: 6.0, CAttn: 0.001}
	cs := &ClusterSimulator{
		parentRequests: map[string]*ParentRequest{},
		workCoeffs:     c,
		recordWorkTrace: true,
	}
	// Inject a stub per-instance work snapshot (the builder consumes this map).
	cs.workByInstance = map[string]map[string]sim.ReqWork{
		"instance_0": {"r1": {
			SLOClass: "batch", Ar: 100, ApRealized: 100, ORealized: 3, PrefillChunks: 1,
			RealizedPrefillWork: 6.0*100 + 0.001*100*(100+50), RealizedDecodeWork: 5*3 + 0.05*(100+101+102),
		}},
	}
	recs := cs.buildWorkTraceRecordsFrom(cs.workByInstance)
	if len(recs) != 1 || recs[0].RequestID != "r1" {
		t.Fatalf("want 1 record for r1, got %+v", recs)
	}
	r := recs[0]
	if r.WpClosed != c.Wp(100, 100) || r.WdClosed != c.Wd(100, 3) {
		t.Fatalf("closed forms wrong: WpClosed=%v (want %v) WdClosed=%v (want %v)",
			r.WpClosed, c.Wp(100, 100), r.WdClosed, c.Wd(100, 3))
	}
	if r.WpClosedNoCacheOld != 6.0*100+(0.001/2)*100*100 {
		t.Fatalf("old-form column wrong: %v", r.WpClosedNoCacheOld)
	}
}
```

(This test exercises the pure record-building logic through a seam `buildWorkTraceRecordsFrom(map[string]map[string]sim.ReqWork)` so it needs no full cluster. `BuildWorkTraceRecords()` is the thin wrapper that gathers `workByInstance` from live instances then calls it. Add `workCoeffs sim.EDPPCoeffs`, `recordWorkTrace bool`, and `workByInstance map[string]map[string]sim.ReqWork` fields to `ClusterSimulator`.)

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/cluster/ -run TestBuildWorkTraceRecords -v`
Expected: FAIL — fields/methods undefined.

- [ ] **Step 3: Add cluster fields, gather, and builder**

In `sim/cluster/cluster.go`, add fields to `ClusterSimulator`:

```go
	recordWorkTrace bool
	workCoeffs      sim.EDPPCoeffs
	workByInstance  map[string]map[string]sim.ReqWork // instanceID → reqID → work (Stage B work trace)
```

Add:

```go
// EnableWorkTrace turns on per-request work accumulation on every instance simulator.
func (cs *ClusterSimulator) EnableWorkTrace(coeffs sim.EDPPCoeffs) {
	cs.recordWorkTrace = true
	cs.workCoeffs = coeffs
	for _, inst := range cs.instances { // confirm the instance-slice field/accessor name in situ
		inst.sim.SetWorkTrace(coeffs)
	}
}

// gatherWorkByInstance snapshots each instance's per-request work accumulators.
func (cs *ClusterSimulator) gatherWorkByInstance() map[string]map[string]sim.ReqWork {
	out := make(map[string]map[string]sim.ReqWork)
	for _, inst := range cs.instances {
		out[string(inst.id)] = inst.sim.WorkAccumulators() // confirm inst.id / inst.sim names in situ
	}
	return out
}

// BuildWorkTraceRecords gathers live instance accumulators and correlates
// prefill/decode sub-requests back to parents. Sorted by request_id (INV-6).
func (cs *ClusterSimulator) BuildWorkTraceRecords() []trace.WorkTraceRecord {
	return cs.buildWorkTraceRecordsFrom(cs.gatherWorkByInstance())
}

func (cs *ClusterSimulator) buildWorkTraceRecordsFrom(byInst map[string]map[string]sim.ReqWork) []trace.WorkTraceRecord {
	claimed := make(map[string]map[string]bool) // instanceID → reqID → claimed by a parent
	mark := func(inst, id string) {
		if claimed[inst] == nil {
			claimed[inst] = map[string]bool{}
		}
		claimed[inst][id] = true
	}
	get := func(inst, id string) (sim.ReqWork, bool) {
		m := byInst[inst]
		if m == nil {
			return sim.ReqWork{}, false
		}
		w, ok := m[id]
		return w, ok
	}
	mk := func(id, slo string, ar, ap, o int64, chunks int, pfWork, decWork float64) trace.WorkTraceRecord {
		chf := 0.0
		if ar > 0 {
			chf = 1.0 - float64(ap)/float64(ar)
		}
		return trace.WorkTraceRecord{
			RequestID: id, SLOClass: slo, Ar: ar, ApRealized: ap, ORealized: o,
			PrefillChunks: chunks, CacheHitFrac: chf,
			RealizedPrefillWork: pfWork, RealizedDecodeWork: decWork,
			WpClosed:           cs.workCoeffs.Wp(int(ap), int(ar)),
			WdClosed:           cs.workCoeffs.Wd(int(ar), float64(o)),
			WpClosedNoCacheOld: cs.workCoeffs.CPf*float64(ap) + (cs.workCoeffs.CAttn/2.0)*float64(ap)*float64(ap),
		}
	}

	recs := make([]trace.WorkTraceRecord, 0)
	for pid, p := range cs.parentRequests {
		pf, _ := get(string(p.PrefillInstanceID), p.PrefillSubReqID)
		dec, _ := get(string(p.DecodeInstanceID), p.DecodeSubReqID)
		mark(string(p.PrefillInstanceID), p.PrefillSubReqID)
		mark(string(p.DecodeInstanceID), p.DecodeSubReqID)
		slo, ar := "", int64(0)
		if p.OriginalRequest != nil {
			slo = p.OriginalRequest.SLOClass
			ar = util.Len64(p.OriginalRequest.InputTokens)
		}
		recs = append(recs, mk(pid, slo, ar, pf.ApRealized, dec.ORealized, pf.PrefillChunks, pf.RealizedPrefillWork, dec.RealizedDecodeWork))
	}
	for inst, m := range byInst {
		for id, w := range m {
			if claimed[inst][id] {
				continue
			}
			recs = append(recs, mk(id, w.SLOClass, w.Ar, w.ApRealized, w.ORealized, w.PrefillChunks, w.RealizedPrefillWork, w.RealizedDecodeWork))
		}
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].RequestID < recs[j].RequestID })
	return recs
}
```

Ensure `sort`, `util`, and the `trace` package are imported in `cluster.go`.

- [ ] **Step 4: Run builder test**

Run: `go test ./sim/cluster/ -run TestBuildWorkTraceRecords -v`
Expected: PASS.

- [ ] **Step 5: Wire the `--edpp-work-trace` flag (run + replay)**

In `cmd/root.go`, near `pdOutcomeTracePath` (~line 127):

```go
	edppWorkTracePath string // Stage B: per-request realized-vs-closed work CSV
```

Register it alongside `--pd-outcome-trace` (`registerSimConfigFlags`, ~line 1178):

```go
	cmd.Flags().StringVar(&edppWorkTracePath, "edpp-work-trace", "", "Write per-request realized-vs-closed work model CSV (Stage B validation). Requires --pd-decider edpp (uses its coeffs).")
```

Where the cluster is constructed (run path), after construction, when the flag is set and EDPP coeffs are available, enable it (the EDPP coeffs are on the deployment config as `DeploymentConfig.EDPPCoeffs`, loaded by `--edpp-coeffs`):

```go
	if edppWorkTracePath != "" {
		clusterSim.EnableWorkTrace(deploymentCfg.EDPPCoeffs) // confirm the coeffs field/var name in situ
	}
```

At run end (after the `pdOutcomeTracePath` block), mirror the shared-helper pattern used for `--pd-outcome-trace`:

```go
	if edppWorkTracePath != "" {
		recs := clusterSim.BuildWorkTraceRecords()
		if len(recs) == 0 {
			logrus.Warnf("--edpp-work-trace: no work records (need --pd-decider edpp)")
		} else if f, err := os.Create(edppWorkTracePath); err != nil {
			logrus.Errorf("--edpp-work-trace: could not create %q: %v", edppWorkTracePath, err)
		} else {
			werr := trace.WriteWorkTraceCSV(f, recs)
			cerr := f.Close()
			if werr != nil {
				logrus.Errorf("--edpp-work-trace: write failed: %v", werr)
			} else if cerr != nil {
				logrus.Errorf("--edpp-work-trace: close failed: %v", cerr)
			} else {
				logrus.Infof("Wrote %d work-trace records to %s", len(recs), edppWorkTracePath)
			}
		}
	}
```

In `cmd/replay.go`, mirror the enable (after cluster construction) and the write block (near the `pdOutcomeTracePath` replay block), using replay's cluster/coeffs variable names. INV-13 requires identical behavior.

- [ ] **Step 6: Build, smoke test, run full suite**

```bash
go build -o blis main.go
./blis run --help | grep edpp-work-trace   # flag registered
go test ./... 2>&1 | tail -20
gofmt -l sim/ cmd/ && go vet ./sim/... ./cmd/...
```
Expected: flag present; tests PASS; gofmt/vet clean. (A short PD run producing a non-empty CSV header is a good extra smoke check — adapt a `--pd-decider edpp --edpp-coeffs ...` invocation.)

- [ ] **Step 7: Commit**

```bash
git add sim/cluster/cluster.go cmd/root.go cmd/replay.go sim/cluster/work_trace_test.go
git commit -m "feat(edpp): --edpp-work-trace flag, cluster work-record builder (run+replay)"
```

---

### Task 5: Work-model validation analysis script

**Files:**
- Create: `campaigns/edpp-study/analyze/work_model_validation.py`

**Interfaces:**
- Consumes: the `--edpp-work-trace` CSV (Task 2 header).
- Produces: JSON bias report to stdout / `--out`; optional PNG behind `--plots`.

- [ ] **Step 1: Write the script**

Create `campaigns/edpp-study/analyze/work_model_validation.py`:

```python
#!/usr/bin/env python3
"""Stage B: validate the corrected EDPP work model against per-request realized work.

Reads an --edpp-work-trace CSV and reports relative error of realized vs closed-form
prefill/decode work, and the correction effect vs the old shipped form (basis change
+ cache effect). Expected: ~0 error for single-chunk requests; chunked-prefill residual
equals the documented C_attn·(a_p²−Σs_r²)/2 term.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def rel_err(realized: pd.Series, closed: pd.Series) -> dict:
    d = pd.DataFrame({"r": realized, "c": closed}).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["c"] != 0]
    if d.empty:
        return {"n": 0}
    e = (d["r"] - d["c"]) / d["c"]
    q = lambda p: float(np.percentile(e.abs(), p))
    return {
        "n": int(len(d)),
        "mean_rel_err": float(e.mean()),
        "median_rel_err": float(e.median()),
        "max_abs_rel_err": float(e.abs().max()),
        "abs_rel_err_p90": q(90), "abs_rel_err_p99": q(99),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="--edpp-work-trace CSV")
    ap.add_argument("--out", default="")
    ap.add_argument("--plots", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.work)
    single = df[df["prefill_chunks"] == 1]
    chunked = df[df["prefill_chunks"] > 1]

    report = {
        "total_requests": int(len(df)),
        "single_chunk_prefill": int(len(single)),
        "chunked_prefill": int(len(chunked)),
        "prefill_work_single_chunk": rel_err(single["realized_prefill_work"], single["wp_closed"]),
        "prefill_work_chunked": rel_err(chunked["realized_prefill_work"], chunked["wp_closed"]),
        "decode_work": rel_err(df["realized_decode_work"], df["wd_closed"]),
        "correction_effect": {
            # basis change visible even at cache_hit_frac≈0; cache effect grows with cache_hit_frac.
            "median_wp_over_old": float(
                (df["wp_closed"] / df["wp_closed_nocache_old"].where(df["wp_closed_nocache_old"] != 0, np.nan))
                .median(skipna=True)
            ),
            "median_cache_hit_frac": float(df["cache_hit_frac"].median()),
        },
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].scatter(df["wp_closed"], df["realized_prefill_work"], s=6, alpha=0.4)
        ax[0].set_title("prefill: realized vs closed"); ax[0].set_xlabel("wp_closed"); ax[0].set_ylabel("realized")
        ax[1].scatter(df["wd_closed"], df["realized_decode_work"], s=6, alpha=0.4)
        ax[1].set_title("decode: realized vs closed"); ax[1].set_xlabel("wd_closed"); ax[1].set_ylabel("realized")
        for a in ax:
            lim = max(a.get_xlim()[1], a.get_ylim()[1]); a.plot([0, lim], [0, lim], "k--", lw=1)
        fig.tight_layout(); fig.savefig(args.plots, dpi=120)
        print(f"wrote plot to {args.plots}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Syntax-check + run on a synthetic CSV**

```bash
python3 -m py_compile campaigns/edpp-study/analyze/work_model_validation.py
printf 'request_id,slo_class,a_r,a_p_realized,o_r_realized,prefill_chunks,cache_hit_frac,realized_prefill_work,realized_decode_work,wp_closed,wd_closed,wp_closed_nocache_old\nr1,batch,1000,1000,50,1,0,1506000,313725,1506000,313725,506000\nr2,batch,1000,200,50,1,0.8,220000,313725,220000,313725,206000\n' > /tmp/wt.csv
python3 campaigns/edpp-study/analyze/work_model_validation.py --work /tmp/wt.csv
```
Expected: JSON with `prefill_work_single_chunk.max_abs_rel_err` = 0, `decode_work.max_abs_rel_err` = 0, `correction_effect.median_wp_over_old` ≈ (1506000/506000 and 220000/206000 → median ≈ 1.53).

- [ ] **Step 3: Commit**

```bash
git add campaigns/edpp-study/analyze/work_model_validation.py
git commit -m "feat(edpp): add work_model_validation.py for Stage B"
```

---

### Task 6: Reproduction script, validation runs, and FINDINGS

**Files:**
- Create: `campaigns/edpp-study/repro_stage_b.sh`
- Modify: `campaigns/edpp-study/FINDINGS.md`

**Interfaces:** none (verification + docs).

- [ ] **Step 1: Write the repro script**

Create `campaigns/edpp-study/repro_stage_b.sh` (mirror `repro_stage_a.sh`; run BOTH synth and rag). It must: build `blis` if needed; for each of `synth_rate2.0.yaml` and `rag_rate2.0.yaml`, bake at `--num-instances 4`, replay at 2P2D with `--pd-decider edpp --edpp-coeffs scripts/calibration/coeffs-llama70b-h100-tp4.json` + the per-workload SLO/tau flags (synth: `--edpp-tau-ttft 2s --edpp-tau-itl 150ms --slo-ttft "batch=2s" --slo-itl "batch=150ms"`; rag: `--slo-ttft "standard=500ms,batch=5s" --slo-itl "standard=150ms,batch=200ms" --edpp-tau-ttft-classes "standard=500ms,batch=5s" --edpp-tau-itl-classes "standard=150ms,batch=200ms"`) plus `--edpp-work-trace out/stage_b/<wl>_work.csv`; then run `work_model_validation.py` writing `out/stage_b/<wl>_bias.json`. Outputs under `campaigns/edpp-study/out/stage_b/` (gitignored).

- [ ] **Step 2: Run it and capture the numbers**

Run: `bash campaigns/edpp-study/repro_stage_b.sh 2>/tmp/repro_b.log; cat campaigns/edpp-study/out/stage_b/*_bias.json`
Expected: for BOTH workloads, `prefill_work_single_chunk.max_abs_rel_err` and `decode_work.max_abs_rel_err` are within float tolerance (< 1e-6); chunked-prefill residual (if any) is the documented chunking term; synth `correction_effect.median_cache_hit_frac` ≈ 0 with `median_wp_over_old` ≈ 3 (attention basis change), rag shows higher cache-hit and additional cache effect. If prefill error is NOT ~0, the accumulator does not mirror the active latency model — stop and reconcile.

- [ ] **Step 3: Record the synth decision shift (expected, not a regression)**

Compare disagg fraction / decision trace on synth@2P2D against the pre-Task-1 commit (checkout the parent of the Task 1 commit into a scratch build, or reuse the Stage A anchor's decision trace). Note the pre/post disagg-fraction delta — this is the corrected `W_p` changing behavior, expected.

- [ ] **Step 4: Write the FINDINGS "Stage B" section**

Append to `campaigns/edpp-study/FINDINGS.md` a "Stage B — corrected work model + validation" section with: the corrected formulas, the exactness result (realized-vs-closed ~0 for both works, both workloads), the basis-change (~3× attention) and RAG cache-effect numbers, the expected synth decision shift, the reproduction command + checkpoint numbers, and the **§7 fidelity note verbatim** (trained-physics over-counts causal attention vs roofline; `W_p` matches the active model deliberately; a latency-model fix + `C_attn` refit is a separate deferred task).

- [ ] **Step 5: Commit**

```bash
git add campaigns/edpp-study/repro_stage_b.sh campaigns/edpp-study/FINDINGS.md
git commit -m "docs(edpp): Stage B repro script + work-model validation findings"
```

---

## Self-Review Notes

- **Spec coverage:** §2 corrections → Task 1. §3 accumulator (mirrors active model, gated) → Task 3; CSV type/writer → Task 2; flag/builder/wiring/correlation → Task 4. §4 analysis → Task 5. §5 tests → Tasks 1/3 (units), 4 (INV-13/INV-6 to verify in Step 6), 6 (validation runs + expected decision shift). §6 deliverables all mapped. §7 fidelity note → Task 6 Step 4 (FINDINGS) + the §3.6 reconciliation is called out as a later design-branch pass (not code here).
- **Confirm-in-situ points (flagged, not placeholders):** the instance-slice field and `inst.sim`/`inst.id` accessor names in `cluster.go` (same ones used at the `OnAdmit` site ~line 601); the EDPP coeffs field name on the deployment/replay config (`DeploymentConfig.EDPPCoeffs`); the exact `pdOutcomeTracePath` write-block location to mirror. Each names the anchor to copy from.
- **Type consistency:** `Wp(ap, ar int)`, `Wd(ar int, o float64)`, `ReqWork` fields, `WorkTraceRecord` fields, `SetWorkTrace`/`WorkAccumulators`/`EnableWorkTrace`/`BuildWorkTraceRecords` used identically across Tasks 1–5. `wp_closed_nocache_old` = `C_pf·a_p + (C_attn/2)·a_p²` consistently in Task 2 (doc), Task 4 (builder), Task 5 (analyzer).
- **INV-13:** Task 4 mirrors the run write block into replay; verify identical work-trace CSV in Task 6 (both bake+replay). **No byte-identical synth regression** — decision change is expected (Task 6 Step 3).
