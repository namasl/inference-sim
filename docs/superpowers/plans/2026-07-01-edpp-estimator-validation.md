# EDPP Estimator Validation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument BLIS to emit a per-request `--pd-outcome-trace` CSV pairing realized admission delay / TTFT / ITL / E2E with each request's identity, so the shipped EDPP forward TTFT estimators can be validated against reality (Stage A, measurement-only).

**Architecture:** Capture the two currently-missing schedule instants (`PrefillScheduleTime`, `DecodeScheduleTime`) on `ParentRequest` plus a local-admit map, all populated through the existing `OnAdmit` correlation. At run end a cluster builder walks parents + local map + aggregated metrics into `[]trace.PDOutcomeRecord`, written by a new CSV writer gated on `--pd-outcome-trace` (run + replay, INV-13 parity). A new Python script joins the outcome CSV against the existing `--edpp-decision-trace` on `request_id` and reports estimator bias. No decider or routing behavior changes.

**Tech Stack:** Go 1.22+ (`sim/`, `sim/cluster/`, `sim/trace/`, `cmd/`), Python 3 + pandas (`campaigns/edpp-study/analyze/`).

## Global Constraints

- No changes to `sim/edpp.go`, `sim/edpp_coeffs.go`, or any routing/decider behavior — measurement only.
- INV-6 determinism: same seed ⇒ byte-identical stdout and byte-identical outcome CSV. Emit rows sorted by `request_id`.
- INV-13 run/replay parity: a trace exported via `--trace-output` then replayed with `--pd-outcome-trace` must produce an identical outcome CSV.
- INV-5 causality: `enqueue ≤ schedule ≤ completion`; every emitted `t_adm ≥ 0`.
- All timestamps in microseconds, absolute. Zero = phase not reached (existing `ParentRequest` convention).
- The outcome trace and its per-request bookkeeping (local-admit map) must impose **zero cost** when `--pd-outcome-trace` is unset (gate all new work on the config flag).
- `go test ./...` and `golangci-lint run ./...` must pass after every task.

---

### Task 1: Capture per-sub-request and local admission times

**Files:**
- Modify: `sim/cluster/parent_request.go` (add two fields)
- Modify: `sim/cluster/cluster.go` (local-admit map field ~line 56 region; `recordAdmissionTime` method near `feedAdmission` ~line 1288; call sites in the two `OnAdmit` closures at ~601 and ~1129; init map where `parentRequests` is initialized ~267)
- Test: `sim/cluster/pd_outcome_test.go` (new)

**Interfaces:**
- Consumes: existing `admissionConservationKey(req) (key string, prefillSide bool, known bool)`, `cs.parentRequests map[string]*ParentRequest`, `req.IsDecodeSubRequest`, `cs.pendingPrefillCompletions`.
- Produces: `ParentRequest.PrefillScheduleTime int64`, `ParentRequest.DecodeScheduleTime int64`; `cs.localAdmitTimes map[string]int64`; method `func (cs *ClusterSimulator) recordAdmissionTime(req *sim.Request, tick int64)`; config gate `cs.recordPDOutcomes bool` (wired in Task 4 — for now default false, set directly in tests).

- [ ] **Step 1: Write the failing test**

Create `sim/cluster/pd_outcome_test.go`:

```go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// recordAdmissionTime must set the correct schedule field on the correct parent
// for disaggregated sub-requests, and record local requests in the local-admit map,
// keeping the FIRST admission time (idempotent under preemption re-admit).
func TestRecordAdmissionTime_RoutesToCorrectSlot(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:           map[string]*ParentRequest{},
		pendingPrefillCompletions: map[string]string{},
		pendingDecodeCompletions:  map[string]string{},
		localAdmitTimes:          map[string]int64{},
		recordPDOutcomes:         true,
	}
	parent := &ParentRequest{ID: "r1", PrefillSubReqID: "r1_prefill", DecodeSubReqID: "r1_decode"}
	cs.parentRequests["r1"] = parent
	cs.pendingPrefillCompletions["r1_prefill"] = "r1"
	cs.pendingDecodeCompletions["r1_decode"] = "r1"

	// Prefill sub-request admitted at t=100.
	cs.recordAdmissionTime(&sim.Request{ID: "r1_prefill"}, 100)
	if parent.PrefillScheduleTime != 100 {
		t.Fatalf("PrefillScheduleTime = %d, want 100", parent.PrefillScheduleTime)
	}
	// Decode sub-request admitted at t=250.
	cs.recordAdmissionTime(&sim.Request{ID: "r1_decode", IsDecodeSubRequest: true}, 250)
	if parent.DecodeScheduleTime != 250 {
		t.Fatalf("DecodeScheduleTime = %d, want 250", parent.DecodeScheduleTime)
	}
	// Local (non-disagg) request admitted at t=40, re-admitted at t=90 — keep 40.
	cs.recordAdmissionTime(&sim.Request{ID: "r2"}, 40)
	cs.recordAdmissionTime(&sim.Request{ID: "r2"}, 90)
	if got := cs.localAdmitTimes["r2"]; got != 40 {
		t.Fatalf("localAdmitTimes[r2] = %d, want 40 (first admission)", got)
	}
}

// When the flag is off, no local-admit bookkeeping happens (zero-cost gate).
func TestRecordAdmissionTime_DisabledIsNoop(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:           map[string]*ParentRequest{},
		pendingPrefillCompletions: map[string]string{},
		pendingDecodeCompletions:  map[string]string{},
		localAdmitTimes:          map[string]int64{},
		recordPDOutcomes:         false,
	}
	cs.recordAdmissionTime(&sim.Request{ID: "r2"}, 40)
	if len(cs.localAdmitTimes) != 0 {
		t.Fatalf("localAdmitTimes populated while disabled: %v", cs.localAdmitTimes)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/ -run TestRecordAdmissionTime -v`
Expected: FAIL — `parent.PrefillScheduleTime` / `cs.localAdmitTimes` / `cs.recordPDOutcomes` / `recordAdmissionTime` undefined (compile error).

- [ ] **Step 3: Add the ParentRequest fields**

In `sim/cluster/parent_request.go`, inside the `Phase timestamps` block (after `DecodeEnqueueTime int64`, before the `CompletionTime` doc comment), add:

```go
	// Schedule instants (microseconds): when each sub-request first entered the
	// running batch (waiting → running). Zero = not yet scheduled. Populated via
	// recordAdmissionTime from the OnAdmit hook, only when --pd-outcome-trace is set.
	// Used to compute realized admission delay T_adm = schedule − enqueue (§3.8).
	PrefillScheduleTime int64
	DecodeScheduleTime  int64
```

- [ ] **Step 4: Add the map field, config gate, init, method, and call sites**

In `sim/cluster/cluster.go`, in the `ClusterSimulator` struct near `parentRequests` (~line 56), add:

```go
	localAdmitTimes  map[string]int64 // request ID → first local-admission tick (non-disagg); populated only when recordPDOutcomes
	recordPDOutcomes bool             // gate: capture per-request admission times for --pd-outcome-trace
```

Where `parentRequests` is initialized (~line 267), add:

```go
	cs.localAdmitTimes = make(map[string]int64)
```

Add the method next to `feedAdmission` (~line 1299):

```go
// recordAdmissionTime captures the first admission instant of a request for the
// --pd-outcome-trace estimator-validation harness (Stage A). No-op unless
// recordPDOutcomes is set. A prefill sub-request sets its parent's
// PrefillScheduleTime; a decode sub-request sets DecodeScheduleTime; a normal
// (non-disaggregated) request is recorded in localAdmitTimes. OnAdmit can fire
// twice under preemption re-admit, so the first (earliest) time is kept.
func (cs *ClusterSimulator) recordAdmissionTime(req *sim.Request, tick int64) {
	if !cs.recordPDOutcomes {
		return
	}
	if req.IsDecodeSubRequest {
		if pid, ok := cs.pendingDecodeCompletions[req.ID]; ok {
			if p := cs.parentRequests[pid]; p != nil && p.DecodeScheduleTime == 0 {
				p.DecodeScheduleTime = tick
			}
		}
		return
	}
	if pid, ok := cs.pendingPrefillCompletions[req.ID]; ok {
		if p := cs.parentRequests[pid]; p != nil && p.PrefillScheduleTime == 0 {
			p.PrefillScheduleTime = tick
		}
		return
	}
	if _, seen := cs.localAdmitTimes[req.ID]; !seen {
		cs.localAdmitTimes[req.ID] = tick
	}
}
```

In BOTH `OnAdmit` closures (~line 601 and ~line 1129), add the capture call alongside the existing `feedAdmission`:

```go
				inst.sim.OnAdmit = func(req *sim.Request, tick int64) {
					cs.feedAdmission(req)
					cs.recordAdmissionTime(req, tick)
				}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test ./sim/cluster/ -run TestRecordAdmissionTime -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add sim/cluster/parent_request.go sim/cluster/cluster.go sim/cluster/pd_outcome_test.go
git commit -m "feat(edpp): capture per-sub-request and local admission times for outcome trace"
```

---

### Task 2: PDOutcomeRecord type and CSV writer

**Files:**
- Modify: `sim/trace/record.go` (add `PDOutcomeRecord` struct)
- Create: `sim/trace/pd_outcome_csv.go`
- Test: `sim/trace/pd_outcome_csv_test.go` (new)

**Interfaces:**
- Produces: `trace.PDOutcomeRecord` (fields below); `func WritePDOutcomeCSV(w io.Writer, records []PDOutcomeRecord) error`.

- [ ] **Step 1: Write the failing test**

Create `sim/trace/pd_outcome_csv_test.go`:

```go
package trace

import (
	"bytes"
	"strings"
	"testing"
)

func TestWritePDOutcomeCSV_HeaderAndRow(t *testing.T) {
	records := []PDOutcomeRecord{{
		RequestID: "r1", SLOClass: "standard", InputTokens: 512, Disaggregated: true,
		PrefillInstance: "instance_0", DecodeInstance: "instance_2",
		PrefillEnqueue: 100, PrefillSchedule: 140, PrefillTAdm: 40,
		DecodeEnqueue: 900, DecodeSchedule: 1200, DecodeTAdm: 300,
		LocalEnqueue: 0, LocalSchedule: 0, LocalTAdm: 0,
		RealizedTTFT: 1500, RealizedMeanITL: 30, RealizedE2E: 42000, Completed: true,
	}}
	var buf bytes.Buffer
	if err := WritePDOutcomeCSV(&buf, records); err != nil {
		t.Fatalf("WritePDOutcomeCSV: %v", err)
	}
	out := buf.String()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 2 {
		t.Fatalf("want header + 1 row, got %d lines:\n%s", len(lines), out)
	}
	if !strings.HasPrefix(lines[0], "request_id,slo_class,input_tokens,disaggregated,") {
		t.Fatalf("unexpected header: %s", lines[0])
	}
	if !strings.Contains(lines[1], "r1,standard,512,true,instance_0,instance_2,") {
		t.Fatalf("unexpected row: %s", lines[1])
	}
	if !strings.HasSuffix(lines[1], ",true") { // completed
		t.Fatalf("row should end with completed=true: %s", lines[1])
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/trace/ -run TestWritePDOutcomeCSV -v`
Expected: FAIL — `PDOutcomeRecord` / `WritePDOutcomeCSV` undefined.

- [ ] **Step 3: Add the record struct**

In `sim/trace/record.go`, add:

```go
// PDOutcomeRecord is one request's realized outcome for EDPP estimator validation
// (Stage A). Joined against EDPPDecisionRecord on RequestID. Times are microseconds,
// absolute; zero means the phase was not reached. Emitted only under --pd-outcome-trace.
type PDOutcomeRecord struct {
	RequestID       string
	SLOClass        string
	InputTokens     int
	Disaggregated   bool
	PrefillInstance string
	DecodeInstance  string

	PrefillEnqueue  int64
	PrefillSchedule int64
	PrefillTAdm     int64
	DecodeEnqueue   int64
	DecodeSchedule  int64
	DecodeTAdm      int64
	LocalEnqueue    int64
	LocalSchedule   int64
	LocalTAdm       int64

	RealizedTTFT    float64
	RealizedMeanITL float64
	RealizedE2E     float64
	Completed       bool
}
```

- [ ] **Step 4: Add the CSV writer**

Create `sim/trace/pd_outcome_csv.go`:

```go
package trace

import (
	"encoding/csv"
	"io"
	"strconv"
)

// pdOutcomeCSVHeader lists the per-request outcome columns in write order.
var pdOutcomeCSVHeader = []string{
	"request_id", "slo_class", "input_tokens", "disaggregated",
	"prefill_instance", "decode_instance",
	"prefill_enqueue", "prefill_schedule", "prefill_t_adm",
	"decode_enqueue", "decode_schedule", "decode_t_adm",
	"local_enqueue", "local_schedule", "local_t_adm",
	"realized_ttft", "realized_mean_itl", "realized_e2e", "completed",
}

// WritePDOutcomeCSV writes realized per-request outcome records to w as CSV
// (header + one row per record). Callers pass records pre-sorted by request_id
// for deterministic output (INV-6). Consumed by --pd-outcome-trace / the
// estimator_validation.py analysis.
func WritePDOutcomeCSV(w io.Writer, records []PDOutcomeRecord) error {
	cw := csv.NewWriter(w)
	if err := cw.Write(pdOutcomeCSVHeader); err != nil {
		return err
	}
	i := func(v int64) string { return strconv.FormatInt(v, 10) }
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	for _, r := range records {
		row := []string{
			r.RequestID, r.SLOClass, strconv.Itoa(r.InputTokens), strconv.FormatBool(r.Disaggregated),
			r.PrefillInstance, r.DecodeInstance,
			i(r.PrefillEnqueue), i(r.PrefillSchedule), i(r.PrefillTAdm),
			i(r.DecodeEnqueue), i(r.DecodeSchedule), i(r.DecodeTAdm),
			i(r.LocalEnqueue), i(r.LocalSchedule), i(r.LocalTAdm),
			f(r.RealizedTTFT), f(r.RealizedMeanITL), f(r.RealizedE2E), strconv.FormatBool(r.Completed),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test ./sim/trace/ -run TestWritePDOutcomeCSV -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add sim/trace/record.go sim/trace/pd_outcome_csv.go sim/trace/pd_outcome_csv_test.go
git commit -m "feat(edpp): add PDOutcomeRecord and CSV writer for outcome trace"
```

---

### Task 3: Build outcome records from parents + local map + metrics

**Files:**
- Modify: `sim/cluster/cluster.go` (add `BuildPDOutcomeRecords` method)
- Test: `sim/cluster/pd_outcome_test.go` (extend from Task 1)

**Interfaces:**
- Consumes: `cs.parentRequests`, `cs.localAdmitTimes`, `ParentRequest` timestamp fields (Task 1), realized per-request maps from `*sim.Metrics` (`RequestTTFTs`, `RequestITLs`, `RequestE2Es`, all keyed by original request ID). The aggregated metrics are passed in by the caller (Task 4) as `*sim.Metrics`.
- Produces: `func (cs *ClusterSimulator) BuildPDOutcomeRecords(m *sim.Metrics) []trace.PDOutcomeRecord` — records sorted by `RequestID`.

**Note on realized-metric source:** `sim.Metrics.Request{TTFTs,ITLs,E2Es}` are keyed by the **original request ID** (the parent ID for PD requests). `RequestE2Es[id] > 0` is the established "completed" test (see `sim/metrics.go:140`). Use the same convention here.

- [ ] **Step 1: Write the failing test**

Append to `sim/cluster/pd_outcome_test.go`:

```go
import (
	"sort"
	// keep existing imports; add "sort" and the sim metrics package if not present
)

func TestBuildPDOutcomeRecords_DisaggAndLocal(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:  map[string]*ParentRequest{},
		localAdmitTimes: map[string]int64{},
		recordPDOutcomes: true,
	}
	// Disaggregated request r1.
	cs.parentRequests["r1"] = &ParentRequest{
		ID: "r1", OriginalRequest: &sim.Request{ID: "r1", InputTokens: make([]int, 512), SLOClass: "standard"},
		Disaggregated: true, // see note below
		PrefillInstanceID: "instance_0", DecodeInstanceID: "instance_2",
		PrefillEnqueueTime: 100, PrefillScheduleTime: 140,
		DecodeEnqueueTime: 900, DecodeScheduleTime: 1200,
		CompletionTime: 43000,
	}
	// Local request r2 (no parent).
	cs.localAdmitTimes["r2"] = 40

	m := sim.NewMetrics()
	m.RequestTTFTs["r1"] = 1500
	m.RequestITLs["r1"] = 30
	m.RequestE2Es["r1"] = 42000
	m.RequestTTFTs["r2"] = 200
	m.RequestITLs["r2"] = 25
	m.RequestE2Es["r2"] = 8000

	recs := cs.BuildPDOutcomeRecords(m)
	if len(recs) != 2 {
		t.Fatalf("want 2 records, got %d", len(recs))
	}
	// Sorted by request_id: r1 then r2.
	if recs[0].RequestID != "r1" || recs[1].RequestID != "r2" {
		t.Fatalf("records not sorted by request_id: %s, %s", recs[0].RequestID, recs[1].RequestID)
	}
	r1 := recs[0]
	if !r1.Disaggregated || r1.PrefillTAdm != 40 || r1.DecodeTAdm != 300 {
		t.Fatalf("r1 disagg fields wrong: disagg=%v prefillTAdm=%d decodeTAdm=%d", r1.Disaggregated, r1.PrefillTAdm, r1.DecodeTAdm)
	}
	if r1.InputTokens != 512 || r1.RealizedE2E != 42000 || !r1.Completed {
		t.Fatalf("r1 metrics wrong: in=%d e2e=%v done=%v", r1.InputTokens, r1.RealizedE2E, r1.Completed)
	}
	r2 := recs[1]
	if r2.Disaggregated || r2.LocalTAdm != 0 || r2.LocalSchedule != 40 {
		t.Fatalf("r2 local fields wrong: disagg=%v localTAdm=%d localSchedule=%d", r2.Disaggregated, r2.LocalTAdm, r2.LocalSchedule)
	}
}
```

**Sub-note:** the test references `ParentRequest.Disaggregated`. `ParentRequest` does not currently carry that flag; derive "disaggregated" in the builder from `DecodeInstanceID != "" && DecodeInstanceID != PrefillInstanceID` instead, and **drop the `Disaggregated:` literal from the test**, asserting `r1.Disaggregated` is computed true. (Adjust the test literal accordingly before running.)

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/ -run TestBuildPDOutcomeRecords -v`
Expected: FAIL — `BuildPDOutcomeRecords` undefined.

- [ ] **Step 3: Implement the builder**

Add to `sim/cluster/cluster.go` (ensure `sort` and `github.com/inference-sim/inference-sim/sim/trace` are imported):

```go
// BuildPDOutcomeRecords assembles one realized-outcome record per request for the
// --pd-outcome-trace estimator-validation harness (Stage A). It walks disaggregated
// parents and locally-admitted requests, pairs each with its realized TTFT/ITL/E2E
// from m (keyed by original request ID), and returns records sorted by RequestID
// (INV-6). A t_adm is emitted only when both its enqueue and schedule instants are
// set; otherwise it stays zero. "disaggregated" means a distinct decode instance was
// used. Completion follows the metrics convention: RequestE2Es[id] > 0.
func (cs *ClusterSimulator) BuildPDOutcomeRecords(m *sim.Metrics) []trace.PDOutcomeRecord {
	recs := make([]trace.PDOutcomeRecord, 0, len(cs.parentRequests)+len(cs.localAdmitTimes))
	tadm := func(enq, sched int64) int64 {
		if enq > 0 && sched >= enq {
			return sched - enq
		}
		return 0
	}
	realized := func(id string) (ttft, itl, e2e float64, done bool) {
		e2e = m.RequestE2Es[id]
		return m.RequestTTFTs[id], m.RequestITLs[id], e2e, e2e > 0
	}

	for id, p := range cs.parentRequests {
		ttft, itl, e2e, done := realized(id)
		class, in := "", 0
		if p.OriginalRequest != nil {
			class = p.OriginalRequest.SLOClass
			in = len(p.OriginalRequest.InputTokens)
		}
		recs = append(recs, trace.PDOutcomeRecord{
			RequestID: id, SLOClass: class, InputTokens: in,
			Disaggregated:   p.DecodeInstanceID != "" && p.DecodeInstanceID != p.PrefillInstanceID,
			PrefillInstance: string(p.PrefillInstanceID), DecodeInstance: string(p.DecodeInstanceID),
			PrefillEnqueue:  p.PrefillEnqueueTime, PrefillSchedule: p.PrefillScheduleTime, PrefillTAdm: tadm(p.PrefillEnqueueTime, p.PrefillScheduleTime),
			DecodeEnqueue:   p.DecodeEnqueueTime, DecodeSchedule: p.DecodeScheduleTime, DecodeTAdm: tadm(p.DecodeEnqueueTime, p.DecodeScheduleTime),
			RealizedTTFT:    ttft, RealizedMeanITL: itl, RealizedE2E: e2e, Completed: done,
		})
	}

	for id, admit := range cs.localAdmitTimes {
		if _, isParent := cs.parentRequests[id]; isParent {
			continue // already emitted as a disagg/local-via-parent record
		}
		ttft, itl, e2e, done := realized(id)
		recs = append(recs, trace.PDOutcomeRecord{
			RequestID: id, Disaggregated: false,
			LocalEnqueue: 0, LocalSchedule: admit, LocalTAdm: 0, // enqueue instant not separately tracked for local; schedule captured
			RealizedTTFT: ttft, RealizedMeanITL: itl, RealizedE2E: e2e, Completed: done,
		})
	}

	sort.Slice(recs, func(i, j int) bool { return recs[i].RequestID < recs[j].RequestID })
	return recs
}
```

**Local `t_adm` note:** the non-disagg enqueue instant is not currently tracked as an absolute tick (only `GatewayEnqueueTime` under flow control). This task captures the local **schedule** time; `local_t_adm` is left 0 unless a local enqueue instant is available. If flow control is enabled, set `LocalEnqueue = req.GatewayEnqueueTime` and compute `LocalTAdm` — but that requires threading the request; defer to a follow-up if the anchor workload runs without flow control. Document this limitation in the record's producing comment.

- [ ] **Step 4: Add a causality invariant test**

Append to `sim/cluster/pd_outcome_test.go`:

```go
func TestBuildPDOutcomeRecords_CausalityAndNonNegativeTAdm(t *testing.T) {
	cs := &ClusterSimulator{
		parentRequests:  map[string]*ParentRequest{},
		localAdmitTimes: map[string]int64{},
		recordPDOutcomes: true,
	}
	cs.parentRequests["r1"] = &ParentRequest{
		ID: "r1", OriginalRequest: &sim.Request{ID: "r1"},
		PrefillInstanceID: "p0", DecodeInstanceID: "d0",
		PrefillEnqueueTime: 100, PrefillScheduleTime: 140,
		DecodeEnqueueTime: 900, DecodeScheduleTime: 1200,
	}
	m := sim.NewMetrics()
	m.RequestE2Es["r1"] = 42000
	for _, r := range cs.BuildPDOutcomeRecords(m) {
		if r.PrefillTAdm < 0 || r.DecodeTAdm < 0 || r.LocalTAdm < 0 {
			t.Fatalf("negative t_adm in %+v", r)
		}
		if r.PrefillSchedule != 0 && r.PrefillSchedule < r.PrefillEnqueue {
			t.Fatalf("schedule before enqueue (prefill) in %+v", r)
		}
		if r.DecodeSchedule != 0 && r.DecodeSchedule < r.DecodeEnqueue {
			t.Fatalf("schedule before enqueue (decode) in %+v", r)
		}
	}
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./sim/cluster/ -run TestBuildPDOutcomeRecords -v`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/pd_outcome_test.go
git commit -m "feat(edpp): build per-request outcome records from parents, local map, metrics"
```

---

### Task 4: Wire the --pd-outcome-trace flag on run and replay

**Files:**
- Modify: `cmd/root.go` (flag var ~line 126 region; flag registration ~line 1148 region; set `recordPDOutcomes` on the cluster where it is constructed; write CSV at run end near the `edppDecisionTracePath` block ~line 2112)
- Modify: `cmd/replay.go` (mirror write block near ~line 706)
- Test: `sim/cluster/pd_outcome_parity_test.go` (new — run/replay parity + determinism at the builder level) and a CLI smoke check

**Interfaces:**
- Consumes: `cs.BuildPDOutcomeRecords(m)` (Task 3), `trace.WritePDOutcomeCSV` (Task 2), the cluster's aggregated `*sim.Metrics` at run end, a cluster setter/field `recordPDOutcomes`.
- Produces: CLI flag `--pd-outcome-trace <path>` on `run` and `replay`; a way to set `cs.recordPDOutcomes = true` when the flag is non-empty (add exported setter `func (cs *ClusterSimulator) SetRecordPDOutcomes(v bool)` if the field is unexported and set from cmd).

- [ ] **Step 1: Write the failing parity/determinism test**

Create `sim/cluster/pd_outcome_parity_test.go`:

```go
package cluster

import (
	"bytes"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// BuildPDOutcomeRecords + WritePDOutcomeCSV must be deterministic: identical
// inputs yield byte-identical CSV (INV-6), independent of Go map iteration order.
func TestPDOutcome_DeterministicCSV(t *testing.T) {
	build := func() []byte {
		cs := &ClusterSimulator{parentRequests: map[string]*ParentRequest{}, localAdmitTimes: map[string]int64{}, recordPDOutcomes: true}
		for _, id := range []string{"r3", "r1", "r2"} {
			cs.localAdmitTimes[id] = int64(len(id) * 10)
		}
		m := sim.NewMetrics()
		for _, id := range []string{"r1", "r2", "r3"} {
			m.RequestE2Es[id] = 1000
		}
		var buf bytes.Buffer
		if err := trace.WritePDOutcomeCSV(&buf, cs.BuildPDOutcomeRecords(m)); err != nil {
			t.Fatalf("write: %v", err)
		}
		return buf.Bytes()
	}
	a, b := build(), build()
	if !bytes.Equal(a, b) {
		t.Fatalf("non-deterministic CSV:\n%s\n---\n%s", a, b)
	}
}
```

- [ ] **Step 2: Run test to verify it passes (builder already deterministic)**

Run: `go test ./sim/cluster/ -run TestPDOutcome_DeterministicCSV -v`
Expected: PASS (sort guarantees determinism). If it fails, the sort in Task 3 is missing — fix there.

- [ ] **Step 3: Add the flag variable and registration**

In `cmd/root.go`, near `edppDecisionTracePath` (~line 126):

```go
	pdOutcomeTracePath string // Path to write per-request realized-outcome CSV (Stage A estimator validation)
```

Near the `--edpp-decision-trace` registration (~line 1148):

```go
	cmd.Flags().StringVar(&pdOutcomeTracePath, "pd-outcome-trace", "", "Write per-request realized outcomes (T_adm/TTFT/ITL/E2E) to this CSV path for EDPP estimator validation. Requires PD/disaggregation.")
```

- [ ] **Step 4: Set the gate at cluster construction and write at run end**

Where the `ClusterSimulator` is constructed in `cmd/root.go`'s run path, after construction add:

```go
	if pdOutcomeTracePath != "" {
		clusterSim.SetRecordPDOutcomes(true)
	}
```

Add the setter to `sim/cluster/cluster.go`:

```go
// SetRecordPDOutcomes enables per-request admission-time capture for --pd-outcome-trace.
func (cs *ClusterSimulator) SetRecordPDOutcomes(v bool) { cs.recordPDOutcomes = v }
```

At run end in `cmd/root.go`, after the `edppDecisionTracePath` write block (~line 2126), add (use the same aggregated metrics variable the metrics output uses — confirm its name in situ, referred to here as `finalMetrics *sim.Metrics`):

```go
	if pdOutcomeTracePath != "" {
		recs := clusterSim.BuildPDOutcomeRecords(finalMetrics)
		if len(recs) == 0 {
			logrus.Warnf("--pd-outcome-trace: no outcome records (need PD/disaggregation enabled)")
		} else if f, err := os.Create(pdOutcomeTracePath); err != nil {
			logrus.Errorf("--pd-outcome-trace: could not create %q: %v", pdOutcomeTracePath, err)
		} else {
			werr := trace.WritePDOutcomeCSV(f, recs)
			cerr := f.Close()
			if werr != nil {
				logrus.Errorf("--pd-outcome-trace: write failed: %v", werr)
			} else if cerr != nil {
				logrus.Errorf("--pd-outcome-trace: close failed: %v", cerr)
			} else {
				logrus.Infof("Wrote %d PD outcome records to %s", len(recs), pdOutcomeTracePath)
			}
		}
	}
```

- [ ] **Step 5: Mirror the flag and write block in replay**

In `cmd/replay.go`, register the same flag (if `replay` uses a separate flag set — confirm; if flags are shared via `root.go`, skip re-registration) and add the identical write block near the existing `edppDecisionTracePath` block (~line 706), using replay's cluster + metrics variable names. This is what INV-13 parity requires.

- [ ] **Step 6: Build, smoke-test the CLI, and run the parity test**

```bash
go build -o blis main.go
# Smoke: a short PD run should produce a non-empty CSV with the header.
./blis run --model qwen/qwen3-14b --pd-decider edpp --edpp-coeffs scripts/calibration/coeffs-llama70b-h100-tp4.json \
  --pd-outcome-trace /tmp/pd_outcome.csv 2>/dev/null || true
head -1 /tmp/pd_outcome.csv
go test ./sim/cluster/ ./sim/trace/ -run 'PDOutcome|BuildPDOutcome|RecordAdmission|WritePDOutcome' -v
```
Expected: `/tmp/pd_outcome.csv` first line is the header; tests PASS. (Exact run flags for a PD topology mirror the study harness; adjust to a known-good PD invocation from `campaigns/edpp-study/`.)

- [ ] **Step 7: Full regression + lint**

Run: `go test ./... && golangci-lint run ./...`
Expected: all PASS / clean.

- [ ] **Step 8: Commit**

```bash
git add cmd/root.go cmd/replay.go sim/cluster/cluster.go sim/cluster/pd_outcome_parity_test.go
git commit -m "feat(edpp): add --pd-outcome-trace flag on run and replay"
```

---

### Task 5: Estimator-validation analysis script

**Files:**
- Create: `campaigns/edpp-study/analyze/estimator_validation.py`
- Test: manual run against the anchor CSVs (Task 6); no unit-test framework in the analyze/ dir.

**Interfaces:**
- Consumes: `--pd-outcome-trace` CSV (Task 2 header) and the existing `--edpp-decision-trace` CSV (`request_id`, `ttft_p`, `ttft_d`, `qp_raw`, `qd_raw`, `mu_p_nom`, `mu_d_nom`).
- Produces: a JSON bias report to stdout / `--out <path>`; optional PNG behind `--plots`.

- [ ] **Step 1: Write the script**

Create `campaigns/edpp-study/analyze/estimator_validation.py`:

```python
#!/usr/bin/env python3
"""Stage A: validate shipped EDPP forward TTFT estimators against realized outcomes.

Joins a --pd-outcome-trace CSV with an --edpp-decision-trace CSV on request_id and
reports estimator bias (predicted vs realized TTFT, and predicted admission component
vs realized T_adm), split by disaggregated x slo_class, over completed requests.

The shipped estimator does not log its admission component directly; we reconstruct it
from the decision trace as the queue-wait term: qp_raw/mu_p_nom (prefill side) and
qd_raw/mu_d_nom (decode side).
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def bias_stats(pred: pd.Series, real: pd.Series) -> dict:
    d = pd.DataFrame({"pred": pred, "real": real}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return {"n": 0}
    err = d["real"] - d["pred"]
    ratio = d["real"] / d["pred"].where(d["pred"] != 0, np.nan)
    q = lambda s, p: float(np.percentile(s, p)) if len(s) else None
    return {
        "n": int(len(d)),
        "mean_signed_error": float(err.mean()),
        "median_signed_error": float(err.median()),
        "median_ratio_real_over_pred": float(ratio.median(skipna=True)),
        "pred_p50": q(d["pred"], 50), "pred_p90": q(d["pred"], 90), "pred_p99": q(d["pred"], 99),
        "real_p50": q(d["real"], 50), "real_p90": q(d["real"], 90), "real_p99": q(d["real"], 99),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", required=True, help="--pd-outcome-trace CSV")
    ap.add_argument("--decision", required=True, help="--edpp-decision-trace CSV")
    ap.add_argument("--out", default="", help="write JSON report here (default stdout)")
    ap.add_argument("--plots", default="", help="write predicted-vs-realized scatter PNG here")
    args = ap.parse_args()

    out = pd.read_csv(args.outcome)
    dec = pd.read_csv(args.decision)
    df = out.merge(dec, on="request_id", how="inner", suffixes=("", "_dec"))

    total = len(out)
    completed = df[df["completed"] == True].copy()  # noqa: E712
    truncated = total - int((out["completed"] == True).sum())  # noqa: E712

    # Reconstruct admission components from the decision trace (µs).
    completed["pred_prefill_adm"] = completed["qp_raw"] / completed["mu_p_nom"].where(completed["mu_p_nom"] != 0, np.nan)
    completed["pred_decode_adm"] = completed["qd_raw"] / completed["mu_d_nom"].where(completed["mu_d_nom"] != 0, np.nan)

    report = {"total_requests": int(total), "completed": int(len(completed)), "truncated_or_dropped": int(truncated), "groups": {}}
    for (disagg, cls), g in completed.groupby(["disaggregated", "slo_class"]):
        key = f"disagg={disagg},class={cls}"
        # ttft_p validates disagg-path TTFT; ttft_d validates the local alternative.
        ttft_pred = g["ttft_p"] if disagg else g["ttft_d"]
        report["groups"][key] = {
            "ttft_pred_vs_realized": bias_stats(ttft_pred, g["realized_ttft"]),
            "prefill_adm_pred_vs_realized": bias_stats(g["pred_prefill_adm"], g["prefill_t_adm"]) if disagg else {"n": 0},
            "decode_adm_pred_vs_realized": bias_stats(g["pred_decode_adm"], g["decode_t_adm"]) if disagg else {"n": 0},
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
        fig, ax = plt.subplots(figsize=(6, 6))
        pred = np.where(completed["disaggregated"], completed["ttft_p"], completed["ttft_d"])
        ax.scatter(pred, completed["realized_ttft"], s=6, alpha=0.4)
        lim = max(float(np.nanmax(pred)), float(completed["realized_ttft"].max()))
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
        ax.set_xlabel("predicted TTFT (µs)"); ax.set_ylabel("realized TTFT (µs)")
        ax.set_title("EDPP predicted vs realized TTFT"); ax.legend()
        fig.tight_layout(); fig.savefig(args.plots, dpi=120)
        print(f"wrote plot to {args.plots}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Syntax-check the script**

Run: `python3 -m py_compile campaigns/edpp-study/analyze/estimator_validation.py`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
git add campaigns/edpp-study/analyze/estimator_validation.py
git commit -m "feat(edpp): add estimator_validation.py bias report for Stage A"
```

---

### Task 6: Sanity-anchor run and documentation

**Files:**
- Modify: `campaigns/edpp-study/FINDINGS.md` (record the Stage A anchor result) — tracked doc.
- Reference: `out/diag/REPRO.md` for the exact synth edpp@2P2D rate-2.0 invocation (on-disk, gitignored).

**Interfaces:** none (verification + docs).

- [ ] **Step 1: Reproduce the anchor operating point**

Read the exact command from `out/diag/REPRO.md` for synth edpp@2P2D rate 2.0, and add `--pd-outcome-trace` and `--edpp-decision-trace` to it. Example shape (substitute the real spec path and coeffs from REPRO.md):

```bash
./blis run --workload-spec campaigns/edpp-study/specs/synth/synth_2p2d.yaml \
  --pd-decider edpp --edpp-coeffs scripts/calibration/coeffs-llama70b-h100-tp4.json \
  --trace-level decisions \
  --edpp-decision-trace /tmp/anchor_decision.csv \
  --pd-outcome-trace /tmp/anchor_outcome.csv
```

- [ ] **Step 2: Run the analysis**

```bash
python3 campaigns/edpp-study/analyze/estimator_validation.py \
  --outcome /tmp/anchor_outcome.csv --decision /tmp/anchor_decision.csv \
  --out /tmp/anchor_bias.json
```
Expected: for the disagg/local groups, `ttft_pred_vs_realized` shows a large `median_ratio_real_over_pred` on the kept-local group — the archived ~174× under-prediction the older worklog documented. This is the harness sanity check: if the ratio is near 1 where the worklog says it should be ~100×+, the harness (not the estimator) is wrong.

- [ ] **Step 3: Record the finding**

Append a short "Stage A — estimator validation harness" subsection to `campaigns/edpp-study/FINDINGS.md`: the command, the group-wise bias table headline (kept-local under-prediction ratio, completed vs truncated counts), and the conclusion that the harness reproduces the archived figure. Note explicitly that this validates the *shipped* waiting-only estimator and is the baseline Stage C will improve on.

- [ ] **Step 4: Commit**

```bash
git add campaigns/edpp-study/FINDINGS.md
git commit -m "docs(edpp): record Stage A estimator-validation anchor result"
```

---

## Self-Review Notes

- **Spec coverage:** §2 CSV columns → Task 2 (+ `input_tokens` present). §3 instrumentation (schedule fields, OnAdmit, emit, flag) → Tasks 1, 3, 4. §4 analysis (JSON always, optional `--plots`, reconstructed admission component) → Task 5. §5 tests (unit, causality, INV-13, INV-6, anchor) → Tasks 1/3 (unit+causality), 4 (parity/determinism), 6 (anchor). §6 deliverables all mapped.
- **Known confirm-in-situ points (flagged inline, not placeholders):** exact aggregated-metrics variable name at the `cmd/root.go` and `cmd/replay.go` write sites; whether `replay` shares `root.go`'s flag set; whether `pendingDecodeCompletions`/`pendingPrefillCompletions` are the exact field names (verified present at `cluster.go:1276-1283`). The local `t_adm` limitation (no absolute local enqueue tick without flow control) is documented and deferred, not silently dropped.
- **Type consistency:** `PDOutcomeRecord` field names identical across Tasks 2, 3, 4. `BuildPDOutcomeRecords(*sim.Metrics) []trace.PDOutcomeRecord` and `SetRecordPDOutcomes(bool)` used consistently. `recordPDOutcomes` gate referenced in Tasks 1, 3, 4.
