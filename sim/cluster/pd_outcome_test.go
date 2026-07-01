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
		parentRequests:            map[string]*ParentRequest{},
		pendingPrefillCompletions: map[string]string{},
		pendingDecodeCompletions:  map[string]string{},
		localAdmitTimes:           map[string]int64{},
		recordPDOutcomes:          true,
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
		parentRequests:            map[string]*ParentRequest{},
		pendingPrefillCompletions: map[string]string{},
		pendingDecodeCompletions:  map[string]string{},
		localAdmitTimes:           map[string]int64{},
		recordPDOutcomes:          false,
	}
	cs.recordAdmissionTime(&sim.Request{ID: "r2"}, 40)
	if len(cs.localAdmitTimes) != 0 {
		t.Fatalf("localAdmitTimes populated while disabled: %v", cs.localAdmitTimes)
	}
}
