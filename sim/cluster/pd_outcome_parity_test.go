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
