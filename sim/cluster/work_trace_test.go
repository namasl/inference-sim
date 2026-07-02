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
		parentRequests:  map[string]*ParentRequest{},
		workCoeffs:      c,
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
