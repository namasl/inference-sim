package trace

import "testing"

func TestRecordEDPPDecision_Appends(t *testing.T) {
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	if st.EDPPDecisions == nil {
		t.Fatal("EDPPDecisions slice should be initialized by NewSimulationTrace")
	}
	st.RecordEDPPDecision(EDPPDecisionRecord{
		RequestID: "r1", Clock: 42, Class: "batch",
		LHS: 0.5, RHS: 0.2, Disaggregate: true,
		TransferTerm: 0.05, TTFTTerm: 0.1, ITLTerm: 0.05,
	})
	st.RecordEDPPDecision(EDPPDecisionRecord{RequestID: "r2", Clock: 43, SkipReason: "empty-prompt"})

	if len(st.EDPPDecisions) != 2 {
		t.Fatalf("len(EDPPDecisions) = %d, want 2", len(st.EDPPDecisions))
	}
	if got := st.EDPPDecisions[0]; got.RequestID != "r1" || got.Clock != 42 || !got.Disaggregate {
		t.Errorf("record[0] = %+v, want RequestID=r1 Clock=42 Disaggregate=true", got)
	}
	if got := st.EDPPDecisions[1]; got.SkipReason != "empty-prompt" {
		t.Errorf("record[1].SkipReason = %q, want empty-prompt", got.SkipReason)
	}
}
