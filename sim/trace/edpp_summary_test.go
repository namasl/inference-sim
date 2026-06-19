package trace

import (
	"math"
	"testing"
)

func TestSummarize_EDPPDecomposition(t *testing.T) {
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	// 1: disaggregated (LHS > RHS).
	st.RecordEDPPDecision(EDPPDecisionRecord{RequestID: "1", LHS: 0.5, RHS: 0.2,
		BalanceTermD: 0.6, BalanceTermP: 0.1, TransferTerm: 0.05, TTFTTerm: 0.1, ITLTerm: 0.05, Disaggregate: true})
	// 2: kept local, transfer-dominated suppressor (only positive RHS term is transfer).
	st.RecordEDPPDecision(EDPPDecisionRecord{RequestID: "2", LHS: 0.03, RHS: 0.05,
		TransferTerm: 0.05, TTFTTerm: 0, ITLTerm: 0, Disaggregate: false})
	// 3: kept local, weak (non-positive) LHS.
	st.RecordEDPPDecision(EDPPDecisionRecord{RequestID: "3", LHS: -0.1, RHS: 0.0, Disaggregate: false})
	// 4: kept local, ITL-term-dominated suppressor.
	st.RecordEDPPDecision(EDPPDecisionRecord{RequestID: "4", LHS: 0.2, RHS: 0.3,
		TransferTerm: 0.05, TTFTTerm: 0.05, ITLTerm: 0.2, Disaggregate: false})
	// 5: skipped (early return).
	st.RecordEDPPDecision(EDPPDecisionRecord{RequestID: "5", SkipReason: "empty-prompt"})

	s := Summarize(st).EDPP

	if s.Total != 5 || s.Skipped != 1 || s.Evaluated != 4 {
		t.Errorf("counts: Total=%d Skipped=%d Evaluated=%d, want 5/1/4", s.Total, s.Skipped, s.Evaluated)
	}
	if s.DisaggregatedCount != 1 || s.KeptLocalCount != 3 {
		t.Errorf("Disaggregated=%d KeptLocal=%d, want 1/3", s.DisaggregatedCount, s.KeptLocalCount)
	}
	if s.SuppressorTransfer != 1 || s.SuppressorITL != 1 || s.SuppressorWeakLHS != 1 || s.SuppressorTTFT != 0 {
		t.Errorf("suppressors: transfer=%d ttft=%d itl=%d weakLHS=%d, want 1/0/1/1",
			s.SuppressorTransfer, s.SuppressorTTFT, s.SuppressorITL, s.SuppressorWeakLHS)
	}
	// Means over the 4 evaluated decisions.
	if math.Abs(s.MeanLHS-0.63/4) > 1e-9 {
		t.Errorf("MeanLHS = %v, want %v", s.MeanLHS, 0.63/4)
	}
	if math.Abs(s.MeanRHS-0.55/4) > 1e-9 {
		t.Errorf("MeanRHS = %v, want %v", s.MeanRHS, 0.55/4)
	}
	if math.Abs(s.MeanTransferTerm-0.15/4) > 1e-9 {
		t.Errorf("MeanTransferTerm = %v, want %v", s.MeanTransferTerm, 0.15/4)
	}
}

func TestSummarize_EDPP_EmptyIsZero(t *testing.T) {
	s := Summarize(NewSimulationTrace(TraceConfig{})).EDPP
	if s.Total != 0 || s.Evaluated != 0 || s.MeanLHS != 0 {
		t.Errorf("empty EDPP summary should be zero-valued, got %+v", s)
	}
}
