package trace

// TraceSummary aggregates statistics from a SimulationTrace.
type TraceSummary struct {
	TotalDecisions     int
	AdmittedCount      int
	RejectedCount      int
	MeanRegret         float64
	MaxRegret          float64
	UniqueTargets      int
	TargetDistribution map[string]int // instance ID → count of requests routed via standard routing only (not PD pool routing); use PrefillRoutings for prefill-pool counts

	// PD disaggregation summary (zero when disaggregation is not configured)
	DisaggregationCount  int     // number of disaggregation decisions recorded (true and false combined)
	DisaggregatedCount   int     // number of requests for which disaggregation was decided (Disaggregate=true); prefill routing happens in a subsequent event
	KVTransferCount      int     // number of KV transfers that completed with successful decode KV allocation
	MeanTransferDuration float64 // mean KV transfer duration in microseconds; zero when KVTransferCount == 0

	// EDPP decomposes the EDPP decider's rule terms (zero-valued when the EDPP
	// decider is not in use or tracing is off).
	EDPP EDPPDecisionSummary
}

// EDPPDecisionSummary aggregates the EDPP rule-term traces to explain why the decider
// disaggregated or kept work local. Means are taken over EVALUATED decisions only
// (SkipReason == ""). The suppressor tally classifies each KEPT-LOCAL decision by the
// factor that most explains RHS ≥ LHS: a non-positive LHS (no balancing benefit to begin
// with) is WeakLHS; otherwise the largest positive RHS component (transfer / TTFT / ITL).
type EDPPDecisionSummary struct {
	Total              int // all EDPP decision records
	Skipped            int // early-return records (SkipReason != "")
	Evaluated          int // records where the rule was evaluated
	DisaggregatedCount int // evaluated decisions with Disaggregate == true
	KeptLocalCount     int // evaluated decisions with Disaggregate == false

	MeanLHS          float64
	MeanRHS          float64
	MeanBalanceTermD float64
	MeanBalanceTermP float64
	MeanTransferTerm float64
	MeanTTFTTerm     float64
	MeanITLTerm      float64

	// Dominant-suppressor tally over kept-local decisions (sums to KeptLocalCount).
	SuppressorTransfer int
	SuppressorTTFT     int
	SuppressorITL      int
	SuppressorWeakLHS  int
}

// Summarize computes aggregate statistics from a SimulationTrace.
// Safe for nil or empty traces (returns zero-value fields).
func Summarize(st *SimulationTrace) *TraceSummary {
	summary := &TraceSummary{
		TargetDistribution: make(map[string]int),
	}
	if st == nil {
		return summary
	}

	summary.TotalDecisions = len(st.Admissions)
	for _, a := range st.Admissions {
		if a.Admitted {
			summary.AdmittedCount++
		} else {
			summary.RejectedCount++
		}
	}

	if len(st.Routings) > 0 {
		totalRegret := 0.0
		for _, r := range st.Routings {
			summary.TargetDistribution[r.ChosenInstance]++
			totalRegret += r.Regret
			if r.Regret > summary.MaxRegret {
				summary.MaxRegret = r.Regret
			}
		}
		summary.MeanRegret = totalRegret / float64(len(st.Routings))
	}

	// UniqueTargets counts distinct instances in TargetDistribution (standard routing only, not PD pool routing).
	summary.UniqueTargets = len(summary.TargetDistribution)

	// PD disaggregation summary
	summary.DisaggregationCount = len(st.Disaggregations)
	for _, d := range st.Disaggregations {
		if d.Disaggregate {
			summary.DisaggregatedCount++
		}
	}

	summary.KVTransferCount = len(st.KVTransfers)
	if len(st.KVTransfers) > 0 {
		// Accumulate in float64 to avoid int64 overflow for large simulations with many
		// long-duration transfers (int64 max ~9.22×10^18 µs; float64 exact up to ~9×10^15 µs).
		totalDuration := 0.0
		for _, kv := range st.KVTransfers {
			totalDuration += float64(kv.TransferDuration)
		}
		summary.MeanTransferDuration = totalDuration / float64(len(st.KVTransfers))
	}

	summary.EDPP = summarizeEDPP(st.EDPPDecisions)

	return summary
}

// summarizeEDPP aggregates EDPP rule-term records into an EDPPDecisionSummary.
func summarizeEDPP(records []EDPPDecisionRecord) EDPPDecisionSummary {
	var s EDPPDecisionSummary
	s.Total = len(records)
	var sumLHS, sumRHS, sumBD, sumBP, sumXfer, sumTTFT, sumITL float64
	for _, r := range records {
		if r.SkipReason != "" {
			s.Skipped++
			continue
		}
		s.Evaluated++
		sumLHS += r.LHS
		sumRHS += r.RHS
		sumBD += r.BalanceTermD
		sumBP += r.BalanceTermP
		sumXfer += r.TransferTerm
		sumTTFT += r.TTFTTerm
		sumITL += r.ITLTerm
		if r.Disaggregate {
			s.DisaggregatedCount++
			continue
		}
		s.KeptLocalCount++
		// Dominant suppressor for this kept-local decision.
		if r.LHS <= 0 {
			s.SuppressorWeakLHS++
			continue
		}
		switch {
		case r.TransferTerm >= r.TTFTTerm && r.TransferTerm >= r.ITLTerm:
			s.SuppressorTransfer++
		case r.TTFTTerm >= r.TransferTerm && r.TTFTTerm >= r.ITLTerm:
			s.SuppressorTTFT++
		default:
			s.SuppressorITL++
		}
	}
	if s.Evaluated > 0 {
		n := float64(s.Evaluated)
		s.MeanLHS = sumLHS / n
		s.MeanRHS = sumRHS / n
		s.MeanBalanceTermD = sumBD / n
		s.MeanBalanceTermP = sumBP / n
		s.MeanTransferTerm = sumXfer / n
		s.MeanTTFTTerm = sumTTFT / n
		s.MeanITLTerm = sumITL / n
	}
	return s
}
