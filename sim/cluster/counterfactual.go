package cluster

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// copyScores returns a shallow copy of the scores map.
// Defensive copy: prevents trace data corruption if a future RoutingPolicy reuses its Scores map.
// Returns nil for nil input.
func copyScores(scores map[string]float64) map[string]float64 {
	if scores == nil {
		return nil
	}
	cp := make(map[string]float64, len(scores))
	for k, v := range scores {
		cp[k] = v
	}
	return cp
}

// buildRoutingTraceCandidates builds the full per-candidate record set for one
// routing target selection (--routing-decision-trace). Unlike computeCounterfactual
// it captures ALL candidates (not top-k), in deterministic instance-ID order
// (INV-6), and surfaces each scorer's per-instance contribution from
// decision.ScorerBreakdown. The composite score comes from decision.Scores (nil for
// non-scoring policies → composite 0, no per-scorer columns). Regret is the best
// composite minus the chosen composite (≥0; 0 when there are no scores).
func buildRoutingTraceCandidates(chosenID string, decision sim.RoutingDecision, snapshots []sim.RoutingSnapshot) ([]trace.RoutingTraceCandidate, float64) {
	sorted := make([]sim.RoutingSnapshot, len(snapshots))
	copy(sorted, snapshots)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	cands := make([]trace.RoutingTraceCandidate, 0, len(sorted))
	var bestScore, chosenScore float64
	haveBest := false
	for _, snap := range sorted {
		comp := decision.Scores[snap.ID] // zero value when Scores is nil
		if !haveBest || comp > bestScore {
			bestScore = comp
			haveBest = true
		}
		if snap.ID == chosenID {
			chosenScore = comp
		}

		var perScorer map[string]float64
		if decision.ScorerBreakdown != nil {
			perScorer = make(map[string]float64, len(decision.ScorerBreakdown))
			for name, byInst := range decision.ScorerBreakdown {
				perScorer[name] = byInst[snap.ID]
			}
		}

		cands = append(cands, trace.RoutingTraceCandidate{
			InstanceID:       snap.ID,
			IsChosen:         snap.ID == chosenID,
			CompositeScore:   comp,
			ScorerScores:     perScorer,
			QueueDepth:       snap.QueueDepth,
			BatchSize:        snap.BatchSize,
			InFlightRequests: snap.InFlightRequests,
			KVUtilization:    snap.KVUtilization,
			FreeKVBlocks:     snap.FreeKVBlocks,
		})
	}
	regret := bestScore - chosenScore
	if regret < 0 {
		regret = 0
	}
	return cands, regret
}

// computeCounterfactual builds a ranked list of candidate instances and computes
// regret (how much better the best alternative was compared to the chosen instance).
//
// When scores is non-nil (from WeightedScoring), candidates are ranked by policy scores.
// When scores is nil (RoundRobin, LeastLoaded), a synthetic load-based score is used:
// -(QueueDepth + BatchSize + InFlightRequests), so lower-load instances rank higher (#175).
//
// Returns top-k candidates sorted by score descending and regret (≥ 0).
func computeCounterfactual(chosenID string, scores map[string]float64, snapshots []sim.RoutingSnapshot, k int) ([]trace.CandidateScore, float64) {
	if k <= 0 || len(snapshots) == 0 {
		return nil, 0
	}

	type scored struct {
		snap  sim.RoutingSnapshot
		score float64
	}

	all := make([]scored, len(snapshots))
	var chosenScore float64
	chosenFound := false
	for i, snap := range snapshots {
		s := 0.0
		if scores != nil {
			s = scores[snap.ID]
		} else {
			// Load-based fallback: negative load so lower load ranks higher (#175)
			s = -float64(snap.EffectiveLoad())
		}
		all[i] = scored{snap: snap, score: s}
		if snap.ID == chosenID {
			chosenScore = s
			chosenFound = true
		}
	}

	// If chosen ID not in snapshots (should not happen), return candidates with 0 regret
	if !chosenFound {
		return nil, 0
	}

	// Sort by score descending; tie-break by instance ID ascending for determinism
	sort.Slice(all, func(i, j int) bool {
		if all[i].score != all[j].score {
			return all[i].score > all[j].score
		}
		return all[i].snap.ID < all[j].snap.ID
	})

	// Clamp k to available instances
	n := min(k, len(all))

	result := make([]trace.CandidateScore, n)
	for i := 0; i < n; i++ {
		result[i] = trace.CandidateScore{
			InstanceID:       all[i].snap.ID,
			Score:            all[i].score,
			QueueDepth:       all[i].snap.QueueDepth,
			BatchSize:        all[i].snap.BatchSize,
			InFlightRequests: all[i].snap.InFlightRequests,
			KVUtilization:    all[i].snap.KVUtilization,
			FreeKVBlocks:     all[i].snap.FreeKVBlocks,
		}
	}

	// Regret = best score - chosen score; 0 if chosen is best
	regret := all[0].score - chosenScore
	if regret < 0 {
		regret = 0
	}

	return result, regret
}
