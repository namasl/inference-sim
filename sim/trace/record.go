// Package trace provides decision-trace recording for cluster-level policy analysis.
// This package has no dependencies on sim/ or sim/cluster/ — it stores pure data types.
package trace

// AdmissionRecord captures a single admission policy decision.
type AdmissionRecord struct {
	RequestID string
	Clock     int64
	Admitted  bool
	Reason    string
}

// CandidateScore captures a counterfactual candidate instance with its score and state.
type CandidateScore struct {
	InstanceID       string
	Score            float64
	QueueDepth       int
	BatchSize        int
	InFlightRequests int
	KVUtilization    float64
	FreeKVBlocks     int64
}

// RoutingRecord captures a single routing policy decision with optional counterfactual analysis.
type RoutingRecord struct {
	RequestID      string
	Clock          int64
	ChosenInstance string
	Reason         string
	Scores         map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates     []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret         float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// DisaggregationRecord captures a PD disaggregation decision.
// When Disaggregate=true, the request follows the disaggregated path. Paired
// PrefillRoutingRecord and KVTransferRecord are recorded for the same RequestID
// except in two drop scenarios:
//
//  1. No routable prefill pool instances (routingRejections++): no downstream records.
//  2. Decode-side drop (droppedAtDecodeKV++): PrefillRoutingRecord exists but
//     KVTransferRecord is absent. Two sub-cases, both counted the same:
//     - At KVTransferStartedEvent: decode pod became non-routable, or
//     ReserveTransferredKV fails (insufficient decode KV). Drop fires at
//     transfer start; no KVTransferCompletedEvent is scheduled. Issue #1343.
//     - At KVTransferCompletedEvent: decode pod became non-routable between
//     transfer start and transfer complete. Reserved KV is released.
//
// To detect case 1: check for the absence of a PrefillRoutingRecord with a
// matching ParentRequestID in the trace. To detect case 2: check the
// absence of a KVTransferRecord for a given ParentRequestID.
// Note: DecodeRoutingRecord is never emitted; the decode pod is pre-selected at
// executeDisaggregatedRouting time (no second routing decision).
type DisaggregationRecord struct {
	RequestID string
	// Clock is the simulation time at which the routing (and disaggregation)
	// decision was made: admission_time + routingLatency. Since #1261 unified
	// the PD and standard routing paths under RoutingDecisionEvent, this value
	// reflects the routing-fire time rather than the admission time.
	Clock        int64
	Disaggregate bool // true = routed to prefill pool; false = standard routing to decode pool
}

// PrefillRoutingRecord captures a prefill pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type PrefillRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Values are raw weighted-scorer outputs; not normalized. Nil when scoring is disabled.
	Scores     map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// DecodeRoutingRecord captures a decode pool routing decision with optional counterfactual analysis.
// ParentRequestID equals the RequestID in the corresponding DisaggregationRecord for this request.
type DecodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Values are raw weighted-scorer outputs; not normalized. Nil when scoring is disabled.
	Scores     map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64            // max(alternative scores) - score(chosen); 0 if chosen is best; always >= 0
}

// EncodeRoutingRecord captures an encode pool routing decision (GAP-4, issue #1264).
// Emitted when an EncodeDecider approves encoding for a request during
// executeDisaggregatedRouting. Under option A (zero-duration encode), the record
// reflects a routing decision only — no encode sub-request is injected.
// ParentRequestID equals the request ID (one encode record per parent).
type EncodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	// Scores maps instance ID → composite routing score (higher = more preferred).
	// Nil when scoring is disabled.
	Scores     map[string]float64
	Candidates []CandidateScore // top-k candidates sorted by score desc (nil if k=0)
	Regret     float64          // max(alternative scores) - score(chosen); >= 0
}

// EDPPDecisionRecord captures the intermediate terms of one EDPP (E14) rule evaluation
// for a request, for diagnostic analysis of why the decider chose P or D. It is a flat
// mirror of sim.EDPPDecisionTrace (this package has no dependency on sim/), plus the
// request ID and the decision clock. Recorded only when the EDPP decider has tracing
// enabled and trace-level=decisions. The two sides compose exactly:
//
//	LHS = BalanceTermD − BalanceTermP
//	RHS = TransferTerm + TTFTTerm + ITLTerm
//	Disaggregate = LHS > RHS
//
// On early-return paths SkipReason names the path ("empty-prompt"/"fully-cached") and the
// term fields are zero.
type EDPPDecisionRecord struct {
	RequestID    string
	Clock        int64
	Class        string
	SkipReason   string
	Ap           int
	Wp           float64
	DeltaPfChunk float64
	QdRaw        float64
	QpRaw        float64
	Qd           float64
	Qp           float64
	MuDNom       float64
	MuPNom       float64
	WStarD       float64
	WStarP       float64
	TauTTFT      float64
	TauITL       float64
	TTFTP        float64
	TTFTD        float64
	ITLP         float64
	ITLD         float64
	ZTTFT        float64
	ZITL         float64
	BalanceTermD float64
	BalanceTermP float64
	TransferTerm float64
	TTFTTerm     float64
	ITLTerm      float64
	LHS          float64
	RHS          float64
	Disaggregate bool
}

// RoutingTraceCandidate is one candidate instance considered during a routing
// target selection, captured for the --routing-decision-trace CSV.
type RoutingTraceCandidate struct {
	InstanceID     string
	IsChosen       bool
	CompositeScore float64            // weighted composite (RoutingDecision.Scores); 0 for non-scoring policies
	ScorerScores   map[string]float64 // scorer name → raw clamped [0,1] score; nil for non-scoring policies
	QueueDepth     int
	BatchSize      int
	// InFlightRequests is the dispatched-but-not-completed count as the router saw
	// it, INCLUDING decode targets reserved at selection but not yet transferred
	// (the in-flight reservation, commit 6a97a2f) — so reserved-pending decodes are
	// reflected here.
	InFlightRequests int
	KVUtilization    float64
	FreeKVBlocks     int64
}

// RoutingDecisionTraceRecord captures one routing target selection (prefill,
// decode, standard, or encode) with the full candidate set, for the
// --routing-decision-trace CSV. One record per selection; the CSV writer emits
// one row per candidate.
type RoutingDecisionTraceRecord struct {
	Clock          int64
	Stage          string // "standard" | "prefill" | "decode" | "encode"
	RequestID      string
	ChosenInstance string
	Regret         float64 // best composite − chosen composite (≥0); 0 for non-scoring policies
	Candidates     []RoutingTraceCandidate
}

// KVTransferRecord captures a KV cache transfer event between prefill and decode instances.
// TransferDuration is always >= 0; negative values are clamped to 0 with a warning in
// KVTransferCompletedEvent.Execute() (sim/cluster/pd_events.go) if INV-PD-4 is ever violated.
type KVTransferRecord struct {
	ParentRequestID   string
	TransferStartTime int64 // microseconds (sim clock)
	TransferDuration  int64 // microseconds; >= 0 (clamped at recording site)
	NumKVBlocks       int64
	PrefillInstanceID string
	DecodeInstanceID  string
}
