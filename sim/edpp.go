package sim

import (
	"fmt"
	"math"
)

// EDPP (Lyapunov drift-plus-penalty) prefill/decode placement decider.
//
// EDPP realizes the design in docs/superpowers/specs/2026-06-18-edpp-dpp-routing-design.md
// as a DisaggregationDecider: the per-request choice x ∈ {P, D} maps onto
// DisaggregationDecision.Disaggregate (P = true = remote prefill, D = false =
// prefill-on-decode). The rule (E14) trades a transfer penalty against backlog
// balancing and SLO pressure, where the SLO weights are virtual-queue states
// (z_ttft, z_itl) rather than hand-tuned constants.
//
// # BLIS reformulation
//
// The design recovers the iteration-time coefficients α (per-step fixed cost) and
// δ (per-request marginal work) by rolling OLS over scraped vLLM metrics (§3). BLIS
// has no live server to scrape — the injected LatencyModel *is* the ground-truth
// physics — so EDPP recovers the same coefficients by finite-difference on StepTime:
//
//	α    = 2·StepTime([r]) − StepTime([r, r])   (the per-step intercept)
//	δ(r) =   StepTime([r, r]) − StepTime([r])   (one copy's marginal work; cancels α)
//
// StepTime([]) returns 1 (the LatencyModel contract), never α, so the empty batch
// cannot be used to read α directly — hence the two-point difference above.
//
// # Oracle safety (INV-9)
//
// Decide reads only input-side quantities: len(req.InputTokens) and the prefix-cache
// hit (via cacheQuery), never req.OutputTokens. The probe Requests built here are
// synthetic fixtures used to query the latency model's physics; fabricating a probe
// with OutputTokens set does NOT read the real request's hidden output length.
//
// # Modeling residuals (carried from design §10)
//
//   - Backlog Q is estimated as (QueueDepth + BatchSize) · δ̄ per pool — monotone in
//     batch size B, so the §11 signal-direction anchor holds by construction.
//   - δ_pf-chunk (ITL inflation from prefill-on-decode) is approximated by W_p; refine
//     here first if measured ITL spikes during co-scheduled prefill disagree.
//   - For MoE models, finite-difference α slightly under-counts because weight-load
//     grows weakly with B via nEff; exact for dense models.
//   - Optimistic in-flight backlog increments (§8.1) are omitted in the base
//     implementation; the z-feedback half of the rule is immune to scrape staleness.

// EDPPConfig holds the controller's fixed knobs. All durations are microseconds.
type EDPPConfig struct {
	TauTTFTUs        int64   // τ_ttft: time-average TTFT SLO target (µs)
	TauITLUs         int64   // τ_itl: time-average ITL SLO target (µs)
	V                float64 // penalty/stability tradeoff knob (Neely's V); larger ⇒ fewer offloads
	CXferUs          int64   // c_xfer: KV-transfer cost paid when routing P (µs)
	NomPrefillTokens int     // S_nom: nominal prefill chunk for the fixed prefill normalizer
	NomDecodeCtx     int     // L_nom: nominal decode context for the fixed decode normalizer
	BlockSize        int     // token block size for the prefix-cache a_p computation
}

func (c EDPPConfig) validate() {
	switch {
	case c.TauTTFTUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauTTFTUs must be > 0, got %d", c.TauTTFTUs))
	case c.TauITLUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauITLUs must be > 0, got %d", c.TauITLUs))
	case c.V < 0:
		panic(fmt.Sprintf("EDPPConfig: V must be >= 0, got %v", c.V))
	case c.CXferUs < 0:
		panic(fmt.Sprintf("EDPPConfig: CXferUs must be >= 0, got %d", c.CXferUs))
	case c.NomPrefillTokens <= 0:
		panic(fmt.Sprintf("EDPPConfig: NomPrefillTokens must be > 0, got %d", c.NomPrefillTokens))
	case c.NomDecodeCtx <= 0:
		panic(fmt.Sprintf("EDPPConfig: NomDecodeCtx must be > 0, got %d", c.NomDecodeCtx))
	case c.BlockSize <= 0:
		panic(fmt.Sprintf("EDPPConfig: BlockSize must be > 0, got %d", c.BlockSize))
	}
}

// edppMinMu floors the nominal drain rate so the normalizers and predictor
// denominators never collapse to 0 when α ≥ T_iter (a degenerate operating point).
const edppMinMu = 1e-3

// edppPrefillProbe builds a synthetic prefill request processing n tokens this step.
// ProgressIndex (0) < len(InputTokens) (n) selects the latency model's prefill branch.
func edppPrefillProbe(n int) *Request {
	if n < 1 {
		n = 1
	}
	return &Request{InputTokens: make([]int, n), NumNewTokens: n, ProgressIndex: 0}
}

// edppDecodeProbe builds a synthetic decode request at context length ctx.
// ProgressIndex (ctx) >= len(InputTokens) (ctx) and len(OutputTokens) > 0 select
// the latency model's decode branch with Σ context = ctx.
func edppDecodeProbe(ctx int) *Request {
	if ctx < 1 {
		ctx = 1
	}
	return &Request{InputTokens: make([]int, ctx), OutputTokens: []int{0}, NumNewTokens: 1, ProgressIndex: int64(ctx)}
}

// edppExtractAlpha returns the per-step fixed cost α = 2·StepTime([r]) − StepTime([r,r]).
func edppExtractAlpha(m LatencyModel, probe *Request) int64 {
	s1 := m.StepTime([]*Request{probe})
	s2 := m.StepTime([]*Request{probe, probe})
	return 2*s1 - s2
}

// edppMarginalDelta returns one copy's marginal work δ = StepTime([r,r]) − StepTime([r]).
func edppMarginalDelta(m LatencyModel, probe *Request) int64 {
	s1 := m.StepTime([]*Request{probe})
	s2 := m.StepTime([]*Request{probe, probe})
	return s2 - s1
}

// SLOFeedbackDecider is the optional completion-feedback hook. Deciders that
// implement it receive each request's realized TTFT and mean ITL so they can
// update SLO-tracking state. Call sites discover it via a type assertion, so
// adding it does not break the DisaggregationDecider interface.
type SLOFeedbackDecider interface {
	// OnComplete reports a completed request's realized end-to-end TTFT and mean
	// inter-token latency, both in microseconds.
	OnComplete(realizedTTFTUs, realizedMeanITLUs int64)
}

// EDPPDecider implements DisaggregationDecider and SLOFeedbackDecider.
type EDPPDecider struct {
	cfg              EDPPConfig
	model            LatencyModel
	cacheQuery       map[string]func([]int) int // shared with precise-prefix-cache scorer; may be nil
	prefillSnapshots func() []RoutingSnapshot   // prefill-pool backlogs; may be nil (⇒ Q_p = 0)

	// Constants precomputed once at construction (E10–E12). μ^nom is fixed: a moving
	// normalizer would break the Lyapunov drift telescoping and invert the congestion
	// signal (design §4.3).
	alphaD, alphaP       int64
	muDNom, muPNom       float64
	deltaBarD, deltaBarP int64
	wStarD, wStarP       float64

	// Controller state: virtual queues (accumulated SLO violation, µs). Owned by the
	// decider, bumped on each completion (E8).
	zTTFT, zITL float64
}

// NewEDPPDecider constructs the decider and precomputes its fixed normalizers from
// the injected latency model. cfg is validated (panics on invalid values, R3).
// cacheQuery and prefillSnapshots may be nil (e.g. unit tests, or no prefill pool).
func NewEDPPDecider(cfg EDPPConfig, model LatencyModel, cacheQuery map[string]func([]int) int, prefillSnapshots func() []RoutingSnapshot) *EDPPDecider {
	cfg.validate()
	if model == nil {
		panic("NewEDPPDecider: model must not be nil")
	}

	d := &EDPPDecider{
		cfg:              cfg,
		model:            model,
		cacheQuery:       cacheQuery,
		prefillSnapshots: prefillSnapshots,
	}

	// Decode coefficients (nominal context) and prefill coefficients (nominal chunk).
	decodeProbe := edppDecodeProbe(cfg.NomDecodeCtx)
	prefillProbe := edppPrefillProbe(cfg.NomPrefillTokens)
	d.alphaD = edppExtractAlpha(model, decodeProbe)
	d.alphaP = edppExtractAlpha(model, prefillProbe)
	d.deltaBarD = edppMarginalDelta(model, decodeProbe)
	d.deltaBarP = edppMarginalDelta(model, prefillProbe)

	// μ_d^nom = 1 − α/τ_itl (E11): decode at the ITL-SLO-critical operating point.
	d.muDNom = clampMu(1.0 - float64(d.alphaD)/float64(cfg.TauITLUs))
	// μ_p^nom = 1 − α/T_iter^nom (E11): prefill at the nominal prefill iteration time.
	tIterPNom := float64(model.StepTime([]*Request{prefillProbe}))
	d.muPNom = clampMu(1.0 - float64(d.alphaP)/tIterPNom)

	// W* = μ^nom · τ_ttft (E10): the backlog whose induced queueing delay equals one
	// TTFT window. Constant, by design — see §4.3 / §11 signal-direction anchor.
	d.wStarD = d.muDNom * float64(cfg.TauTTFTUs)
	d.wStarP = d.muPNom * float64(cfg.TauTTFTUs)
	return d
}

func clampMu(mu float64) float64 {
	if mu < edppMinMu {
		return edppMinMu
	}
	if mu > 1.0 {
		return 1.0
	}
	return mu
}

// normalizedBacklogs returns the dimensionless decode/prefill backlogs q_d, q_p (E9).
// Q = (QueueDepth + BatchSize) · δ̄ per pool [µs]; q = Q / W*. Q grows with batch size,
// and W* uses fixed μ^nom, so q_d cannot decrease as B rises (§11 signal-direction).
func (d *EDPPDecider) normalizedBacklogs(decodeSnaps, prefillSnaps []RoutingSnapshot) (qd, qp float64) {
	var qD float64
	for _, s := range decodeSnaps {
		qD += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarD)
	}
	var qP float64
	for _, s := range prefillSnaps {
		qP += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarP)
	}
	return qD / d.wStarD, qP / d.wStarP
}

// Decide evaluates the E14 rule and returns Disaggregate=true (route P) when the
// backlog-balancing benefit of offloading exceeds the transfer penalty plus SLO
// pressure. Conservative fallbacks (Disaggregate=false): empty prompt, or a request
// whose prompt is fully prefix-cached on the selected decode pod (no prefill to offload).
func (d *EDPPDecider) Decide(req *Request, state *RouterState) DisaggregationDecision {
	keepD := DisaggregationDecision{Disaggregate: false}
	if len(req.InputTokens) == 0 {
		return keepD
	}

	// a_p = uncached prompt tokens (oracle-safe; same source as PrefixThresholdDecider).
	ap := len(req.InputTokens)
	if state != nil && state.SelectedInstance != "" && d.cacheQuery != nil {
		if fn, ok := d.cacheQuery[state.SelectedInstance]; ok && fn != nil {
			ap = len(req.InputTokens) - fn(req.InputTokens)*d.cfg.BlockSize
		}
	}
	if ap <= 0 {
		return keepD // fully cached: no prefill work to disaggregate
	}

	// W_p = prefill demand of this request (E6), via the latency model's marginal cost.
	wp := float64(edppMarginalDelta(d.model, edppPrefillProbe(ap)))

	var decodeSnaps []RoutingSnapshot
	if state != nil {
		decodeSnaps = state.Snapshots
	}
	var prefillSnaps []RoutingSnapshot
	if d.prefillSnapshots != nil {
		prefillSnaps = d.prefillSnapshots()
	}

	// Backlogs in work-seconds (for the §6 predictors) and normalized (for the LHS).
	var qD float64
	for _, s := range decodeSnaps {
		qD += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarD)
	}
	var qP float64
	for _, s := range prefillSnaps {
		qP += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarP)
	}
	qd := qD / d.wStarD
	qp := qP / d.wStarP

	// Current decode ITL (protected under P): measured if available, else nominal.
	itlP := d.selectedDecodeITL(decodeSnaps, state)

	// Counterfactual predictors (§6, E15), all µs.
	ttftP := (qP+wp)/d.muPNom + float64(d.cfg.CXferUs)
	ttftD := (qD + wp) / d.muDNom
	itlD := itlP + wp // δ_pf-chunk ≈ W_p (design §6 residual)

	tauTTFT := float64(d.cfg.TauTTFTUs)
	tauITL := float64(d.cfg.TauITLUs)
	zTTFT := d.zTTFT / tauTTFT
	zITL := d.zITL / tauITL

	// E14: choose P ⟺ LHS > RHS. Every term is dimensionless.
	lhs := qd*(wp/d.wStarD) - qp*(wp/d.wStarP)
	rhs := d.cfg.V*(float64(d.cfg.CXferUs)/tauTTFT) +
		zTTFT*(ttftP-ttftD)/tauTTFT +
		zITL*(itlP-itlD)/tauITL

	return DisaggregationDecision{Disaggregate: lhs > rhs}
}

// selectedDecodeITL returns the measured ITL of the pre-selected decode pod, falling
// back to the nominal iteration time (α_d + δ̄_d) when no measurement is available.
func (d *EDPPDecider) selectedDecodeITL(decodeSnaps []RoutingSnapshot, state *RouterState) float64 {
	if state != nil && state.SelectedInstance != "" {
		for _, s := range decodeSnaps {
			if s.ID == state.SelectedInstance {
				if s.ITL > 0 {
					return s.ITL
				}
				break
			}
		}
	}
	return float64(d.alphaD + d.deltaBarD)
}

// OnComplete bumps the virtual queues from a realized completion (E8). Stabilizing
// Z enforces the time-average SLO; the max(·, 0) floor is the standard Neely update.
func (d *EDPPDecider) OnComplete(realizedTTFTUs, realizedMeanITLUs int64) {
	d.zTTFT = math.Max(d.zTTFT+float64(realizedTTFTUs-d.cfg.TauTTFTUs), 0)
	d.zITL = math.Max(d.zITL+float64(realizedMeanITLUs-d.cfg.TauITLUs), 0)
}

// Compile-time interface compliance checks.
var (
	_ DisaggregationDecider = (*EDPPDecider)(nil)
	_ SLOFeedbackDecider    = (*EDPPDecider)(nil)
)
