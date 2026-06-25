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
//   - δ_pf-chunk (ITL inflation from prefill-on-decode) is the marginal work of ONE
//     prefill chunk — min(a_p, ChunkTokens) tokens, since only one chunk co-schedules
//     per decode iteration (§3.2: δ_pf(s)=c_pf·s, s = chunk tokens). It is distinct
//     from the full prefill demand W_p, which lands on the backlog/TTFT terms instead.
//   - For MoE models, finite-difference α slightly under-counts because weight-load
//     grows weakly with B via nEff; exact for dense models.
//   - Optimistic in-flight backlog increments (§8.1) are omitted in the base
//     implementation; the z-feedback half of the rule is immune to scrape staleness.

// EDPPConfig holds the controller's fixed knobs. All durations are microseconds.
//
// SLO targets are per-SLO-class. TauTTFTUs/TauITLUs are the defaults applied to any
// class without an explicit entry in the per-class override maps (and to the empty
// "" class). TauTTFTByClassUs/TauITLByClassUs override the defaults for named classes
// (e.g. "critical", "batch"). The decision rule is evaluated from the perspective of
// the request's own class: a stricter class reads the same server backlog as more
// threatening, and its realized-SLO feedback accumulates in its own virtual queue.
type EDPPConfig struct {
	TauTTFTUs        int64            // default τ_ttft: time-average TTFT SLO target (µs)
	TauITLUs         int64            // default τ_itl: time-average ITL SLO target (µs)
	TauRefUs         int64            // fixed reference τ for the transfer-penalty normalization (µs); makes the penalty scale 1/τ_ttft² like the other terms. Independent of the operating τ_ttft.
	TauTTFTByClassUs map[string]int64 // per-class τ_ttft overrides (µs); nil = use default for all
	TauITLByClassUs  map[string]int64 // per-class τ_itl overrides (µs); nil = use default for all
	V                float64          // penalty/stability tradeoff knob (Neely's V); larger ⇒ fewer offloads
	CXferUs          int64            // c_xfer: KV-transfer cost paid when routing P (µs)
	NomPrefillTokens int              // S_nom: nominal prefill chunk for the fixed prefill normalizer
	NomDecodeCtx     int              // L_nom: nominal decode context for the fixed decode normalizer
	BlockSize        int              // token block size for the prefix-cache a_p computation
	ChunkTokens      int              // per-step prefill token budget (max_num_batched_tokens); caps δ_pf-chunk. 0 = no cap (whole prefill counts as one chunk)
	Coeffs           EDPPCoeffs       // frozen E3 latency-law coefficients (design §1.1); required
	TraceEnabled     bool             // when true, Decide attaches an EDPPDecisionTrace (intermediate rule terms) to each decision. Off ⇒ zero allocation.
}

// EDPPDecisionTrace records the intermediate terms of one E14 rule evaluation, for
// diagnostics. It is attached to DisaggregationDecision.EDPPTrace only when the decider
// has TraceEnabled set. Every field is dimensionless or in microseconds; LHS and RHS are
// the two sides of the (E14) inequality and decompose exactly into the listed components:
//
//	LHS = BalanceTermD − BalanceTermP
//	RHS = TransferTerm + TTFTTerm + ITLTerm
//	Disaggregate = LHS > RHS
//
// On early-return paths (empty prompt, fully prefix-cached) the rule is not evaluated;
// SkipReason names the path and the term fields are left zero.
type EDPPDecisionTrace struct {
	Class        string  // request SLO class (drives τ resolution)
	SkipReason   string  // "" = rule evaluated; else "empty-prompt" or "fully-cached"
	Ap           int     // a_p: uncached prompt tokens
	Wp           float64 // W_p: prefill demand (µs)
	DeltaPfChunk float64 // δ_pf-chunk: one-chunk ITL inflation (µs)
	QdRaw        float64 // Q_d: raw decode backlog (µs)
	QpRaw        float64 // Q_p: raw prefill backlog (µs)
	Qd           float64 // normalized decode backlog (Q_d / W*_d)
	Qp           float64 // normalized prefill backlog (Q_p / W*_p)
	MuDNom       float64 // μ_d^nom
	MuPNom       float64 // μ_p^nom
	WStarD       float64 // W*_d normalizer (µs)
	WStarP       float64 // W*_p normalizer (µs)
	TauTTFT      float64 // τ_ttft for this class (µs)
	TauITL       float64 // τ_itl for this class (µs)
	TTFTP        float64 // predicted TTFT under P (µs)
	TTFTD        float64 // predicted TTFT under D (µs)
	ITLP         float64 // predicted ITL under P (µs)
	ITLD         float64 // predicted ITL under D (µs)
	ZTTFT        float64 // normalized TTFT virtual queue (z_ttft = Z_ttft / τ_ttft)
	ZITL         float64 // normalized ITL virtual queue (z_itl = Z_itl / τ_itl)
	BalanceTermD float64 // q_d·(W_p/W*_d)
	BalanceTermP float64 // q_p·(W_p/W*_p)
	TransferTerm float64 // V·(c_xfer/τ_ttft)
	TTFTTerm     float64 // z_ttft·(TTFT_P−TTFT_D)/τ_ttft
	ITLTerm      float64 // z_itl·(ITL_P−ITL_D)/τ_itl
	LHS          float64 // backlog-balancing benefit
	RHS          float64 // transfer penalty + SLO pressure
	Disaggregate bool    // the decision (LHS > RHS)
}

func (c EDPPConfig) validate() {
	if err := c.Coeffs.validate(); err != nil {
		panic(fmt.Sprintf("EDPPConfig: %v", err))
	}
	if float64(c.TauITLUs) <= c.Coeffs.AlphaD {
		panic(fmt.Sprintf("EDPPConfig: TauITLUs (%d µs) must exceed decode α (%.1f µs); the ITL SLO is physically unachievable on this hardware", c.TauITLUs, c.Coeffs.AlphaD))
	}
	switch {
	case c.TauTTFTUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauTTFTUs must be > 0, got %d", c.TauTTFTUs))
	case c.TauITLUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauITLUs must be > 0, got %d", c.TauITLUs))
	case c.TauRefUs <= 0:
		panic(fmt.Sprintf("EDPPConfig: TauRefUs must be > 0, got %d", c.TauRefUs))
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
	for cls, v := range c.TauTTFTByClassUs {
		if v <= 0 {
			panic(fmt.Sprintf("EDPPConfig: TauTTFTByClassUs[%q] must be > 0, got %d", cls, v))
		}
	}
	for cls, v := range c.TauITLByClassUs {
		if v <= 0 {
			panic(fmt.Sprintf("EDPPConfig: TauITLByClassUs[%q] must be > 0, got %d", cls, v))
		}
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
	// OnComplete reports a completed request's SLO class and its realized end-to-end
	// TTFT and mean inter-token latency (both microseconds). The class lets the decider
	// attribute the violation to the correct per-class SLO accumulator.
	OnComplete(sloClass string, realizedTTFTUs, realizedMeanITLUs int64)
}

// edppClassState is the per-SLO-class virtual-queue state (accumulated SLO
// violation, µs). Bumped on each completion (E8).
type edppClassState struct {
	zTTFT, zITL float64
}

// edppNorm bundles the class-resolved normalizers used by a single Decide call.
// μ_p^nom is class-independent (it depends on the prefill iteration time, not the
// SLO); μ_d^nom and both W* depend on the class's τ targets.
type edppNorm struct {
	muDNom, muPNom  float64
	wStarD, wStarP  float64
	tauTTFT, tauITL float64
}

// EDPPDecider implements DisaggregationDecider and SLOFeedbackDecider.
type EDPPDecider struct {
	cfg              EDPPConfig
	model            LatencyModel
	cacheQuery       map[string]func([]int) int // shared with precise-prefix-cache scorer; may be nil
	prefillSnapshots func() []RoutingSnapshot   // prefill-pool backlogs; may be nil (⇒ Q_p = 0)

	// Physics constants precomputed once at construction (class-independent).
	// μ_p^nom is fixed: a moving normalizer would break the Lyapunov drift telescoping
	// and invert the congestion signal (design §4.3).
	alphaD, alphaP       int64
	deltaBarD, deltaBarP int64
	muPNom               float64 // μ_p^nom = 1 − α_p/T_iter^nom (E11); does not depend on τ

	// Frozen calibrated coefficients stored for use by downstream tasks.
	coeffs EDPPCoeffs

	// Per-class controller state: virtual queues keyed by SLO class. Lazily created.
	zByClass map[string]*edppClassState
}

// NewEDPPDecider constructs the decider and precomputes its class-independent physics
// constants from the injected latency model. cfg is validated (panics on invalid
// values, R3). cacheQuery and prefillSnapshots may be nil (e.g. unit tests, or no
// prefill pool). The per-class target maps are copied defensively.
func NewEDPPDecider(cfg EDPPConfig, model LatencyModel, cacheQuery map[string]func([]int) int, prefillSnapshots func() []RoutingSnapshot) *EDPPDecider {
	cfg.validate()
	if model == nil {
		panic("NewEDPPDecider: model must not be nil")
	}
	cfg.TauTTFTByClassUs = copyClassTargets(cfg.TauTTFTByClassUs)
	cfg.TauITLByClassUs = copyClassTargets(cfg.TauITLByClassUs)

	d := &EDPPDecider{
		cfg:              cfg,
		model:            model,
		cacheQuery:       cacheQuery,
		prefillSnapshots: prefillSnapshots,
		zByClass:         make(map[string]*edppClassState),
		coeffs:           cfg.Coeffs,
	}

	// Decode coefficients (nominal context) and prefill coefficients (nominal chunk).
	decodeProbe := edppDecodeProbe(cfg.NomDecodeCtx)
	prefillProbe := edppPrefillProbe(cfg.NomPrefillTokens)
	d.alphaD = edppExtractAlpha(model, decodeProbe)
	d.alphaP = edppExtractAlpha(model, prefillProbe)
	d.deltaBarD = edppMarginalDelta(model, decodeProbe)
	d.deltaBarP = edppMarginalDelta(model, prefillProbe)

	// μ_p^nom = 1 − α/T_iter^nom (E11): prefill at the nominal prefill iteration time.
	tIterPNom := float64(model.StepTime([]*Request{prefillProbe}))
	d.muPNom = clampMu(1.0 - float64(d.alphaP)/tIterPNom)
	return d
}

func copyClassTargets(m map[string]int64) map[string]int64 {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]int64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
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

// targetsFor resolves the (τ_ttft, τ_itl) SLO targets for an SLO class: a per-class
// override if present, else the configured defaults (also used for the empty class).
func (d *EDPPDecider) targetsFor(class string) (tauTTFTUs, tauITLUs int64) {
	tauTTFTUs, tauITLUs = d.cfg.TauTTFTUs, d.cfg.TauITLUs
	if v, ok := d.cfg.TauTTFTByClassUs[class]; ok {
		tauTTFTUs = v
	}
	if v, ok := d.cfg.TauITLByClassUs[class]; ok {
		tauITLUs = v
	}
	return
}

// normFor computes the class-resolved normalizers (E10–E12). μ_d^nom = 1 − α/τ_itl
// and W* = μ^nom · τ_ttft are constant for a given class (α and μ_p^nom are fixed),
// so the §11 signal-direction property holds within each class.
func (d *EDPPDecider) normFor(class string) edppNorm {
	tauTTFTUs, tauITLUs := d.targetsFor(class)
	muD := clampMu(1.0 - float64(d.alphaD)/float64(tauITLUs))
	return edppNorm{
		muDNom:  muD,
		muPNom:  d.muPNom,
		wStarD:  muD * float64(tauTTFTUs),
		wStarP:  d.muPNom * float64(tauTTFTUs),
		tauTTFT: float64(tauTTFTUs),
		tauITL:  float64(tauITLUs),
	}
}

// normalizedBacklogs returns the dimensionless decode/prefill backlogs q_d, q_p (E9)
// under the given class normalizers. Q = (QueueDepth + BatchSize) · δ̄ per pool [µs];
// q = Q / W*. Q grows with batch size, and W* uses fixed μ^nom, so q_d cannot decrease
// as B rises (§11 signal-direction).
func (d *EDPPDecider) normalizedBacklogs(decodeSnaps, prefillSnaps []RoutingSnapshot, n edppNorm) (qd, qp float64) {
	var qD float64
	for _, s := range decodeSnaps {
		qD += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarD)
	}
	var qP float64
	for _, s := range prefillSnaps {
		qP += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarP)
	}
	return qD / n.wStarD, qP / n.wStarP
}

// Decide evaluates the E14 rule and returns Disaggregate=true (route P) when the
// backlog-balancing benefit of offloading exceeds the transfer penalty plus SLO
// pressure. Conservative fallbacks (Disaggregate=false): empty prompt, or a request
// whose prompt is fully prefix-cached on the selected decode pod (no prefill to offload).
func (d *EDPPDecider) Decide(req *Request, state *RouterState) DisaggregationDecision {
	keepD := DisaggregationDecision{Disaggregate: false}
	if len(req.InputTokens) == 0 {
		if d.cfg.TraceEnabled {
			keepD.EDPPTrace = &EDPPDecisionTrace{Class: req.SLOClass, SkipReason: "empty-prompt"}
		}
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
		if d.cfg.TraceEnabled {
			keepD.EDPPTrace = &EDPPDecisionTrace{Class: req.SLOClass, SkipReason: "fully-cached", Ap: ap}
		}
		return keepD // fully cached: no prefill work to disaggregate
	}

	// W_p = full prefill demand of this request (E6): all a_p uncached tokens. Drives
	// the backlog (Q) and TTFT terms — the whole prefill must finish before first token.
	wp := float64(edppMarginalDelta(d.model, edppPrefillProbe(ap)))

	// δ_pf-chunk = marginal work of ONE prefill chunk (§3.2/§6, E15). Only one chunk of
	// min(a_p, ChunkTokens) tokens co-schedules per decode iteration, so this — not the
	// whole W_p — is what inflates the decode batch's ITL on that iteration.
	deltaPfChunk := d.chunkInflation(ap)

	var decodeSnaps []RoutingSnapshot
	if state != nil {
		decodeSnaps = state.Snapshots
	}
	var prefillSnaps []RoutingSnapshot
	if d.prefillSnapshots != nil {
		prefillSnaps = d.prefillSnapshots()
	}

	// Resolve this request's class normalizers/targets — the rule is evaluated from
	// the perspective of the request's own SLO.
	n := d.normFor(req.SLOClass)

	// Backlogs in work-seconds (for the §6 predictors) and normalized (for the LHS).
	var qD float64
	for _, s := range decodeSnaps {
		qD += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarD)
	}
	var qP float64
	for _, s := range prefillSnaps {
		qP += float64(s.QueueDepth+s.BatchSize) * float64(d.deltaBarP)
	}
	qd := qD / n.wStarD
	qp := qP / n.wStarP

	// Current decode ITL (protected under P): measured if available, else nominal.
	itlP := d.selectedDecodeITL(decodeSnaps, state)

	// Counterfactual predictors (§6, E15), all µs.
	ttftP := (qP+wp)/n.muPNom + float64(d.cfg.CXferUs)
	ttftD := (qD + wp) / n.muDNom
	itlD := itlP + deltaPfChunk // E15: prefill chunk inflates this decode iteration's ITL

	// Per-class virtual queues (zero when this class has not yet completed a request).
	var zTTFT, zITL float64
	if z := d.zByClass[req.SLOClass]; z != nil {
		zTTFT = z.zTTFT / n.tauTTFT
		zITL = z.zITL / n.tauITL
	}

	// E14: choose P ⟺ LHS > RHS. Every term is dimensionless. Decomposed into named
	// components so the trace can expose each intermediate; the arithmetic is unchanged.
	balanceTermD := qd * (wp / n.wStarD)
	balanceTermP := qp * (wp / n.wStarP)
	lhs := balanceTermD - balanceTermP

	// Transfer penalty, normalized consistently with the balance and SLO terms (all ∝
	// 1/τ_ttft²). The extra τ_ref/τ_ttft factor corrects the lone 1/τ_ttft outlier that
	// let a loose τ_ttft (which heavy workloads need) shrink the load-balancing benefit
	// faster than the penalty, locking out disaggregation. τ_ref (TauRefUs) is a FIXED
	// reference independent of the operating τ_ttft, so loosening τ_ttft genuinely
	// attenuates the penalty; at τ_ttft == τ_ref the factor is 1 (byte-for-byte unchanged).
	tauRef := float64(d.cfg.TauRefUs)
	transferTerm := d.cfg.V * (float64(d.cfg.CXferUs) / n.tauTTFT) * (tauRef / n.tauTTFT)
	ttftTerm := zTTFT * (ttftP - ttftD) / n.tauTTFT
	itlTerm := zITL * (itlP - itlD) / n.tauITL
	rhs := transferTerm + ttftTerm + itlTerm

	dec := DisaggregationDecision{Disaggregate: lhs > rhs}
	if d.cfg.TraceEnabled {
		dec.EDPPTrace = &EDPPDecisionTrace{
			Class: req.SLOClass, Ap: ap, Wp: wp, DeltaPfChunk: deltaPfChunk,
			QdRaw: qD, QpRaw: qP, Qd: qd, Qp: qp,
			MuDNom: n.muDNom, MuPNom: n.muPNom, WStarD: n.wStarD, WStarP: n.wStarP,
			TauTTFT: n.tauTTFT, TauITL: n.tauITL,
			TTFTP: ttftP, TTFTD: ttftD, ITLP: itlP, ITLD: itlD,
			ZTTFT: zTTFT, ZITL: zITL,
			BalanceTermD: balanceTermD, BalanceTermP: balanceTermP,
			TransferTerm: transferTerm, TTFTTerm: ttftTerm, ITLTerm: itlTerm,
			LHS: lhs, RHS: rhs, Disaggregate: lhs > rhs,
		}
	}
	return dec
}

// chunkInflation returns δ_pf-chunk: the marginal work [µs] of co-scheduling one
// prefill chunk of min(a_p, ChunkTokens) tokens onto a decode iteration (§3.2 §6).
// ChunkTokens == 0 means "no cap" — the whole a_p-token prefill counts as one chunk.
func (d *EDPPDecider) chunkInflation(ap int) float64 {
	chunk := ap
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunk {
		chunk = d.cfg.ChunkTokens
	}
	return float64(edppMarginalDelta(d.model, edppPrefillProbe(chunk)))
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

// OnComplete bumps the completing request's per-class virtual queues from its realized
// latencies (E8). Stabilizing Z enforces the time-average SLO for that class; the
// max(·, 0) floor is the standard Neely update. The class's queues are created lazily.
func (d *EDPPDecider) OnComplete(sloClass string, realizedTTFTUs, realizedMeanITLUs int64) {
	tauTTFTUs, tauITLUs := d.targetsFor(sloClass)
	z := d.zByClass[sloClass]
	if z == nil {
		z = &edppClassState{}
		d.zByClass[sloClass] = z
	}
	z.zTTFT = math.Max(z.zTTFT+float64(realizedTTFTUs-tauTTFTUs), 0)
	z.zITL = math.Max(z.zITL+float64(realizedMeanITLUs-tauITLUs), 0)
}

// Compile-time interface compliance checks.
var (
	_ DisaggregationDecider = (*EDPPDecider)(nil)
	_ SLOFeedbackDecider    = (*EDPPDecider)(nil)
)
