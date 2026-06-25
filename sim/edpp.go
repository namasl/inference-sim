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
// Physics coefficients (α, δ, c_pf) come from the frozen EDPPCoeffs in cfg.Coeffs,
// loaded once at construction. The model parameter is retained for the deferred
// recalibration-drift watchdog (§4) and is otherwise unused.
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

// SLOFeedbackDecider is the lifecycle hook. OnRoute fires once when a request is
// committed to a pool (increments the work backlog Q). OnComplete fires at the
// request's terminal completion (decrements Q, bumps the virtual queues, and
// updates the per-class output-length estimate N̂_out). Forget releases the backlog
// for a routed request that reaches a terminal state WITHOUT a normal completion
// (timeout/drop) — no z bump, no N̂_out update. Call sites discover it via a type
// assertion, so adding it does not break DisaggregationDecider.
//
// key is an explicit, stable correlation id chosen by the caller: it MUST be the
// same value at OnRoute and at the matching OnComplete/Forget, so the conservation
// bookkeeping decrements exactly the work the route added. For PD-disaggregated
// requests the routed request and the completing decode sub-request have different
// Request.IDs, so the cluster passes the parent identity here (see the cluster's
// feedSLOFeedback). The decider never parses the key; it only uses it as a map key.
type SLOFeedbackDecider interface {
	OnRoute(req *Request, key string, toPrefill bool, apTokens int)               // increment Q at admission
	OnComplete(req *Request, key string, realizedTTFTUs, realizedMeanITLUs int64) // decrement Q, bump z, update N̂_out
	Forget(key string)                                                            // release Q for a terminally non-completed routed request
}

// edppPendingWork records the work a routed request contributed, so OnComplete can
// remove exactly that amount (conservation; design §6 conservation form).
type edppPendingWork struct {
	toPrefill bool
	wp, wd    float64 // µs added to qp/qd respectively
}

// edppRunningMean is a per-class running mean of realized output lengths (N̂_out).
type edppRunningMean struct {
	n   int64
	sum float64
}

func (r *edppRunningMean) update(v float64) { r.n++; r.sum += v }
func (r *edppRunningMean) mean() float64 {
	if r.n == 0 {
		return 1 // no completions yet: 1-token decode estimate (conservative seed)
	}
	return r.sum / float64(r.n)
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
	muPNom float64 // μ_p^nom = 1 − α_p/T_iter^nom (E11); does not depend on τ

	// Frozen calibrated coefficients stored for use by downstream tasks.
	coeffs EDPPCoeffs

	// Per-class controller state: virtual queues keyed by SLO class. Lazily created.
	zByClass map[string]*edppClassState

	// Conservation bookkeeping (design §6 conservation form).
	qpWork, qdWork float64                     // running work-µs backlogs
	pending        map[string]edppPendingWork  // per-request work, keyed by Request.ID
	nHatOut        map[string]*edppRunningMean // per-class realized output-length estimate
}

// NewEDPPDecider constructs the decider and initializes class-independent physics
// constants from cfg.Coeffs. cfg is validated (panics on invalid values, R3).
// model is retained for the deferred recalibration-drift watchdog (§4).
// cacheQuery and prefillSnapshots may be nil (e.g. unit tests, or no prefill pool).
// The per-class target maps are copied defensively.
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
		pending:          make(map[string]edppPendingWork),
		nHatOut:          make(map[string]*edppRunningMean),
	}

	// μ_p^nom = 1 − α_p/(α_p + c_pf·S_pf^nom) (design §7): from frozen coefficients.
	d.muPNom = d.coeffs.muPNom(cfg.NomPrefillTokens)
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
	muD := d.coeffs.muDNom(float64(tauITLUs))
	return edppNorm{
		muDNom:  muD,
		muPNom:  d.muPNom,
		wStarD:  muD * float64(tauTTFTUs),
		wStarP:  d.muPNom * float64(tauTTFTUs),
		tauTTFT: float64(tauTTFTUs),
		tauITL:  float64(tauITLUs),
	}
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

	// W_p = full prefill demand of this request (E6), from frozen coeffs.
	wp := d.coeffs.Wp(ap)

	// Live decode-server state from the pre-selected decode snapshot (the pod this
	// request would land on); fall back to the first snapshot, else nominal.
	bDec, kv, sPf := d.selectedDecodeState(state)
	muDec := d.coeffs.muDecode(bDec, kv, sPf)
	tBminus1 := d.coeffs.tIterDecode(bDec, kv, sPf)

	// Prefill-server live state: S_pf summed over prefill snapshots.
	var sPfPrefill int64
	var prefillSnaps []RoutingSnapshot
	if d.prefillSnapshots != nil {
		prefillSnaps = d.prefillSnapshots()
		for _, s := range prefillSnaps {
			sPfPrefill += s.ResidentPrefillTokens
		}
	}
	muPf := d.coeffs.muPrefill(sPfPrefill)

	n := d.normFor(req.SLOClass)

	// Conservation-bookkept backlogs (Task 6), in work-µs.
	qP := d.qpWork
	qD := d.qdWork
	qd := qD / n.wStarD
	qp := qP / n.wStarP

	// Explicit-chunk predictors (§5.1, divergence §9.1). chunk = decode batched-token
	// budget; n_chunks = ⌈a_p/chunk⌉; δ_pf-chunk = c_pf·chunk.
	chunk := ap
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunk {
		chunk = d.cfg.ChunkTokens
	}
	nChunks := math.Ceil(float64(ap) / float64(chunk))
	deltaPfChunk := d.coeffs.CPf * float64(chunk)
	ttftP := qP/muPf + nChunks*(d.coeffs.AlphaP+deltaPfChunk) + float64(d.cfg.CXferUs)
	ttftD := qD/muDec + nChunks*(tBminus1+deltaPfChunk)

	// Per-class virtual queues.
	var zTTFT, zITL float64
	if z := d.zByClass[req.SLOClass]; z != nil {
		zTTFT = z.zTTFT / n.tauTTFT
		zITL = z.zITL / n.tauITL
	}

	// E14, with the ITL term in collapsed closed form (§5.2/§9.2):
	//   z_itl·(ITL_P − ITL_D)/τ_itl = − z_itl·(c_pf·chunk)/τ_itl
	balanceTermD := qd * (wp / n.wStarD)
	balanceTermP := qp * (wp / n.wStarP)
	lhs := balanceTermD - balanceTermP

	tauRef := float64(d.cfg.TauRefUs)
	transferTerm := d.cfg.V * (float64(d.cfg.CXferUs) / n.tauTTFT) * (tauRef / n.tauTTFT)
	ttftTerm := zTTFT * (ttftP - ttftD) / n.tauTTFT
	itlTerm := -zITL * (d.coeffs.CPf * float64(chunk)) / n.tauITL
	rhs := transferTerm + ttftTerm + itlTerm

	dec := DisaggregationDecision{Disaggregate: lhs > rhs}
	if d.cfg.TraceEnabled {
		dec.EDPPTrace = &EDPPDecisionTrace{
			Class: req.SLOClass, Ap: ap, Wp: wp, DeltaPfChunk: deltaPfChunk,
			QdRaw: qD, QpRaw: qP, Qd: qd, Qp: qp,
			MuDNom: n.muDNom, MuPNom: n.muPNom, WStarD: n.wStarD, WStarP: n.wStarP,
			TauTTFT: n.tauTTFT, TauITL: n.tauITL,
			TTFTP: ttftP, TTFTD: ttftD,
			ZTTFT: zTTFT, ZITL: zITL,
			BalanceTermD: balanceTermD, BalanceTermP: balanceTermP,
			TransferTerm: transferTerm, TTFTTerm: ttftTerm, ITLTerm: itlTerm,
			LHS: lhs, RHS: rhs, Disaggregate: lhs > rhs,
		}
	}
	return dec
}

// selectedDecodeState returns (B_dec, KV, S_pf) for the decode pod this request
// would land on (state.SelectedInstance), falling back to the first decode
// snapshot, then to a nominal single-request decode batch.
func (d *EDPPDecider) selectedDecodeState(state *RouterState) (bDec int, kv, sPf int64) {
	if state != nil {
		snaps := state.Snapshots
		if state.SelectedInstance != "" {
			for _, s := range snaps {
				if s.ID == state.SelectedInstance {
					return s.BatchSize, s.KvTokensInUse, s.ResidentPrefillTokens
				}
			}
		}
		if len(snaps) > 0 {
			s := snaps[0]
			return s.BatchSize, s.KvTokensInUse, s.ResidentPrefillTokens
		}
	}
	return 1, int64(d.cfg.NomDecodeCtx), 0
}

// nHatFor returns the per-class running-mean output length, lazily created.
func (d *EDPPDecider) nHatFor(class string) *edppRunningMean {
	m := d.nHatOut[class]
	if m == nil {
		m = &edppRunningMean{}
		d.nHatOut[class] = m
	}
	return m
}

// OnRoute increments the work backlog for a committed request (design §6.1,
// conservation form). apTokens is the uncached prompt token count (input-only; INV-9
// safe). W_d uses the class N̂_out estimate at the nominal decode context.
func (d *EDPPDecider) OnRoute(req *Request, key string, toPrefill bool, apTokens int) {
	if apTokens <= 0 {
		return
	}
	wp := d.coeffs.Wp(apTokens)
	wd := d.nHatFor(req.SLOClass).mean() * d.coeffs.deltaBarDecode(float64(d.cfg.NomDecodeCtx))
	pw := edppPendingWork{toPrefill: toPrefill}
	if toPrefill {
		pw.wp = wp // prefill work lands on the prefill pool
		pw.wd = wd // decode-only work lands on the decode pool
		d.qpWork += wp
		d.qdWork += wd
	} else {
		pw.wd = wp + wd // mixed prefill+decode on the decode pool
		d.qdWork += wp + wd
	}
	d.pending[key] = pw
}

// OnComplete removes the request's work (conservation), bumps the per-class
// virtual queues from realized latencies (E8), and updates N̂_out. Reading the
// realized output length here is post-completion and INV-9-permitted (§6.3).
func (d *EDPPDecider) OnComplete(req *Request, key string, realizedTTFTUs, realizedMeanITLUs int64) {
	if pw, ok := d.pending[key]; ok {
		d.qpWork -= pw.wp
		d.qdWork -= pw.wd
		if d.qpWork < 0 {
			d.qpWork = 0
		}
		if d.qdWork < 0 {
			d.qdWork = 0
		}
		delete(d.pending, key)
	}
	d.nHatFor(req.SLOClass).update(float64(len(req.OutputTokens)))

	tauTTFTUs, tauITLUs := d.targetsFor(req.SLOClass)
	z := d.zByClass[req.SLOClass]
	if z == nil {
		z = &edppClassState{}
		d.zByClass[req.SLOClass] = z
	}
	z.zTTFT = math.Max(z.zTTFT+float64(realizedTTFTUs-tauTTFTUs), 0)
	z.zITL = math.Max(z.zITL+float64(realizedMeanITLUs-tauITLUs), 0)
}

// Forget releases the conservation backlog a routed request contributed when that
// request reaches a terminal state WITHOUT a normal completion (timeout/drop): it
// removes pending[key] and decrements qp/qd by the stored amounts (0-clamped). It
// deliberately does NOT bump the virtual queues z or touch N̂_out — a request that
// never completed carries no realized SLO signal and no realized output length. It
// is idempotent: forgetting an unknown key is a no-op. This is the conservation
// counterpart to OnComplete for the non-completion terminal paths (design §6).
func (d *EDPPDecider) Forget(key string) {
	pw, ok := d.pending[key]
	if !ok {
		return
	}
	d.qpWork -= pw.wp
	d.qdWork -= pw.wd
	if d.qpWork < 0 {
		d.qpWork = 0
	}
	if d.qdWork < 0 {
		d.qdWork = 0
	}
	delete(d.pending, key)
}

// BacklogForTest exposes the conservation bookkeeping state for tests only: the
// running prefill/decode work backlogs and the number of still-pending routed
// requests. It lets cluster-package tests assert observable conservation (Q→0,
// pending empty) without reaching into unexported fields. Not part of any contract;
// do not use outside tests.
func (d *EDPPDecider) BacklogForTest() (qp, qd float64, pendingLen int) {
	return d.qpWork, d.qdWork, len(d.pending)
}

// Compile-time interface compliance checks.
var (
	_ DisaggregationDecider = (*EDPPDecider)(nil)
	_ SLOFeedbackDecider    = (*EDPPDecider)(nil)
)
