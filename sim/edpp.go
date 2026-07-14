package sim

import (
	"fmt"
	"math"
	"sort"
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
// # Physics source
//
// EDPP's iteration-time coefficients {α, α_p, c0, c1, c_pf, c_attn} are FROZEN,
// loaded from a calibration JSON (scripts/calibration/coeffs-*.json) via
// LoadEDPPCoeffs and carried in EDPPConfig.Coeffs. They feed the demand W_p (§3),
// the live drain rates μ (§4), the normalizers W* (§7), and the counterfactual
// predictors (§5). See docs/superpowers/specs/2026-06-24-edpp-rule-coeffs-wiring-design.md.
// The model parameter on NewEDPPDecider is retained for the deferred
// recalibration-drift watchdog (§4) and is otherwise unused.
//
// # Backlog accounting
//
// Q_p/Q_d are the WAITING backlog (design §1.2, §6.1): work of requests routed to a
// server but not yet admitted to its running batch. OnRoute adds a request's work;
// OnAdmit removes the admitted side's share (Q_p when the prefill sub-request enters
// the prefill batch, Q_d when the decode/normal request enters the decode batch);
// OnComplete updates only z and N̂_out (the running batch is captured by T(B−1)/μ in
// the predictors, not by Q). Forget releases any still-waiting share on terminal
// non-completion (timeout/drop). Decode work uses the per-class running-mean output
// length N̂_out (INV-9: realized length read only at completion). The TTFT predictors
// are symmetric co-residency: TTFT_x = Q_x^wait/μ_x + n·(T_x(B−1)+δ_pf-chunk) [+c_xfer].
//
// # Oracle safety (INV-9)
//
// Decide reads only input-side quantities: len(req.InputTokens) and the prefix-cache
// hit (via cacheQuery), never req.OutputTokens. Per-class N̂_out is updated only at
// completion from realized output length — not used for servability decisions, so
// INV-9 is not violated.
//
// # Modeling residuals (carried from design §10)
//
//   - δ_pf-chunk (ITL inflation from prefill-on-decode) is the marginal work of ONE
//     prefill chunk — min(a_p, ChunkTokens) tokens, since only one chunk co-schedules
//     per decode iteration (§3.2: δ_pf(s)=c_pf·s, s = chunk tokens). It is distinct
//     from the full prefill demand W_p, which lands on the backlog/TTFT terms instead.
//   - For MoE models, c0/c1 slightly under-count because weight-load grows weakly
//     with B via nEff; exact for dense models.
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
	TauTTFTUs         int64                 // default τ_ttft: time-average TTFT SLO target (µs)
	TauITLUs          int64                 // default τ_itl: time-average ITL SLO target (µs)
	TauRefUs          int64                 // fixed reference τ for the transfer-penalty normalization (µs); makes the penalty scale 1/τ_ttft² like the other terms. Independent of the operating τ_ttft.
	TauTTFTByClassUs  map[string]int64      // per-class τ_ttft overrides (µs); nil = use default for all
	TauITLByClassUs   map[string]int64      // per-class τ_itl overrides (µs); nil = use default for all
	V                 float64               // penalty/stability tradeoff knob (Neely's V); larger ⇒ fewer offloads
	CXferUs           int64                 // c_xfer: KV-transfer cost paid when routing P (µs)
	NomPrefillTokens  int                   // S_nom: nominal prefill chunk for the fixed prefill normalizer
	NomDecodeCtx      int                   // L_nom: nominal decode context for the fixed decode normalizer
	BlockSize         int                   // token block size for the prefix-cache a_p computation
	ChunkTokens       int                   // per-step prefill token budget (max_num_batched_tokens); caps δ_pf-chunk. 0 = no cap (whole prefill counts as one chunk)
	Coeffs            EDPPCoeffs            // frozen E3 latency-law coefficients (design §1.1); required
	CoeffsByGPU       map[string]EDPPCoeffs // per-GPU-type θ_i overrides; nil ⇒ use Coeffs for every candidate (homogeneous)
	TraceEnabled      bool                  // when true, Decide attaches an EDPPDecisionTrace (intermediate rule terms) to each decision. Off ⇒ zero allocation.
	TAdmEstimator     string                // admission-delay estimator name ("" ⇒ waiting, the current formula)
	Joint             bool                  // when true, Decide enumerates all (decode, prefill) candidates and picks the drift-plus-penalty argmin (joint P/D routing, --edpp-joint); false ⇒ the reduced fixed-d local-vs-disagg rule.
	JointTraceEnabled bool                  // when true (joint mode only), decideJoint attaches an EDPPJointDecisionTrace comparing the scorer's (d,p) pick to the joint argmin. Off ⇒ zero allocation, no shadow prefill scorer run.
}

// EDPPJointDecisionTrace records, for one joint (--edpp-joint) decision, the scorer's
// pick vs the joint argmin, so an analysis pass can quantify how often (and by how much)
// the joint objective overrides the composable scorer. It is attached to
// DisaggregationDecision.EDPPJointTrace only when the decider has JointTraceEnabled set,
// and is pure instrumentation — computing it does not change the routing decision (INV-6).
//
// ScorerD is the decode-routing policy's pick (state.SelectedInstance). ScorerP is the
// prefill-routing policy's pick, obtained by SHADOW-running the injected prefill scorer
// over the prefill snapshots (compute-only, not acted on); it is populated only on
// disaggregate decisions (JointP != ""). JScorer is J evaluated at the scorer's slice —
// disagg (ScorerD, ScorerP) on a disagg decision, else local on ScorerD — and JJoint is
// the argmin's J. Because the argmin ranges over a superset that always includes the
// scorer's slice, JJoint <= JScorer by construction (asserted as an internal invariant).
type EDPPJointDecisionTrace struct {
	Class        string  // request SLO class
	ScorerD      string  // decode-routing policy pick (state.SelectedInstance)
	JointD       string  // joint argmin decode node
	ScorerP      string  // shadow prefill-scorer pick (disagg only; else "")
	JointP       string  // joint argmin prefill node (disagg only; else "")
	AgreeD       bool    // ScorerD == JointD
	AgreeP       bool    // ScorerP == JointP (trivially true when both empty, i.e. local)
	JScorer      float64 // J at the scorer's slice
	JJoint       float64 // J at the joint argmin (== the committed decision's cost)
	Disaggregate bool    // the joint decision (disagg vs local)
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
	ITLP         float64 // retained for compatibility; always 0 — ITL pressure is now the collapsed ITLTerm (Task 7 design); Decide no longer sets this field
	ITLD         float64 // retained for compatibility; always 0 — ITL pressure is now the collapsed ITLTerm (Task 7 design); Decide no longer sets this field
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
// committed to a pool (increments the work backlog Q). OnAdmit fires when a routed
// request first enters a running batch (drains the admitted side's waiting backlog
// share). OnComplete fires at the request's terminal completion (bumps the virtual
// queues and updates the per-class output-length estimate N̂_out — backlog is already
// gone via OnAdmit). Forget releases any remaining backlog for a routed request that
// reaches a terminal state WITHOUT a normal completion (timeout/drop) — no z bump, no
// N̂_out update. Call sites discover it via a type assertion, so adding it does not
// break DisaggregationDecider.
//
// key is an explicit, stable correlation id chosen by the caller: it MUST be the
// same value at OnRoute and at the matching OnAdmit/OnComplete/Forget, so the
// conservation bookkeeping decrements exactly the work the route added. For
// PD-disaggregated requests the routed request and the completing decode sub-request
// have different Request.IDs, so the cluster passes the parent identity here (see the
// cluster's feedSLOFeedback / feedAdmission). The decider never parses the key; it
// only uses it as a map key.
type SLOFeedbackDecider interface {
	OnRoute(req *Request, key string, toPrefill bool, apTokens int, decodeInst, prefillInst string) // increment Q at routing (per-pool + per-instance)
	OnAdmit(key string, prefillSide bool)                                                           // drain waiting-work share at admission
	OnFirstToken(key string, nowUs int64)                                                           // true up z_ttft from the realized first-token time
	OnComplete(req *Request, key string, realizedTTFTUs, realizedMeanITLUs int64)                   // bump z_itl, update N̂_out (z_ttft owned by OnFirstToken)
	Forget(key string)                                                                              // release Q for a terminally non-completed routed request
}

// edppAwaiting tracks a routed request still awaiting its first token, so z_ttft can be
// credited continuously from its observed elapsed wait (a certain lower bound on its TTFT
// miss) instead of only once at completion. startUs is the request's arrival time (when
// the TTFT clock starts); credited is the cumulative amount already applied to z_ttft for
// this request, so each sweep applies only the increment (no double-count).
type edppAwaiting struct {
	startUs  int64
	class    string
	credited float64
}

// edppPendingWork records the remaining waiting-work shares of a routed request.
// wp/wd are the µs still counted in qp/qd respectively; OnAdmit zeroes the
// admitted side's share and removes the entry once both are zero. Forget drains
// whatever share remains when the request reaches a terminal non-completion state.
type edppPendingWork struct {
	toPrefill bool
	wp, wd    float64 // remaining µs in qp/qd (drained to 0 as each side is admitted)

	// Routed instances recorded at OnRoute so the drains (OnAdmit/Forget) can mirror
	// the pool-scalar decrement per-instance. decodeInst carries the wd share;
	// prefillInst carries the wp share (disagg only). "" is ignored (nil-safe).
	decodeInst, prefillInst string
}

// edppInstWork is the per-instance waiting-work accumulator (µs). It mirrors the
// pool-level qp/qd scalars split by the routed instance, for the future joint
// P/D routing reader; the reduced (pool-level) path never reads it.
type edppInstWork struct {
	wp, wd float64
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

	// Per-GPU-type θ_i (design 2026-07-14). nil ⇒ homogeneous: coeffsFor returns coeffs.
	coeffsByGPU map[string]EDPPCoeffs

	// joint selects the (decode, prefill) argmin enumeration path (--edpp-joint).
	// false ⇒ the reduced fixed-d local-vs-disagg rule (byte-identical legacy path).
	joint bool

	// Pluggable admission-delay estimator (default "waiting" reproduces QWork/Mu).
	tadmEstimator AdmissionDelayEstimator

	// Per-class controller state: virtual queues keyed by SLO class. Lazily created.
	zByClass map[string]*edppClassState

	// Conservation bookkeeping (design §6 conservation form).
	qpWork, qdWork float64                    // running work-µs backlogs
	pending        map[string]edppPendingWork // per-request work, keyed by Request.ID

	// Per-instance congestion queues Q_i: the same waiting-work the pool scalars
	// track, split by the routed instance. Populated always (in OnRoute/drained in
	// OnAdmit/Forget) but READ only by the future joint P/D routing path. Invariant:
	// sum over instances of wp == qpWork, and of wd == qdWork.
	qByInstance map[string]*edppInstWork
	nHatOut     map[string]*edppRunningMean // per-class realized output-length estimate

	// Requests routed but not yet at their first token, keyed by the same conservation
	// key as pending. Drives the continuous z_ttft credit (design: responsive z_ttft).
	awaitingFirstToken map[string]*edppAwaiting

	// captureAdmissionCtx, when set, makes Decide attach the assembled per-pool
	// AdmissionContext(s) to the returned DisaggregationDecision so the cluster's
	// --edpp-admission-trace companion trace can recompute all six estimator predictions
	// at end of run. Off by default (zero-cost). Logging-only path (INV-9).
	captureAdmissionCtx bool

	// prefillScorer, when set, is the injected prefill-routing scorer used ONLY to
	// shadow-compute EDPPJointDecisionTrace.ScorerP on disaggregate joint decisions.
	// It returns the instance ID the prefill routing policy would pick over the given
	// prefill snapshots. It MUST NOT perturb production routing state (the cluster wires
	// a dedicated-RNG policy instance) — the shadow pick is logged, never acted on
	// (INV-6). nil ⇒ ScorerP is left empty and J_scorer falls back to the local slice.
	prefillScorer func(*Request, []RoutingSnapshot) string
}

// SetPrefillScorer injects the shadow prefill-routing scorer used to populate
// EDPPJointDecisionTrace.ScorerP (joint divergence trace). Logging-only; the returned
// pick is never acted on and must not mutate production routing RNG (INV-6).
func (d *EDPPDecider) SetPrefillScorer(fn func(*Request, []RoutingSnapshot) string) {
	d.prefillScorer = fn
}

// SetCaptureAdmissionContext toggles attaching the assembled per-pool AdmissionContext
// to each DisaggregationDecision (for the --edpp-admission-trace companion trace).
// Off by default; enabling it does not affect the routing decision itself (INV-9).
func (d *EDPPDecider) SetCaptureAdmissionContext(v bool) { d.captureAdmissionCtx = v }

// NewEDPPDecider constructs the decider and initializes class-independent physics
// constants from cfg.Coeffs. cfg is validated (panics on invalid values, R3).
// model is retained for the deferred recalibration-drift watchdog (§4).
// cacheQuery and prefillSnapshots may be nil (e.g. unit tests, or no prefill pool).
// The per-class target maps are copied defensively.
func NewEDPPDecider(cfg EDPPConfig, model LatencyModel, cacheQuery map[string]func([]int) int, prefillSnapshots func() []RoutingSnapshot) *EDPPDecider {
	cfg.validate()
	for gpu, c := range cfg.CoeffsByGPU {
		if err := c.validate(); err != nil {
			panic(fmt.Sprintf("NewEDPPDecider: coeffs_by_gpu[%q]: %v", gpu, err))
		}
	}
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
		joint:            cfg.Joint,
		zByClass:         make(map[string]*edppClassState),
		coeffs:           cfg.Coeffs,
		coeffsByGPU:      cfg.CoeffsByGPU,
		pending:          make(map[string]edppPendingWork),
		qByInstance:      make(map[string]*edppInstWork),
		nHatOut:          make(map[string]*edppRunningMean),

		awaitingFirstToken: make(map[string]*edppAwaiting),
	}

	// INV-9 guard: oracle admission estimators read TRUE remaining output, so they
	// may only be logged (via NewAdmissionEstimator directly), never drive routing.
	if !IsDeployableEstimator(cfg.TAdmEstimator) {
		panic(fmt.Sprintf("NewEDPPDecider: oracle admission estimators are logging-only, not routing drivers (TAdmEstimator=%q)", cfg.TAdmEstimator))
	}
	est, err := NewAdmissionEstimator(cfg.TAdmEstimator)
	if err != nil {
		// Library boundary: mirror cfg.validate()'s panic-on-invalid-config style (R3).
		panic(fmt.Sprintf("NewEDPPDecider: %v", err))
	}
	d.tadmEstimator = est

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

// coeffsFor returns the per-GPU-type θ_i for gpuType, or the global coeffs when
// no override exists (nil map, unmapped type, or empty gpuType). This is the single
// selection point for per-instance heterogeneous coefficients; with no coeffs_by_gpu
// it returns d.coeffs for every candidate, preserving byte-identity (INV-6).
func (d *EDPPDecider) coeffsFor(gpuType string) EDPPCoeffs {
	if gpuType != "" {
		if c, ok := d.coeffsByGPU[gpuType]; ok {
			return c
		}
	}
	return d.coeffs
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
	// Credit z_ttft from the observed elapsed wait of all in-flight requests still
	// awaiting their first token, so the SLO virtual queue reflects trouble happening
	// right now (not only past completions). Lazy: runs exactly when z is about to be
	// read. state.Clock is the current sim time; nil state (unit tests) ⇒ skip.
	if state != nil {
		d.creditAwaiting(state.Clock)
	}

	// Joint P/D routing (--edpp-joint): enumerate all (decode, prefill) candidates and
	// pick the drift-plus-penalty argmin. The reduced fixed-d rule below is left untouched
	// and stays byte-identical (INV-6) — the guard is the only change to its entry.
	if d.joint {
		return d.decideJoint(req, state)
	}

	keepD := DisaggregationDecision{Disaggregate: false}
	if len(req.InputTokens) == 0 {
		if d.cfg.TraceEnabled {
			keepD.EDPPTrace = &EDPPDecisionTrace{Class: req.SLOClass, SkipReason: "empty-prompt"}
		}
		return keepD
	}

	// a_p = uncached prompt tokens (oracle-safe; same source as PrefixThresholdDecider).
	sel := ""
	if state != nil {
		sel = state.SelectedInstance
	}
	ap := d.apForInstance(req, sel)
	if ap <= 0 {
		if d.cfg.TraceEnabled {
			keepD.EDPPTrace = &EDPPDecisionTrace{Class: req.SLOClass, SkipReason: "fully-cached", Ap: ap}
		}
		return keepD // fully cached: no prefill work to disaggregate
	}

	// W_p = full prefill demand of this request (E6), from frozen coeffs.
	wp := d.coeffs.Wp(ap, len(req.InputTokens))

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
	// Occupancy inputs for the admission-delay estimators (fluid/rollforward, Tasks 4/5).
	// Decode side reads the selected decode snapshot; prefill side reads the first prefill snapshot.
	decSnap, _ := d.selectedDecodeSnapshot(state)
	// ReqKVNeed: KV blocks this request needs ≈ ⌈a_r / blockSize⌉ (a_r = full input length; oracle-safe).
	reqKVNeed := int64(0)
	if d.cfg.BlockSize > 0 {
		reqKVNeed = int64((len(req.InputTokens) + d.cfg.BlockSize - 1) / d.cfg.BlockSize)
	}
	// RemainingStepsEst (deployable): per-running-request censored estimate, NOT a mean that
	// can go negative. A request that has produced StepsDone tokens has o_r ≥ StepsDone
	// (censored lower bound), so floor the class output estimate by the max in-flight elapsed.
	remStepsEst := d.decodeRemStepsEst(decSnap, req.SLOClass)
	var prefillSnap RoutingSnapshot
	if len(prefillSnaps) > 0 {
		prefillSnap = prefillSnaps[0]
	}
	// Prefill-pool RemainingStepsEst: symmetric to the decode censored estimate, but
	// derived from the prefill snapshot's own running-prefill occupants (RunningPrefill)
	// so the ttft_p estimators are live on the prefill pool. StepsDone/TrueRemaining here
	// are prefill-chunk counts (see Simulator.RunningPrefillState). Floors at 1 so fluid's
	// wave term is non-zero when the prefill batch is full.
	prefillRemStepsEst := d.prefillRemStepsEst(prefillSnap)
	// AdmissionRate (req/µs) for the little estimator: prefer the explicit field if a
	// source ever populates it, else fall back to the observed completion rate
	// (DispatchRate, req/s → req/µs). In steady state admission ≈ completion (§3.8),
	// so DispatchRate is the arrival-free observable for L̄_q/λ_adm.
	decAdmRate := admissionRateFromSnapshot(decSnap)
	prefillAdmRate := admissionRateFromSnapshot(prefillSnap)

	prefillCtx := AdmissionContext{
		QWork: qP, Mu: muPf,
		BatchSize: prefillSnap.BatchSize, MaxBatchSize: int(prefillSnap.MaxBatchSize),
		FreeKVBlocks: prefillSnap.FreeKVBlocks, ReqKVNeed: reqKVNeed,
		TIter: d.coeffs.tIterPrefill(sPfPrefill), QueueDepth: prefillSnap.QueueDepth,
		AdmissionRate: prefillAdmRate, RemainingStepsEst: prefillRemStepsEst,
		// INV-9 asymmetry: prefill remaining (inLen − ProgressIndex) is known input, so
		// it is deployable — NOT censored. Decode's Running below stays censored.
		Running: prefillSnap.RunningPrefill,
	}
	decodeCtx := AdmissionContext{
		QWork: qD, Mu: muDec,
		BatchSize: decSnap.BatchSize, MaxBatchSize: int(decSnap.MaxBatchSize),
		FreeKVBlocks: decSnap.FreeKVBlocks, ReqKVNeed: reqKVNeed,
		TIter: tBminus1, QueueDepth: decSnap.QueueDepth,
		AdmissionRate: decAdmRate, RemainingStepsEst: remStepsEst,
		Running: censorOracleRemaining(decSnap.RunningDecode),
	}
	tAdmP := d.tadmEstimator.EstimateTAdm(prefillCtx)
	tAdmD := d.tadmEstimator.EstimateTAdm(decodeCtx)
	ttftP := tAdmP + nChunks*(d.coeffs.tIterPrefill(sPfPrefill)+deltaPfChunk) + float64(d.cfg.CXferUs)
	ttftD := tAdmD + nChunks*(tBminus1+deltaPfChunk)

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

	transferTerm := d.transferPenalty(n)
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
	if d.captureAdmissionCtx {
		dc := decodeCtx
		dec.AdmissionCtxDecode = &dc
		if len(prefillSnaps) > 0 {
			pc := prefillCtx
			dec.AdmissionCtxPrefill = &pc
		}
	}
	return dec
}

// apForInstance returns a_p, the uncached prompt tokens for req on instance instID
// (block-aligned prefix-cache query; oracle-safe, input-only per INV-9). instID == ""
// or an absent/nil cacheQuery entry yields the full prompt length (cold). Shared by the
// reduced path (with instID = state.SelectedInstance) and the per-candidate joint path.
func (d *EDPPDecider) apForInstance(req *Request, instID string) int {
	ap := len(req.InputTokens)
	if instID != "" && d.cacheQuery != nil {
		if fn, ok := d.cacheQuery[instID]; ok && fn != nil {
			ap = len(req.InputTokens) - fn(req.InputTokens)*d.cfg.BlockSize
		}
	}
	return ap
}

// decodeRemStepsEst is the deployable per-running-request censored remaining-steps
// estimate for a decode snapshot (design: NOT a mean that can go negative). A request
// that has produced StepsDone tokens has o_r ≥ StepsDone (censored lower bound), so the
// class output estimate is floored by the max in-flight elapsed. Returns 1 when the
// snapshot has no running decode occupants. Shared by the reduced and joint paths.
func (d *EDPPDecider) decodeRemStepsEst(snap RoutingSnapshot, class string) float64 {
	n := len(snap.RunningDecode)
	if n == 0 {
		return 1.0
	}
	nHatOut := d.nHatFor(class).mean()
	var maxSteps int64
	for _, r := range snap.RunningDecode {
		if r.StepsDone > maxSteps {
			maxSteps = r.StepsDone
		}
	}
	nHatEff := math.Max(nHatOut, float64(maxSteps)) // censored: N̂_out ≥ longest in-flight elapsed
	var sum float64
	for _, r := range snap.RunningDecode {
		sum += math.Max(nHatEff-float64(r.StepsDone), 1) // per-request remaining, floored at 1
	}
	return sum / float64(n)
}

// prefillRemStepsEst is the symmetric censored remaining-steps estimate for a prefill
// snapshot, derived from its running-prefill occupants' TrueRemaining (prefill-chunk
// counts, floored at 1). Returns 1 when the snapshot has no running prefill occupants.
// Shared by the reduced and joint paths.
func (d *EDPPDecider) prefillRemStepsEst(snap RoutingSnapshot) float64 {
	n := len(snap.RunningPrefill)
	if n == 0 {
		return 1.0
	}
	var sum float64
	for _, r := range snap.RunningPrefill {
		rem := r.TrueRemaining
		if rem < 1 {
			rem = 1
		}
		sum += float64(rem)
	}
	return sum / float64(n)
}

// transferPenalty is the normalized KV-transfer cost paid when offloading prefill:
// V·(c_xfer/τ_ttft)·(τ_ref/τ_ttft) (design §3; scales like 1/τ_ttft² as the other terms).
// Shared by the reduced path (RHS) and the joint path (added to disagg candidates).
func (d *EDPPDecider) transferPenalty(n edppNorm) float64 {
	return d.cfg.V * (float64(d.cfg.CXferUs) / n.tauTTFT) * (float64(d.cfg.TauRefUs) / n.tauTTFT)
}

// instWorkRaw reads the per-instance waiting-work backlog (µs) for id WITHOUT creating a
// map entry (unlike instWork). Absent id ⇒ (0, 0). Used by the joint path to read the
// per-candidate normalized congestion q_i = W_i / W*.
func (d *EDPPDecider) instWorkRaw(id string) (wp, wd float64) {
	if w, ok := d.qByInstance[id]; ok {
		return w.wp, w.wd
	}
	return 0, 0
}

// chunkTerms returns (n_chunks, δ_pf-chunk) for a_p uncached prefill tokens under the
// decode batched-token budget (ChunkTokens). a_p ≤ 0 (fully cached / empty) ⇒ (0, 0):
// no prefill work, no per-chunk ITL inflation.
func (d *EDPPDecider) chunkTerms(ap int) (nChunks, deltaPfChunk float64) {
	if ap <= 0 {
		return 0, 0
	}
	chunk := ap
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunk {
		chunk = d.cfg.ChunkTokens
	}
	return math.Ceil(float64(ap) / float64(chunk)), d.coeffs.CPf * float64(chunk)
}

// reqKVNeed is ⌈a_r / blockSize⌉ (a_r = full input length; oracle-safe). 0 when BlockSize ≤ 0.
func (d *EDPPDecider) reqKVNeed(req *Request) int64 {
	if d.cfg.BlockSize <= 0 {
		return 0
	}
	return int64((len(req.InputTokens) + d.cfg.BlockSize - 1) / d.cfg.BlockSize)
}

// decideJoint implements the joint P/D routing rule (--edpp-joint): it enumerates every
// (decode d, placement) candidate — local (prefill+decode on d) and disagg (decode on d,
// prefill on each prefill node p) — computes the NORMALIZED absolute objective J(d,·) for
// each, and returns the argmin. Unlike the reduced rule (which differences ttft_p−ttft_d
// for a single fixed d), J carries every term with its divisor and uses the ABSOLUTE
// per-candidate admission-delay predictor T̂(a), so the argmin correctly ranks distinct
// decode nodes. Restricted to the scorer's single d it reproduces the reduced local-vs-
// disagg decision (§5.5; see TestJoint_ReducesToScorerSliceMatchesReduced).
//
// Determinism (INV-6): decode and prefill snapshots are iterated in ascending instance-ID
// order, local is considered before disagg, and a strictly-lower J (by more than 1e-12)
// is required to replace the incumbent — so ties resolve to the lowest-index, local-first
// candidate, matching the reduced rule's strict Disaggregate = LHS > RHS tie handling.
func (d *EDPPDecider) decideJoint(req *Request, state *RouterState) DisaggregationDecision {
	class := req.SLOClass
	n := d.normFor(class)

	// Per-class virtual queues, normalized by their τ (as in the reduced rule).
	var zTTFT, zITL float64
	if z := d.zByClass[class]; z != nil {
		zTTFT = z.zTTFT / n.tauTTFT
		zITL = z.zITL / n.tauITL
	}

	// Deterministic candidate ordering (INV-6): sort snapshots by instance ID.
	decodeSnaps := sortedSnapshotsByID(stateSnapshots(state))
	var prefillSnaps []RoutingSnapshot
	if d.prefillSnapshots != nil {
		prefillSnaps = sortedSnapshotsByID(d.prefillSnapshots())
	}

	reqKVNeed := d.reqKVNeed(req)
	nHatOut := d.nHatFor(class).mean()
	wd := d.coeffs.Wd(len(req.InputTokens), nHatOut)

	// Base decode-step ITL marginal m_dec(d) = δ̄_dec at mean context (design §3
	// z_itl term). m_dec is candidate-invariant under homogeneous θ (so argmin-invariant
	// now) but is included for §3 fidelity and becomes discriminating under per-instance
	// θ_i (sub-project 2). Added to BOTH local and disagg J (decode happens on d either way).
	mDec := d.coeffs.deltaBarDecode(float64(len(req.InputTokens)) + nHatOut/2)
	jDecodeITL := zITL * (mDec / n.tauITL)

	// Candidate-invariant terms shared by every (d,·) cost evaluation this decision.
	ec := &jointEvalCtx{
		req: req, n: n, zTTFT: zTTFT, zITL: zITL,
		reqKVNeed: reqKVNeed, wd: wd, jDecodeITL: jDecodeITL,
	}

	var best *cand
	consider := func(c cand) {
		if best == nil || c.J < best.J-1e-12 {
			cc := c
			best = &cc
		}
	}

	for _, ds := range decodeSnaps {
		// --- local: prefill+decode co-resident on d ---
		consider(cand{dID: ds.ID, local: true, J: d.jointCandidateCost(ec, ds, nil)})
		// --- disagg: decode on d, prefill on each prefill node p ---
		for _, ps := range prefillSnaps {
			psCopy := ps
			consider(cand{dID: ds.ID, pID: ps.ID, local: false, J: d.jointCandidateCost(ec, ds, &psCopy)})
		}
	}

	if best == nil {
		return DisaggregationDecision{Disaggregate: false} // no candidates (empty snapshots)
	}
	var dec DisaggregationDecision
	if best.local {
		dec = DisaggregationDecision{Disaggregate: false, DecodePodOverride: best.dID}
	} else {
		dec = DisaggregationDecision{Disaggregate: true, DecodePodOverride: best.dID, PrefillPodHint: best.pID}
	}
	// Scorer-vs-joint divergence trace (pure instrumentation, gated; INV-6): compute only
	// when enabled, after the decision is committed, so it can never influence the argmin.
	if d.cfg.JointTraceEnabled {
		dec.EDPPJointTrace = d.buildJointTrace(ec, state, decodeSnaps, prefillSnaps, best)
	}
	return dec
}

// cand is one enumerated joint (decode, placement) candidate: local (prefill co-resident
// on the decode node) or disagg (decode on dID, prefill on pID), with its objective J.
type cand struct {
	dID, pID string
	local    bool
	J        float64
}

// jointEvalCtx holds the per-decision, candidate-invariant terms shared by every (d,·)
// cost evaluation in one joint decision, so jointCandidateCost can be called both from the
// argmin enumeration and from the scorer-slice shadow evaluation (divergence trace) without
// recomputing them — and so both paths produce byte-identical arithmetic (INV-6).
type jointEvalCtx struct {
	req        *Request
	n          edppNorm
	zTTFT      float64
	zITL       float64
	reqKVNeed  int64
	wd         float64
	jDecodeITL float64
}

// jointCandidateCost evaluates the normalized joint objective J(d, ·) for one candidate:
// local (ps == nil, prefill co-resident on the decode node ds) or disagg (decode on ds,
// prefill on *ps). It reproduces exactly the arithmetic the argmin enumeration uses, so the
// enumeration and the scorer-slice shadow evaluation share one code path. The decode-side
// terms depend only on ds and are recomputed per call with identical operands (byte-identical
// float result, INV-6).
func (d *EDPPDecider) jointCandidateCost(ec *jointEvalCtx, ds RoutingSnapshot, ps *RoutingSnapshot) float64 {
	n := ec.n
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := d.coeffs.tIterDecode(bDec, kv, sPfD)
	_, qdRaw := d.instWorkRaw(ds.ID)
	qd := qdRaw / n.wStarD
	decodeCtx := AdmissionContext{
		QWork: qdRaw, Mu: d.coeffs.muDecode(bDec, kv, sPfD),
		BatchSize: ds.BatchSize, MaxBatchSize: int(ds.MaxBatchSize),
		FreeKVBlocks: ds.FreeKVBlocks, ReqKVNeed: ec.reqKVNeed,
		TIter: tIterD, QueueDepth: ds.QueueDepth,
		AdmissionRate: admissionRateFromSnapshot(ds), RemainingStepsEst: d.decodeRemStepsEst(ds, ec.req.SLOClass),
		Running: censorOracleRemaining(ds.RunningDecode),
	}
	tAdmD := d.tadmEstimator.EstimateTAdm(decodeCtx)

	// Decode backlog term (same for local and disagg on this d — cancels within a d,
	// distinguishes across d): q_d·(W_d/W*_d).
	jDecodeBacklog := qd * (ec.wd / n.wStarD)

	if ps == nil {
		// --- local: prefill+decode co-resident on d ---
		apLoc := d.apForInstance(ec.req, ds.ID)
		nChunksLoc, deltaPfLoc := d.chunkTerms(apLoc)
		wpLoc := d.coeffs.Wp(maxInt(apLoc, 0), len(ec.req.InputTokens))
		tHatLocal := tAdmD + nChunksLoc*(tIterD+deltaPfLoc) // ABSOLUTE T̂_local(d)
		return jDecodeBacklog +
			qd*(wpLoc/n.wStarD) +
			ec.zTTFT*(tHatLocal/n.tauTTFT) +
			ec.jDecodeITL + // base decode-step ITL marginal m_dec (candidate-invariant under homogeneous θ)
			ec.zITL*(deltaPfLoc/n.tauITL) // prefill-on-decode ITL inflation lands on local only
	}

	// --- disagg: decode on d, prefill on node *ps ---
	apP := d.apForInstance(ec.req, ps.ID)
	nChunksP, deltaPfP := d.chunkTerms(apP)
	wpP := d.coeffs.Wp(maxInt(apP, 0), len(ec.req.InputTokens))
	qpRaw, _ := d.instWorkRaw(ps.ID)
	qp := qpRaw / n.wStarP
	sPfP := ps.ResidentPrefillTokens
	tIterP := d.coeffs.tIterPrefill(sPfP)
	prefillCtx := AdmissionContext{
		QWork: qpRaw, Mu: d.coeffs.muPrefill(sPfP),
		BatchSize: ps.BatchSize, MaxBatchSize: int(ps.MaxBatchSize),
		FreeKVBlocks: ps.FreeKVBlocks, ReqKVNeed: ec.reqKVNeed,
		TIter: tIterP, QueueDepth: ps.QueueDepth,
		AdmissionRate: admissionRateFromSnapshot(*ps), RemainingStepsEst: d.prefillRemStepsEst(*ps),
		Running: ps.RunningPrefill,
	}
	tAdmP := d.tadmEstimator.EstimateTAdm(prefillCtx)
	tHatDisagg := tAdmP + nChunksP*(tIterP+deltaPfP) + float64(d.cfg.CXferUs) // ABSOLUTE T̂_disagg(d,p)
	return jDecodeBacklog +
		qp*(wpP/n.wStarP) +
		ec.zTTFT*(tHatDisagg/n.tauTTFT) +
		ec.jDecodeITL + // base decode-step ITL marginal m_dec (same as local: decode is on d)
		d.transferPenalty(n) // disagg pays the KV-transfer penalty; no local ITL inflation
}

// buildJointTrace assembles the scorer-vs-joint divergence record for a committed joint
// decision (best). It reads the decode scorer's pick from state.SelectedInstance and, on a
// disaggregate decision, SHADOW-runs the injected prefill scorer for scorer_p — compute-only,
// never acted on. J_scorer is J at the scorer's slice (disagg (scorer_d, scorer_p) on a
// disagg decision, else local on scorer_d); since the argmin ranges over a superset that
// includes that slice, J_joint <= J_scorer always (asserted in tests as an internal invariant).
func (d *EDPPDecider) buildJointTrace(ec *jointEvalCtx, state *RouterState, decodeSnaps, prefillSnaps []RoutingSnapshot, best *cand) *EDPPJointDecisionTrace {
	tr := &EDPPJointDecisionTrace{
		Class:        ec.req.SLOClass,
		JointD:       best.dID,
		JJoint:       best.J,
		Disaggregate: !best.local,
	}
	if !best.local {
		tr.JointP = best.pID
	}

	// Scorer's decode pick (state.SelectedInstance); resolve its snapshot for the J eval,
	// falling back to the first candidate when the pre-selection is absent/unknown.
	if state != nil {
		tr.ScorerD = state.SelectedInstance
	}
	tr.AgreeD = tr.ScorerD == tr.JointD
	scorerDSnap, ok := findSnapshotByID(decodeSnaps, tr.ScorerD)
	if !ok {
		scorerDSnap = decodeSnaps[0] // decodeSnaps is non-empty here (best != nil)
	}

	if best.local {
		tr.AgreeP = true // both prefill nodes empty ⇒ trivially agree
		tr.JScorer = d.jointCandidateCost(ec, scorerDSnap, nil)
		return tr
	}

	// Disagg decision: shadow-run the prefill scorer for scorer_p (logging-only, INV-6).
	if d.prefillScorer != nil {
		tr.ScorerP = d.prefillScorer(ec.req, prefillSnaps)
	}
	tr.AgreeP = tr.ScorerP == tr.JointP
	if sp, ok := findSnapshotByID(prefillSnaps, tr.ScorerP); ok {
		tr.JScorer = d.jointCandidateCost(ec, scorerDSnap, &sp)
	} else {
		// No usable shadow prefill pick ⇒ score the scorer's local slice (still an
		// enumerated candidate, so the J_joint <= J_scorer invariant is preserved).
		tr.JScorer = d.jointCandidateCost(ec, scorerDSnap, nil)
	}
	return tr
}

// findSnapshotByID returns the snapshot with the given ID (ok=false when id is empty or absent).
func findSnapshotByID(snaps []RoutingSnapshot, id string) (RoutingSnapshot, bool) {
	if id == "" {
		return RoutingSnapshot{}, false
	}
	for _, s := range snaps {
		if s.ID == id {
			return s, true
		}
	}
	return RoutingSnapshot{}, false
}

// stateSnapshots returns state.Snapshots (nil-safe).
func stateSnapshots(state *RouterState) []RoutingSnapshot {
	if state == nil {
		return nil
	}
	return state.Snapshots
}

// sortedSnapshotsByID returns a copy of snaps sorted ascending by instance ID, so the
// joint enumeration order (and its deterministic tie-break) is stable across runs (INV-6).
func sortedSnapshotsByID(snaps []RoutingSnapshot) []RoutingSnapshot {
	if len(snaps) == 0 {
		return nil
	}
	out := make([]RoutingSnapshot, len(snaps))
	copy(out, snaps)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// selectedDecodeSnapshot returns the decode snapshot this request would land on
// (state.SelectedInstance), falling back to the first snapshot; ok is false when
// no snapshot is available (nominal-state path).
func (d *EDPPDecider) selectedDecodeSnapshot(state *RouterState) (snap RoutingSnapshot, ok bool) {
	if state == nil {
		return RoutingSnapshot{}, false
	}
	snaps := state.Snapshots
	if state.SelectedInstance != "" {
		for _, s := range snaps {
			if s.ID == state.SelectedInstance {
				return s, true
			}
		}
	}
	if len(snaps) > 0 {
		return snaps[0], true
	}
	return RoutingSnapshot{}, false
}

// censorOracleRemaining returns a deep copy of the running-request slice with every
// TrueRemaining censored to -1, so the runtime routing driver (the deployable
// admission-delay estimator) can never consume oracle output-length info via
// Running[].TrueRemaining (INV-9 on the control path). This mirrors the logging-path
// stripOracle in cluster.BuildAdmissionRecords. The deployable RemainingStepsEst is the
// only remaining-steps signal the routing estimators may read. The copy is required
// because the snapshot slice (e.g. decSnap.RunningDecode) is shared and read elsewhere;
// mutating it in place would corrupt the shared state. Returns nil for an empty input.
func censorOracleRemaining(running []RunningReqState) []RunningReqState {
	if len(running) == 0 {
		return nil
	}
	out := make([]RunningReqState, len(running))
	copy(out, running)
	for i := range out {
		out[i].TrueRemaining = -1
	}
	return out
}

// admissionRateFromSnapshot returns the admission rate (req/µs) for the little
// estimator. It prefers the explicit AdmissionRate field when populated, else
// derives it from the observed completion rate DispatchRate (req/s → req/µs);
// admission ≈ completion in steady state (§3.8). Returns 0 when neither is available.
func admissionRateFromSnapshot(snap RoutingSnapshot) float64 {
	if snap.AdmissionRate > 0 {
		return snap.AdmissionRate
	}
	if snap.DispatchRate > 0 {
		return snap.DispatchRate / 1e6
	}
	return 0
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

// ensureZ returns the per-class virtual-queue state, lazily creating it.
func (d *EDPPDecider) ensureZ(class string) *edppClassState {
	z := d.zByClass[class]
	if z == nil {
		z = &edppClassState{}
		d.zByClass[class] = z
	}
	return z
}

// creditAwaiting credits z_ttft with the certain lower-bound TTFT miss of every in-flight
// request still awaiting its first token. For a request that arrived at startUs and has no
// first token by nowUs, max(nowUs−startUs−τ, 0) is a guaranteed lower bound on its eventual
// miss; we apply only the increment over what was already credited (so repeated sweeps do
// not double-count). Iteration is in sorted-key order so the floating-point accumulation is
// byte-identical across runs (INV-6). The full realized miss is reconciled at first token
// (OnFirstToken) or, as a fallback, at completion (OnComplete) — see the design's
// faithfulness invariant: same total contribution, credited earlier.
func (d *EDPPDecider) creditAwaiting(nowUs int64) {
	if len(d.awaitingFirstToken) == 0 {
		return
	}
	keys := make([]string, 0, len(d.awaitingFirstToken))
	for k := range d.awaitingFirstToken {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		rec := d.awaitingFirstToken[k]
		tauTTFTUs, _ := d.targetsFor(rec.class)
		lb := float64(nowUs - rec.startUs - tauTTFTUs)
		if lb < 0 {
			lb = 0
		}
		delta := lb - rec.credited
		if delta == 0 {
			continue
		}
		z := d.ensureZ(rec.class)
		z.zTTFT = math.Max(z.zTTFT+delta, 0)
		rec.credited = lb
	}
}

// OnFirstToken trues up z_ttft to the realized TTFT (nowUs − startUs) for a request that
// has just produced its first token, applying the signed delta over what creditAwaiting
// already credited (which may be negative when the SLO was met), then stops tracking it.
// Idempotent: a second call for the same key (e.g. the prefill sub-request of a PD request
// after the decode sub-request already fired) finds no record and no-ops.
func (d *EDPPDecider) OnFirstToken(key string, nowUs int64) {
	rec, ok := d.awaitingFirstToken[key]
	if !ok {
		return
	}
	tauTTFTUs, _ := d.targetsFor(rec.class)
	target := float64(nowUs-rec.startUs) - float64(tauTTFTUs)
	z := d.ensureZ(rec.class)
	z.zTTFT = math.Max(z.zTTFT+(target-rec.credited), 0)
	delete(d.awaitingFirstToken, key)
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
func (d *EDPPDecider) OnRoute(req *Request, key string, toPrefill bool, apTokens int, decodeInst, prefillInst string) {
	// Track every routed request for the continuous z_ttft credit, regardless of apTokens
	// (a fully-cached request still has a TTFT and can still wait). The TTFT clock starts
	// at arrival.
	d.awaitingFirstToken[key] = &edppAwaiting{startUs: req.ArrivalTime, class: req.SLOClass}

	if apTokens <= 0 {
		return
	}
	wp := d.coeffs.Wp(apTokens, len(req.InputTokens))
	// W_d now uses the exact discrete decode sum Wd(a_r, N̂_out); it no longer uses NomDecodeCtx.
	wd := d.coeffs.Wd(len(req.InputTokens), d.nHatFor(req.SLOClass).mean())
	pw := edppPendingWork{toPrefill: toPrefill, decodeInst: decodeInst, prefillInst: prefillInst}
	if toPrefill {
		pw.wp = wp // prefill work lands on the prefill pool
		pw.wd = wd // decode-only work lands on the decode pool
		d.qpWork += wp
		d.qdWork += wd
		// Mirror per-instance: prefill share → prefillInst, decode share → decodeInst.
		d.instWork(prefillInst).wp += wp
		d.instWork(decodeInst).wd += wd
	} else {
		pw.wd = wp + wd // mixed prefill+decode on the decode pool
		d.qdWork += wp + wd
		d.instWork(decodeInst).wd += wp + wd
	}
	d.pending[key] = pw
}

// instWork returns (creating if needed) the per-instance work accumulator for id.
// id == "" is ignored and returns a throwaway accumulator (nil-safe), so an unknown
// instance never panics and never pollutes the map with a "" key.
func (d *EDPPDecider) instWork(id string) *edppInstWork {
	if id == "" {
		return &edppInstWork{}
	}
	w, ok := d.qByInstance[id]
	if !ok {
		w = &edppInstWork{}
		d.qByInstance[id] = w
	}
	return w
}

// drainInst decrements the per-instance accumulator by (wp, wd), mirroring the
// pool-scalar drain, and 0-clamps like the pool path. id == "" or an unknown id is
// a no-op (the work was never attributed to a live instance). It never resurrects a
// deleted key beyond what instWork already created at OnRoute.
func (d *EDPPDecider) drainInst(id string, wp, wd float64) {
	if id == "" {
		return
	}
	w, ok := d.qByInstance[id]
	if !ok {
		return
	}
	w.wp -= wp
	w.wd -= wd
	if w.wp < 0 {
		w.wp = 0
	}
	if w.wd < 0 {
		w.wd = 0
	}
}

// QByInstance returns a snapshot copy of the per-instance congestion queues Q_i
// (waiting work µs, split into prefill Wp and decode Wd). For tests and the future
// joint P/D routing reader; the reduced (pool-level) path does not consume it.
func (d *EDPPDecider) QByInstance() map[string]struct{ Wp, Wd float64 } {
	out := make(map[string]struct{ Wp, Wd float64 }, len(d.qByInstance))
	for id, w := range d.qByInstance {
		out[id] = struct{ Wp, Wd float64 }{Wp: w.wp, Wd: w.wd}
	}
	return out
}

// OnAdmit removes the waiting-work share of a routed request that has just been
// admitted to a running batch (design §6.2, event-exact drain). prefillSide=true
// drains the Q_p share (prefill sub-request admitted to the prefill server);
// prefillSide=false drains the Q_d share (decode/normal request admitted to the
// decode server). pending[key] is deleted once both shares are gone. Idempotent:
// a re-admission (e.g. after preemption) finds the share already 0 and no-ops.
func (d *EDPPDecider) OnAdmit(key string, prefillSide bool) {
	pw, ok := d.pending[key]
	if !ok {
		return
	}
	if prefillSide {
		d.qpWork -= pw.wp
		d.drainInst(pw.prefillInst, pw.wp, 0) // mirror the wp decrement per-instance
		pw.wp = 0
	} else {
		d.qdWork -= pw.wd
		d.drainInst(pw.decodeInst, 0, pw.wd) // mirror the wd decrement per-instance
		pw.wd = 0
	}
	if d.qpWork < 0 {
		d.qpWork = 0
	}
	if d.qdWork < 0 {
		d.qdWork = 0
	}
	if pw.wp == 0 && pw.wd == 0 {
		delete(d.pending, key)
	} else {
		d.pending[key] = pw
	}
}

// OnComplete bumps the per-class virtual queues from realized latencies (E8) and
// updates N̂_out. It does NOT drain the waiting backlog — that work left q_p/q_d at
// admission (OnAdmit, §6.2). Reading realized output length here is post-completion
// and INV-9-permitted (§6.3).
func (d *EDPPDecider) OnComplete(req *Request, key string, realizedTTFTUs, realizedMeanITLUs int64) {
	d.nHatFor(req.SLOClass).update(float64(len(req.OutputTokens)))
	tauTTFTUs, tauITLUs := d.targetsFor(req.SLOClass)
	z := d.ensureZ(req.SLOClass)
	z.zITL = math.Max(z.zITL+float64(realizedMeanITLUs-tauITLUs), 0)

	// z_ttft is normally owned by the continuous credit + OnFirstToken true-up. If the
	// first-token hook never fired for this request (record still present), fall back to
	// truing up from the realized TTFT here so the signal is never lost — same total
	// contribution as the pre-change completion-time bump.
	if rec, ok := d.awaitingFirstToken[key]; ok {
		target := float64(realizedTTFTUs) - float64(tauTTFTUs)
		z.zTTFT = math.Max(z.zTTFT+(target-rec.credited), 0)
		delete(d.awaitingFirstToken, key)
	}
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
	d.drainInst(pw.prefillInst, pw.wp, 0) // mirror wp per-instance
	d.drainInst(pw.decodeInst, 0, pw.wd)  // mirror wd per-instance
	if d.qpWork < 0 {
		d.qpWork = 0
	}
	if d.qdWork < 0 {
		d.qdWork = 0
	}
	delete(d.pending, key)
	// Keep any z_ttft credit already applied: a request that waited past its target and
	// then dropped/timed-out is a real SLO failure (a censored observation), not a
	// non-event. Just stop tracking it for further credit.
	delete(d.awaitingFirstToken, key)
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
