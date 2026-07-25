package sim

import "math"

// Value-at-risk (VaR) drift oracle for EDPP.
//
// Design: docs/superpowers/specs/2026-07-21-edpp-var-oracle-design.md.
//
// The reduced/joint EDPP rule prices the backlog externality of a placement in WORK
// (microseconds). This unit re-prices that externality in VALUE — the marginal goodput
// destroyed among the co-residents already running on the candidate decode instance (and,
// for a disaggregated placement, the prefill-pool co-residents delayed by the request's
// remote prefill). The rule then compares VaR_local − VaR_disagg (the value-currency
// externality) against the unchanged z-weighted TTFT/ITL/transfer self terms.
//
// It is a DIAGNOSTIC ORACLE (UPPER BOUND): the co-resident remaining-step counts it reads
// (varReTiming's baseline / the completion model) come from the un-censored oracle
// TrueRemaining (INV-9 violation, gated behind the same admissionDetailOracle switch as the
// other EDPP oracles). Everything else it reads — arrival, realized first-token, class,
// SLO targets — is deployable/input-derived.
//
// All functions here are pure (no hidden state): identical inputs ⇒ identical float output
// (INV-6). The decider (sim/edpp.go) assembles the inputs and calls varReducedLHS / the
// per-candidate joint helper.

// varKernel selects the g(·) scoring kernel (design §2). All three are exercised by the
// experiment: A (flip) is the hyperparameter-free ceiling; B (util) makes the predicted
// "saturation ⇒ neglect" trap measurable; C (hazard) is the smoothed deployable-target shape.
type varKernel int

const (
	varKernelFlip   varKernel = iota // A: binary composite-good flip count (true→false)
	varKernelUtil                    // B: saturating slack-utility drop
	varKernelHazard                  // C: deadline-slack hazard weight × completion delay
)

// parseVarKernel maps the CLI/config string to a kernel. ok=false for an unknown value
// (the config boundary turns this into a panic, R3).
func parseVarKernel(s string) (varKernel, bool) {
	switch s {
	case "flip":
		return varKernelFlip, true
	case "util":
		return varKernelUtil, true
	case "hazard":
		return varKernelHazard, true
	}
	return 0, false
}

// varSLO bundles the per-co-resident SLO thresholds g() evaluates against (all µs). Resolved
// per co-resident CLASS (not the deciding request's class), because a co-resident meeting or
// missing its own SLO is what defines the goodput destroyed. A zero threshold disables that
// dimension's conjunct in g() (matching cluster.SLOAttainmentMultiDim, which skips a
// dimension whose target is 0).
type varSLO struct {
	tauTTFTUs float64
	tauITLUs  float64
	tauE2EUs  float64
}

// varDecodeCoResident is one decode co-resident's VaR inputs. rem is the oracle remaining
// decode steps (un-censored TrueRemaining); rem < 0 marks a censored/unknown co-resident,
// which contributes zero VaR (skipped). arrivalUs/firstTokenUs/ttftSet are deployable
// (input-derived) and fix the realized TTFT.
type varDecodeCoResident struct {
	rem          int64
	arrivalUs    int64
	firstTokenUs int64
	ttftSet      bool
	slo          varSLO
}

// varPrefillCoResident is one prefill-pool co-resident (disaggregated placement only).
// remPrefillTokens is its remaining prompt tokens — deployable (known input length), never
// oracle. Its VaR flip is TTFT-side (first-token pushed by the deciding request's prefill).
type varPrefillCoResident struct {
	remPrefillTokens int64
	arrivalUs        int64
	slo              varSLO
}

// varReTiming holds the batch-level per-iteration decode times the completion model needs
// (design §2, "full B+1 re-timing"). Every co-resident in one decode batch shares one
// per-iteration time, so these are batch-level, not per-co-resident. All µs.
//
//   - tIter0:       current batch B per-iter time, tIterDecode(B, kv, sPf) — the baseline.
//   - tIterOverlap: local placement, while the deciding request R prefills co-scheduled on the
//     decode batch (chunked prefill adds `chunk` resident prefill tokens):
//     tIterDecode(B, kv, sPf+chunk) = tIter0 + C_pf·chunk = tIter0 + δ_pf-chunk.
//   - tIterAfter:   after R joins the decode batch, tIterDecode(B+1, kv+Δkv_R, sPf), where
//     Δkv_R = R's resident context tokens (its full input length; input-only, oracle-safe).
//     "Full re-timing" = recompute tIterDecode with B+1 and kv+Δkv_R (not a marginal add).
type varReTiming struct {
	tIter0       float64
	tIterOverlap float64
	tIterAfter   float64
	// cAttn and chunk parameterize the causal prefill-attention added to the overlap
	// window in cLocal: each of R's co-scheduled chunks attends to its causal prefix.
	cAttn float64
	chunk float64
}

// cBase is a decode co-resident's projected completion with rem steps left at the current
// (pre-R) batch per-iter time. now is the decision instant (µs).
func (rt varReTiming) cBase(nowUs float64, rem int64) float64 {
	return nowUs + float64(rem)*rt.tIter0
}

// cLocal is the co-resident's completion under LOCAL placement: its first min(nChunks, rem)
// steps run at the prefill-overlap per-iter time (R prefilling co-scheduled), the remainder
// at the B+1 re-timed per-iter time (R decoding alongside).
func (rt varReTiming) cLocal(nowUs float64, rem int64, nChunks float64) float64 {
	overlap := math.Min(nChunks, float64(rem))
	// Causal prefill attention over R's co-scheduled chunks j=0..overlap-1, each charged
	// against causal prefix j·chunk + chunk/2 (start prefix 0; matches prefix_length:0
	// workloads). Σ = c_attn·chunk²·overlap²/2.
	attn := rt.cAttn * rt.chunk * rt.chunk * overlap * overlap / 2.0
	return nowUs + overlap*rt.tIterOverlap + attn + (float64(rem)-overlap)*rt.tIterAfter
}

// cDisagg is the co-resident's completion under DISAGG placement: R's prefill runs remotely,
// so the decode instance is undisturbed for arrivalSteps iterations (R's remote prefill + KV
// transfer window, in units of tIter0); only the tail max(rem−arrivalSteps, 0) steps run at
// the B+1 re-timed per-iter time. No prefill-overlap inflation on this instance.
func (rt varReTiming) cDisagg(nowUs float64, rem int64, arrivalSteps float64) float64 {
	pre := math.Min(arrivalSteps, float64(rem))
	return nowUs + pre*rt.tIter0 + (float64(rem)-pre)*rt.tIterAfter
}

// ttftMet reports whether a decode co-resident met its TTFT SLO. TTFT is realized (a
// decoding co-resident has already produced its first token) and therefore fixed under this
// placement decision. A co-resident that has NOT produced a first token (ttftSet false) or
// that missed its TTFT contributes zero to the flip/util VaR (g stays 0 before and after).
func (cr varDecodeCoResident) ttftMet() bool {
	if !cr.ttftSet {
		return false
	}
	if cr.slo.tauTTFTUs <= 0 {
		return true // no TTFT target configured ⇒ trivially met
	}
	return float64(cr.firstTokenUs-cr.arrivalUs) <= cr.slo.tauTTFTUs
}

// gDecodeFlip is kernel A's composite-good indicator for a decode co-resident completing at
// cUs (µs): 1 iff TTFT met (fixed) ∧ mean-ITL met ∧ E2E ≤ deadline, else 0. The mean ITL over
// the remaining steps is (cUs − now)/rem, the average per-iter time the co-resident would
// experience under the placement (baseline uses tIter0, which meets τ_itl iff the batch was
// already SLO-feasible). deadline = arrival + τ_e2e.
func gDecodeFlip(cr varDecodeCoResident, cUs, nowUs float64) float64 {
	if !cr.ttftMet() {
		return 0
	}
	if cr.rem <= 0 {
		return 1 // no remaining steps: ITL/E2E already realized; treat as good (no flip)
	}
	if cr.slo.tauE2EUs > 0 && cUs > float64(cr.arrivalUs)+cr.slo.tauE2EUs {
		return 0
	}
	if cr.slo.tauITLUs > 0 {
		meanITL := (cUs - nowUs) / float64(cr.rem)
		if meanITL > cr.slo.tauITLUs {
			return 0
		}
	}
	return 1
}

// gDecodeUtil is kernel B's saturating slack utility for a decode co-resident completing at
// cUs: σ((deadline − cUs)/scale). It saturates to ~1 with comfortable E2E slack and decays to
// ~0 past the deadline. scale is the E2E deadline budget's τ_ttft (a natural latency unit); a
// doomed co-resident sits deep in the σ→0 flat region, so its marginal utility drop is ~0 —
// the design's predicted "saturation ⇒ neglect" trap, which this kernel exists to MEASURE.
func gDecodeUtil(cr varDecodeCoResident, cUs float64) float64 {
	deadline := float64(cr.arrivalUs) + cr.slo.tauE2EUs
	scale := cr.slo.tauTTFTUs
	if scale <= 0 {
		scale = 1
	}
	return 1.0 / (1.0 + math.Exp(-(deadline-cUs)/scale))
}

// hazardWeight is kernel C's deadline-slack hazard for a decode co-resident whose BASELINE
// completion is cBaseUs: a heavy-tailed (Cauchy-like) bump 1/(1+x²) peaking at slack 0 (right
// at the E2E deadline) and decaying GENTLY (polynomially, not Gaussian-fast) on both sides, so
// even a deeply doomed co-resident (large negative slack) keeps a small NONZERO weight —
// avoiding kernel B's hard-zero neglect by construction (design §2 C). band is the τ_ttft (an
// SLO-derived latency scale). The kernel C VaR contribution is hazardWeight · Δ (the completion
// delay caused by the placement).
func hazardWeight(cr varDecodeCoResident, cBaseUs float64) float64 {
	slack := float64(cr.arrivalUs) + cr.slo.tauE2EUs - cBaseUs
	band := cr.slo.tauTTFTUs
	if band <= 0 {
		band = 1
	}
	x := slack / band
	return 1.0 / (1.0 + x*x)
}

// varDecodeLocal sums, over the decode co-residents, the value-at-risk of the LOCAL
// placement: Σ_j contribution(j) where the co-resident is delayed from cBase to cLocal.
// For flip/util the contribution is g(before) − g(after); for hazard it is
// hazardWeight(slack) · (cLocal − cBase). Censored co-residents (rem < 0) are skipped.
func varDecodeLocal(nowUs float64, crs []varDecodeCoResident, rt varReTiming, nChunks float64, kernel varKernel) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(nowUs, cr.rem)
		cp := rt.cLocal(nowUs, cr.rem, nChunks)
		sum += varDecodeContribution(cr, cb, cp, nowUs, kernel)
	}
	return sum
}

// varDecodeDisagg is the DISAGG mirror of varDecodeLocal: the co-resident is delayed from
// cBase to cDisagg (only its tail steps re-timed, R arriving after arrivalSteps iterations).
func varDecodeDisagg(nowUs float64, crs []varDecodeCoResident, rt varReTiming, arrivalSteps float64, kernel varKernel) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(nowUs, cr.rem)
		cp := rt.cDisagg(nowUs, cr.rem, arrivalSteps)
		sum += varDecodeContribution(cr, cb, cp, nowUs, kernel)
	}
	return sum
}

// varDecodeContribution is one decode co-resident's VaR contribution under the active kernel,
// given its baseline completion cb and placed completion cp (both µs). Shared by the local and
// disagg sums so the two paths use byte-identical arithmetic (INV-6).
func varDecodeContribution(cr varDecodeCoResident, cb, cp, nowUs float64, kernel varKernel) float64 {
	switch kernel {
	case varKernelUtil:
		return gDecodeUtil(cr, cb) - gDecodeUtil(cr, cp)
	case varKernelHazard:
		return hazardWeight(cr, cb) * (cp - cb)
	default: // varKernelFlip
		return gDecodeFlip(cr, cb, nowUs) - gDecodeFlip(cr, cp, nowUs)
	}
}

// varPrefillDisagg sums the value-at-risk imposed on the prefill-pool co-residents by the
// deciding request's remote prefill (disaggregated placement only). Each prefill co-resident's
// first-token completion is pushed by rPrefillUs (the deciding request's added prefill duration
// on the pool). Its flip is TTFT-side: deadline = arrival + τ_ttft. tIterP is the prefill
// pool's per-iter time and chunkP its per-step token budget, used to project the co-resident's
// own remaining first-token time. This is a first-order contention model (design §2/§9): the
// decode-side asymmetry is the dominant mechanism; the asymmetry-law test isolates it by
// requiring an idle prefill pool (no prefill co-residents ⇒ this term is 0).
func varPrefillDisagg(nowUs float64, ks []varPrefillCoResident, tIterP, chunkP, rPrefillUs float64, kernel varKernel) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(k.remPrefillTokens) / chunkP)
		cb := nowUs + remIters*tIterP
		cp := cb + rPrefillUs
		sum += varPrefillTTFTContribution(k, cb, cp, kernel)
	}
	return sum
}

// varPrefillTTFTContribution scores one prefill co-resident's first-token value-at-risk under the
// active kernel, given its baseline first-token completion cb and placed completion cp (both µs).
// The flip is TTFT-side: deadline = arrival + τ_ttft. Shared by the disagg prefill-pool term
// (varPrefillDisagg) and the collocated decode-instance term (varCollocPrefill*) so both score
// first-token risk with byte-identical arithmetic (INV-6).
func varPrefillTTFTContribution(k varPrefillCoResident, cb, cp float64, kernel varKernel) float64 {
	deadline := float64(k.arrivalUs) + k.slo.tauTTFTUs
	switch kernel {
	case varKernelUtil:
		scale := k.slo.tauTTFTUs
		if scale <= 0 {
			scale = 1
		}
		return sigmoid((deadline-cb)/scale) - sigmoid((deadline-cp)/scale)
	case varKernelHazard:
		band := k.slo.tauTTFTUs
		if band <= 0 {
			band = 1
		}
		x := (deadline - cb) / band
		return (1.0 / (1.0 + x*x)) * (cp - cb)
	default: // varKernelFlip
		return b2f(k.slo.tauTTFTUs <= 0 || cb <= deadline) - b2f(k.slo.tauTTFTUs <= 0 || cp <= deadline)
	}
}

// varCollocPrefillLocal sums the first-token value-at-risk imposed on the DECODE instance's
// collocated prefill occupants by a LOCAL placement. Such an occupant (placed here by a prior
// collocate decision) has not produced its first token yet; RunningDecodeState skips it, so the
// decode-side VaR terms miss it. It needs remIters = ⌈remPrefillTokens / chunk⌉ more decode-batch
// iterations to reach its first token, and R's co-scheduled prefill then B+1 join re-times those
// iterations exactly like a decode co-resident's remaining steps — so the same cBase→cLocal
// completion model applies. Deployable: remPrefillTokens is known input length (INV-9-safe).
func varCollocPrefillLocal(nowUs float64, ks []varPrefillCoResident, rt varReTiming, chunk, nChunks float64, kernel varKernel) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		cb := rt.cBase(nowUs, remIters)
		cp := rt.cLocal(nowUs, remIters, nChunks)
		sum += varPrefillTTFTContribution(k, cb, cp, kernel)
	}
	return sum
}

// varCollocPrefillDisagg is the DISAGG mirror of varCollocPrefillLocal: R prefills remotely, so
// the decode instance is undisturbed for arrivalSteps iterations. An occupant that reaches its
// first token within that window (remIters ≤ arrivalSteps) sees cDisagg = cBase and contributes
// zero — disagg does not delay it. Only the tail beyond arrivalSteps is re-timed.
func varCollocPrefillDisagg(nowUs float64, ks []varPrefillCoResident, rt varReTiming, chunk, arrivalSteps float64, kernel varKernel) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		cb := rt.cBase(nowUs, remIters)
		cp := rt.cDisagg(nowUs, remIters, arrivalSteps)
		sum += varPrefillTTFTContribution(k, cb, cp, kernel)
	}
	return sum
}

// --- decider-bound assembly (reduced + joint) -------------------------------------------

// varSLOFor resolves a co-resident class's SLO thresholds (µs) from the decider config.
func (d *EDPPDecider) varSLOFor(class string) varSLO {
	tt, itl := d.targetsFor(class)
	return varSLO{
		tauTTFTUs: float64(tt),
		tauITLUs:  float64(itl),
		tauE2EUs:  float64(d.e2eFor(class)),
	}
}

// varDecodeInputs converts a snapshot's decode running-request slice into VaR co-resident
// inputs. Each co-resident's remaining decode steps come from one of two sources:
//
//   - ORACLE (default): the un-censored true remaining, TrueRemaining (gated behind the
//     admissionDetailOracle switch). This is a diagnostic upper bound; when the oracle is off
//     TrueRemaining is -1 and the co-resident is skipped (VaR → 0, rule falls to its rhs sign).
//   - DEPLOYABLE (VarDeployable): the censored per-class output-length estimate,
//     max(N̂_out(class) − StepsDone, 1), which reads no hidden output length (INV-9-safe). This
//     mirrors decodeRemStepsEst's per-request censored form and turns the ceiling into a policy.
func (d *EDPPDecider) varDecodeInputs(running []RunningReqState) []varDecodeCoResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]varDecodeCoResident, 0, len(running))
	for _, r := range running {
		rem := r.TrueRemaining
		if d.varDeployable {
			// Censored estimate: a co-resident that has produced StepsDone tokens has output
			// length ≥ StepsDone, so N̂_out is floored by StepsDone before subtracting; the
			// remaining is then floored at 1 (it is still decoding).
			nHat := d.nHatFor(r.SLOClass).mean()
			rem = int64(math.Max(math.Max(nHat, float64(r.StepsDone))-float64(r.StepsDone), 1))
		}
		out = append(out, varDecodeCoResident{
			rem:          rem,
			arrivalUs:    r.ArrivalUs,
			firstTokenUs: r.FirstTokenUs,
			ttftSet:      r.TTFTSet,
			slo:          d.varSLOFor(r.SLOClass),
		})
	}
	return out
}

// varPrefillInputs converts a snapshot's prefill running-request slice into VaR prefill
// co-resident inputs. remPrefillTokens (TrueRemaining on the prefill slice) is deployable —
// remaining prompt tokens, known input, never oracle-gated (see Simulator.RunningPrefillState).
func (d *EDPPDecider) varPrefillInputs(running []RunningReqState) []varPrefillCoResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]varPrefillCoResident, 0, len(running))
	for _, r := range running {
		out = append(out, varPrefillCoResident{
			remPrefillTokens: r.TrueRemaining,
			arrivalUs:        r.ArrivalUs,
			slo:              d.varSLOFor(r.SLOClass),
		})
	}
	return out
}

// varReTimingFor builds the batch-level per-iter decode times for the completion model under
// decode node physics thetaD at batch state (bDec, kv, sPf). chunk is the deciding request's
// prefill token budget (for the local overlap term); Δkv_R = R's full input length (its
// resident context once it joins the decode batch; input-only, oracle-safe).
func (d *EDPPDecider) varReTimingFor(req *Request, thetaD EDPPCoeffs, bDec int, kv, sPf int64, chunk int) varReTiming {
	dkv := int64(len(req.InputTokens))
	return varReTiming{
		tIter0:       thetaD.tIterDecode(bDec, kv, sPf),
		tIterOverlap: thetaD.tIterDecode(bDec, kv, sPf+int64(chunk)),
		tIterAfter:   thetaD.tIterDecode(bDec+1, kv+dkv, sPf),
		cAttn:        thetaD.CAttn,
		chunk:        float64(chunk),
	}
}

// varReducedLHS computes the reduced-rule value-currency externality lhs_var =
// VaR_local − VaR_disagg for the deciding request R on its selected decode instance. It
// mirrors the reduced Decide's already-computed operands (decode node θ_i, batch state, chunk,
// nChunks, ttftP, prefill-pool occupancy) so no physics is recomputed differently (INV-6).
//
//   - VaR_local:  goodput destroyed among the decode co-residents delayed by R's prefill-on-
//     decode then B+1 join.
//   - VaR_disagg: goodput destroyed among the decode co-residents (only their tail steps
//     re-timed, R arriving after ⌈ttftP/tIter0⌉ iterations) PLUS the prefill-pool co-residents
//     whose first token is pushed by R's remote prefill.
func (d *EDPPDecider) varReducedLHS(
	req *Request, nowUs float64,
	decSnap RoutingSnapshot, prefillSnaps []RoutingSnapshot,
	thetaD EDPPCoeffs, bDec int, kv, sPf int64,
	chunk int, nChunks, ttftP float64, sPfPrefill int64,
) float64 {
	rt := d.varReTimingFor(req, thetaD, bDec, kv, sPf, chunk)
	decode := d.varDecodeInputs(decSnap.RunningDecode)
	kernel := d.varMetric

	varLocal := varDecodeLocal(nowUs, decode, rt, nChunks, kernel)

	arrivalSteps := math.Ceil(ttftP / math.Max(rt.tIter0, 1))
	varDisagg := varDecodeDisagg(nowUs, decode, rt, arrivalSteps, kernel)

	// Collocated prefill occupants on the DECODE instance (placed by a prior collocate decision,
	// still pre-first-token so RunningDecode skips them): local co-schedules R's prefill and
	// delays their first token; disagg leaves those finishing before R arrives untouched. Same
	// re-timing as decode co-residents, TTFT-side flip. Deployable (remaining prompt tokens).
	if d.varCollocPrefill && len(decSnap.RunningPrefill) > 0 {
		colloc := d.varPrefillInputs(decSnap.RunningPrefill)
		varLocal += varCollocPrefillLocal(nowUs, colloc, rt, float64(chunk), nChunks, kernel)
		varDisagg += varCollocPrefillDisagg(nowUs, colloc, rt, float64(chunk), arrivalSteps, kernel)
	}

	// Prefill-side externality of disagg: R's remote prefill delays the prefill-pool occupants.
	// R's added prefill duration on the pool = nChunks·tIterPrefill + Wp (compute injected; the
	// transfer window is not pool contention). Aggregate occupants across all prefill snapshots.
	if len(prefillSnaps) > 0 {
		tIterP := d.coeffs.tIterPrefill(sPfPrefill)
		ap := d.apForInstance(req, "")
		rPrefillUs := nChunks*tIterP + d.coeffs.Wp(maxInt(ap, 0), len(req.InputTokens))
		chunkP := float64(chunk)
		var prefill []varPrefillCoResident
		for _, ps := range prefillSnaps {
			prefill = append(prefill, d.varPrefillInputs(ps.RunningPrefill)...)
		}
		varDisagg += varPrefillDisagg(nowUs, prefill, tIterP, chunkP, rPrefillUs, kernel)
	}

	return varLocal - varDisagg
}

// varJointCandidateExternality computes the value-currency externality term for ONE joint
// candidate, replacing the work-currency backlog contribution (jDecodeBacklog + qd·(Wp/W*_d)
// locally, or + qp·(Wp/W*_p) for disagg). For a local candidate (ps == nil) it is VaR_local
// on ds's decode co-residents; for a disagg candidate it is VaR_disagg on ds's decode
// co-residents plus the prefill-side VaR on *ps's occupants. The decode node physics use ds's
// θ_i; the prefill physics use *ps's θ_i. Mirrors jointCandidateCost's operands (INV-6).
func (d *EDPPDecider) varJointCandidateExternality(
	req *Request, nowUs float64, ds RoutingSnapshot, ps *RoutingSnapshot,
) float64 {
	thetaD := d.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	decode := d.varDecodeInputs(ds.RunningDecode)
	kernel := d.varMetric

	if ps == nil {
		apLoc := d.apForInstance(req, ds.ID)
		chunkLoc := apLoc
		if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunkLoc {
			chunkLoc = d.cfg.ChunkTokens
		}
		rt := d.varReTimingFor(req, thetaD, bDec, kv, sPfD, chunkLoc)
		nChunksLoc, _ := d.chunkTerms(thetaD, apLoc)
		v := varDecodeLocal(nowUs, decode, rt, nChunksLoc, kernel)
		if d.varCollocPrefill && len(ds.RunningPrefill) > 0 {
			colloc := d.varPrefillInputs(ds.RunningPrefill)
			v += varCollocPrefillLocal(nowUs, colloc, rt, float64(chunkLoc), nChunksLoc, kernel)
		}
		return v
	}

	// Disagg: decode co-residents on ds (tail re-timed after R arrives from the prefill node),
	// plus prefill-pool co-residents on *ps delayed by R's remote prefill.
	thetaP := d.coeffsFor(ps.GPUType)
	apP := d.apForInstance(req, ps.ID)
	chunkP := apP
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunkP {
		chunkP = d.cfg.ChunkTokens
	}
	// Δkv_R / decode re-timing still use ds's θ (decode happens on ds in both placements).
	rt := d.varReTimingFor(req, thetaD, bDec, kv, sPfD, chunkP)
	nChunksP, _ := d.chunkTerms(thetaP, apP)
	sPfP := ps.ResidentPrefillTokens
	tIterP := thetaP.tIterPrefill(sPfP)
	cXferUs := d.cXferUsFor(req)
	tAdmP := 0.0 // conservative: R arrives after prefill compute + transfer (admission folded into ttft below)
	ttftP := tAdmP + nChunksP*tIterP + thetaP.Wp(maxInt(apP, 0), len(req.InputTokens)) + cXferUs
	arrivalSteps := math.Ceil(ttftP / math.Max(rt.tIter0, 1))
	v := varDecodeDisagg(nowUs, decode, rt, arrivalSteps, kernel)

	// Collocated prefill occupants on the decode node ds are undisturbed until R arrives from the
	// pool; only those still prefilling past arrivalSteps have their first token delayed.
	if d.varCollocPrefill && len(ds.RunningPrefill) > 0 {
		colloc := d.varPrefillInputs(ds.RunningPrefill)
		v += varCollocPrefillDisagg(nowUs, colloc, rt, float64(chunkP), arrivalSteps, kernel)
	}

	rPrefillUs := nChunksP*tIterP + thetaP.Wp(maxInt(apP, 0), len(req.InputTokens))
	prefill := d.varPrefillInputs(ps.RunningPrefill)
	v += varPrefillDisagg(nowUs, prefill, tIterP, float64(chunkP), rPrefillUs, kernel)
	return v
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
