package sim

import "math"

// Kairos baseline — load-aware prefill deflection for disaggregated LLM serving.
//
// Faithful reproduction of the routing rule in "Towards Load-Aware Prefill Deflection for
// Disaggregated LLM Serving" (arXiv:2607.02043), implemented as an EDPP rule so it reuses this
// package's trained-physics coefficients and routing snapshots. It exists as a STATE-OF-THE-ART
// BASELINE for the study — it is not our contribution.
//
// The rule, per the paper:
//
//  1. Estimate the TTFT the request would see on the PREFILL node: an analytical FIFO model,
//     queue wait + the request's own chunked prefill execution.
//  2. For each DECODE node, search for the largest chunk schedule that keeps the node's in-flight
//     decodes within their time-between-tokens (TBT) SLO — a HARD CONSTRAINT, greedily taking the
//     largest safe chunk per step — and compute the TTFT of prefilling there ("deflection"),
//     which also avoids the inter-node KV transfer.
//  3. Deflect to the fastest feasible decode node when that beats the prefill path; else route to
//     the prefill node.
//
// Differences from our rule, stated plainly so the comparison is honest: Kairos protects
// co-residents with a per-step TBT *constraint* (it never models their remaining decode steps or
// end-to-end completion), fixes the prefill node rather than enumerating (decode, prefill) pairs,
// and assumes homogeneous hardware. Its published estimator is a FIFO approximation plus a
// regressed step-latency model (~10% MAPE); here it is evaluated against the SAME trained-physics
// coefficients our rule uses, so the comparison isolates the POLICY rather than estimator quality.

// kairosStepPrefill is the prefill-node step time for a chunk of chi tokens attending over a
// resident context of k tokens: α_p + c_pf·chi + c_attn·chi·(k + chi/2). This is the per-step
// charge the work model of sim/edpp_coeffs.go integrates into Wp.
func kairosStepPrefill(c EDPPCoeffs, chi, k float64) float64 {
	if chi <= 0 {
		return c.AlphaP
	}
	return c.AlphaP + c.CPf*chi + c.CAttn*chi*(k+chi/2)
}

// kairosMaxSafeChunk returns the largest prefill chunk (tokens) that can be co-scheduled onto a
// decode step whose base (decode-only) time is base, with the chunk attending over ctx tokens,
// while keeping the step within the TBT budget tbt. It solves the per-step constraint
//
//	base + c_pf·chi + c_attn·chi·(ctx + chi/2) ≤ tbt
//
// which is quadratic in chi: (c_attn/2)·chi² + (c_pf + c_attn·ctx)·chi + (base − tbt) ≤ 0.
// Returns 0 when no positive chunk fits (the node cannot host this prefill without violating TBT).
func kairosMaxSafeChunk(c EDPPCoeffs, base, ctx, tbt float64) float64 {
	slack := tbt - base
	if slack <= 0 {
		return 0 // even a bare decode step already violates the TBT budget
	}
	a := c.CAttn / 2
	b := c.CPf + c.CAttn*ctx
	if a <= 0 {
		if b <= 0 {
			return 0
		}
		return slack / b
	}
	// positive root of a·chi² + b·chi − slack = 0
	disc := b*b + 4*a*slack
	if disc <= 0 {
		return 0
	}
	return (-b + math.Sqrt(disc)) / (2 * a)
}

// kairosDeflectTTFT computes the TTFT of prefilling `tokens` uncached tokens on a decode node
// carrying a decode batch of bDec requests holding kv resident tokens, under the TBT budget tbt.
// It greedily takes the largest TBT-safe chunk each step (the paper's "largest chunk schedule"),
// capped by the engine's per-step token budget chunkCap. Returns (ttft, feasible); infeasible when
// no positive chunk fits or the schedule exceeds maxSteps.
func kairosDeflectTTFT(c EDPPCoeffs, bDec int, kv int64, tokens, tbt, chunkCap float64, maxSteps int) (float64, bool) {
	if tokens <= 0 {
		return 0, true
	}
	var elapsed, done float64
	for step := 0; step < maxSteps && done < tokens; step++ {
		// The decode batch's own step time grows as the deflected prefill's KV accumulates.
		base := c.tIterDecode(bDec, kv+int64(done), 0)
		chi := kairosMaxSafeChunk(c, base, done, tbt)
		if chi <= 0 {
			return 0, false
		}
		if chunkCap > 0 && chi > chunkCap {
			chi = chunkCap
		}
		if chi > tokens-done {
			chi = tokens - done
		}
		elapsed += base + c.CPf*chi + c.CAttn*chi*(done+chi/2)
		done += chi
	}
	if done < tokens {
		return 0, false
	}
	return elapsed, true
}

// decideKairos implements the Kairos load-aware prefill-deflection decision for one request.
// Deflection (prefill on a decode node) maps onto our LOCAL placement — prefill co-resident with
// decode, no KV transfer — with the chosen decode node returned as DecodePodOverride. Routing to
// the prefill node maps onto DISAGGREGATION.
func (d *EDPPDecider) decideKairos(req *Request, state *RouterState) DisaggregationDecision {
	keepLocal := DisaggregationDecision{Disaggregate: false}
	if len(req.InputTokens) == 0 {
		return keepLocal
	}
	_, tauITLUs := d.targetsFor(req.SLOClass)
	tbt := d.kairosBeta * float64(tauITLUs)
	chunkCap := float64(d.cfg.ChunkTokens) // engine per-step token budget; 0 ⇒ uncapped
	const maxSteps = 4096

	// --- prefill-node path: FIFO queue wait + own chunked execution + KV transfer ---
	ttftPrefill := math.Inf(1)
	prefillID := ""
	var prefillSnaps []RoutingSnapshot
	if d.prefillSnapshots != nil {
		prefillSnaps = sortedSnapshotsByID(d.prefillSnapshots())
	}
	if len(prefillSnaps) > 0 {
		ps := prefillSnaps[0]
		thetaP := d.coeffsFor(ps.GPUType)
		ap := float64(maxInt(d.apForInstance(req, ps.ID), 0))
		chi := chunkCap
		if chi <= 0 || chi > ap {
			chi = ap
		}
		if chi > 0 {
			// Outstanding prefill tokens ahead of this request. The snapshot carries resident
			// prefill tokens directly; queued requests are approximated by QueueDepth × this
			// request's prompt length (the paper sums their true lengths, which the snapshot
			// does not expose).
			sumL := float64(ps.ResidentPrefillTokens) + float64(ps.QueueDepth)*float64(len(req.InputTokens))
			queueWait := 0.0
			if sumL > 0 {
				// The paper writes the queue wait as (Σℓ/χ)·T_step(χ, Σℓ/2). Read literally, the
				// attention context Σℓ/2 is the whole queue's tokens, which over-charges a chunk
				// enormously once the queue is deep (a chunk attends over its OWN request's context,
				// not the queue's). We cap the context at one request's prompt length — the
				// physically sensible reading, and the one most GENEROUS to this baseline, since it
				// lowers the prefill-path estimate and so makes deflection less automatic.
				ctxQ := math.Min(sumL/2, float64(len(req.InputTokens)))
				queueWait = (sumL / chi) * kairosStepPrefill(thetaP, chi, ctxQ)
			}
			exec := 0.0
			steps := int(math.Ceil(ap / chi))
			for i := 0; i < steps; i++ {
				exec += kairosStepPrefill(thetaP, chi, chi*float64(i))
			}
			ttftPrefill = queueWait + exec + d.cXferUsFor(req)
			prefillID = ps.ID
		}
	}

	// --- deflection candidates: each decode node, largest TBT-safe chunk schedule ---
	// FAIRNESS: the prefill path above carries the paper's FIFO queue wait, so the deflect path
	// must carry its own admission delay too — a deflected prefill cannot start until the decode
	// node admits it (batch slot + KV). Omitting it would make deflection look free on a saturated
	// decode node and hand the baseline a strawman loss. We give Kairos the SAME occupancy-aware
	// admission estimator our own rule consumes (deployable/censored — Kairos reads no oracle), so
	// the comparison isolates the POLICY rather than estimator quality.
	reqKVNeed := d.reqKVNeed(req)
	bestTTFT := ttftPrefill
	bestDecode := ""
	for _, ds := range sortedSnapshotsByID(stateSnapshots(state)) {
		// Paper's constraint: at most one deflected prefill in flight per decode node. A node
		// already carrying resident prefill tokens is therefore not a deflection target.
		if ds.ResidentPrefillTokens > 0 {
			continue
		}
		thetaD := d.coeffsFor(ds.GPUType)
		ap := float64(maxInt(d.apForInstance(req, ds.ID), 0))
		t, ok := kairosDeflectTTFT(thetaD, ds.BatchSize, ds.KvTokensInUse, ap, tbt, chunkCap, maxSteps)
		if !ok {
			continue
		}
		tAdm := d.tadmEstimator.EstimateTAdm(AdmissionContext{
			QWork:     func() float64 { _, qd := d.instWorkRaw(ds.ID); return qd }(),
			Mu:        thetaD.muDecode(ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens),
			BatchSize: ds.BatchSize, MaxBatchSize: int(ds.MaxBatchSize),
			FreeKVBlocks: ds.FreeKVBlocks, ReqKVNeed: reqKVNeed,
			TIter:         thetaD.tIterDecode(ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens),
			QueueDepth:    ds.QueueDepth,
			AdmissionRate: admissionRateFromSnapshot(ds), RemainingStepsEst: d.decodeRemStepsEst(ds, req.SLOClass),
			Running: censorOracleRemaining(ds.RunningDecode),
		})
		if t+tAdm < bestTTFT {
			bestTTFT = t + tAdm
			bestDecode = ds.ID
		}
	}

	if bestDecode != "" {
		// Deflect: prefill co-resident on the winning decode node (no KV transfer).
		return DisaggregationDecision{Disaggregate: false, DecodePodOverride: bestDecode}
	}
	if prefillID == "" || math.IsInf(ttftPrefill, 1) {
		return keepLocal // no prefill pool available; fall back to local
	}
	return DisaggregationDecision{Disaggregate: true, PrefillPodHint: prefillID}
}
