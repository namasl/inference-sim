# EDPP study — backlog / open questions

## The organizing distinction (use this to classify every experiment)

Two separate questions, often conflated:

- **Q1 — SHOULD this workload run disaggregated at all?** A provisioning/topology question
  (how many prefill vs decode instances; disaggregate-vs-collocate as a *deployment* choice).
  EDPP does NOT control this — it's given a fixed pool. Comparing EDPP to `never@4` answers Q1,
  NOT EDPP's quality.
- **Q2 — GIVEN a disaggregated deployment, does EDPP disaggregate the RIGHT requests?** This is the
  algorithm's actual job and the real test of correctness. Baselines must hold topology FIXED:
  `always` (disaggregate all) and `never`-within-the-same-split (disaggregate none) bracket EDPP;
  an oracle/hindsight-optimal per-request labeling is the gold standard.

**Status:** experiments so far answered Q1 for the uniform decode-bound workload (synth): don't
disaggregate — you need more decode nodes, which EDPP can't provide; EDPP ranges harmless→harmful.
**Q2 — the correctness of the per-request decision — is NOT yet determined.** That is the priority.

## TODO — Q2: evaluate per-request decision correctness (PRIORITY)

1. **Heterogeneous favorable-regime workload** (controller's proposal, 2026-06-29):
   - Type A (~85%): small prefill, substantial decode → should stay LOCAL (avoid transfer; protect ITL).
   - Type B (~15%): LARGE prefill (~16k), small decode → should DISAGGREGATE (its prefill, if collocated,
     spikes co-resident A decodes' ITL/TTFT).
   - Distinct prefix_groups for A/B (avoid single-universal-prefix pinning). Decode capacity ADEQUATE
     (load below the decode knee) so the bottleneck is prefill-interference, NOT decode-capacity.
   - SLOs: A = standard (tight ITL ~80ms, TTFT ~1s); B = batch (loose).
   - **Mechanism-check: DONE 2026-06-29 — interference is REAL, via TTFT (not ITL).** Single instance,
     never decider, A(in256/out300) alone vs A+B(in16000/out10). A's TTFT degraded 2× mean / 5.6× p99
     (exactly 12/60 A reqs spiked >100ms = one per B); A's ITL barely moved (+3% tail only — the
     c_pf·S_pf channel exists but is dilute since B is rare vs A's 300-tok decodes). Specs:
     specs/mech/{A_only,AB}.yaml; results out/diag/mech/; detail in SESSION_LOG. **DESIGN CHANGE: build
     the favorable workload around A wanting tight TTFT (the strong, EDPP-responsive lever), NOT tight
     ITL.** To exercise ITL-driven disaggregation (z_itl) instead, amplify B frequency / A-decode overlap.
   - **The balance term will matter here (unlike synth).** balance_term_d = q_d·(W_p/W*_d) ∝ W_p =
     c_pf·a_p + (c_attn/2)·a_p². Synth's tiny prompts made it negligible (~3e-4); Type-B's ~16k prompt
     makes W_p (and the balance term) large — likely the PRIMARY decision driver when decode is adequate
     so z_ttft stays off. Mine the term composition on this workload to confirm. This is also where the
     per-instance-vs-pool-aggregate backlog question (below) becomes first-order.
   - **4 arms + oracle: DONE 2026-06-29 — EDPP FAILS (externality-blind).** 2P2D, decode adequate. A TTFT
     p99: never-in-split 259ms / always 271ms / prefix-16 271ms / **edpp 256ms** / **ORACLE(disagg B only,
     via prefix-threshold=1000) 51ms, 0% viol**. EDPP disaggregates A 0% (right) but B only 2% (WRONG) →
     keeps interfering B local → A ≈ never. A 5× tail win (259→51ms) + 100% SLO is available; EDPP gets none.
     ROOT CAUSE (structural, not tuning): B's huge W_p enters only via balance_term_d = q_d·(W_p/W*_d) and
     q_d≈0 when decode adequate; B's own TTFT << its loose 5s SLO so z_ttft(batch)=0. B's loose SLO inflates
     W*_d so B's balance term is SMALLER than A's → lowering V disaggregates A before B (backwards). EDPP
     judges each request by its OWN class SLO + shared backlog; B-harms-A is an EXTERNALITY the rule can't
     express. Detail: FINDINGS "Q2" + SESSION_LOG. Artifacts: out/diag/hetero/, specs/hetero/.
   - **NEW FIX DIRECTION (the missing ingredient):** an interference/externality term — weight a request's
     prefill cost (c_pf·chunk = the δ_pf it injects into the decode batch) by the CO-RESIDENT decode pool's
     SLO pressure (the victims' z_ttft/z_itl), not the deciding request's own z. Today itl_term uses the
     deciding request's z_itl (B's=0); it should use the pool's. This is what would let EDPP disaggregate B
     to protect A. Design needed (and re-derive within the Lyapunov framework — it's a coupling term).
   - **Robustness:** re-run with amplified B (frequency/size) — structural finding is robust, magnitude
     (2.7% interference) is one operating point.
2. **Oracle baseline: DONE for this setup** — prefix-threshold with threshold between A's and B's uncached
   length (=1000) is a cheap per-class oracle (disagg B only, A local). Generalize if workloads get richer.
3. **Forced-disaggregation correctness: ANSWERED for this case** — EDPP's recall on the should-disaggregate
   subset (B) is ~2% (near-zero). The externality term above is the prerequisite to improve it.

## TODO — mine instrumentation we already produced but haven't analyzed

We have `--edpp-decision-trace` and `--routing-decision-trace` but have only looked at aggregates.
4. **lhs/rhs term decomposition ACROSS the load sweep.** The loadknee runs (rates 0.5–3.0) did NOT save
   `--edpp-decision-trace` — re-run them with it. Then plot, per rate: mean lhs, rhs, and EACH term
   (balance_term_d/p, transfer_term, ttft_term, itl_term, z_ttft, z_itl, qd/qp) — show how the decision
   composition shifts as load rises (e.g. when does z_ttft/z_itl turn on; which term flips lhs vs rhs).
5. **Predicted-vs-observed across load** (not just rate 2.0): ttft_d/ttft_p vs realized TTFT; itl_d/itl_p
   vs realized ITL; the HOL-blind ttft_d under-prediction (174× at rate 2.0) — how does the error scale
   with load? Is there a load where ttft_d is accurate?
6. **routing-decision-trace mining** (`dbg_edpp_2P2D_routetrace.csv` exists, never analyzed): per-decision
   candidate QUEUE LENGTHS, queue/scheduling latencies, KV util, inflight (incl reserved-pending), and
   per-scorer scores. Use to see what the router actually saw at each pick vs what was true.
7. **Backlog accuracy:** does EDPP's qd/qp (waiting-only backlog) track real waiting work? Cross-check
   against routing-trace queue depths. (The waiting-only vs running-occupancy gap is the root of the
   ttft_d HOL-blindness — quantify it directly.)

## TODO — Q1 loose ends (lower priority; mechanism already understood)
8. **RAG (prefill-bound): DONE 2026-06-29 (real inference-perf batch-summarization).** agg-4 (NO disagg)
   WINS at every rate (vector-qa SLO-viol 0/9/53% vs edpp 11/36/90% vs prefix-thresh 15/76/98%). GPU-matched
   dedicated-role splits steal flexible prefill capacity; EDPP OVER-disaggregates the short vector-qa
   (60-76%) because ttft_p under-predicts prefill-pool congestion (clogging doc-read prefills are RUNNING,
   not WAITING — same blindness as ttft_d, prefill side). Specs specs/rag/, results out/diag/rag/. So:
   no decode-node-count inversion — instead the no-disaggregation baseline (agg) wins for prefill-bound too.
   **SYNTHESIS: across decode-bound/mixed/prefill-bound, EDPP never beats no-disaggregation at equal HW.**
   PREDICTOR FIX (TODO 10/12) now has BOTH sides: make ttft_d AND ttft_p running-occupancy-aware.
   Follow-ups: multi-seed; non-GPU-matched (more total GPUs) framing where disagg's real value lives.
9. Pin the synth knee more precisely (one point at rate ~0.75) — cosmetic.

## TODO — apply the responsive-update fix to z_itl (NEXT, ready to pick up)
10b. **`z_itl` has the SAME completion-lag flaw `z_ttft` just had — fix it the same way.**
    `z_itl` is bumped only at completion (`OnComplete`, from realized mean ITL), so it reacts late to
    ITL trouble exactly as `z_ttft` did. ITL is a per-token rate (not a wait), so the "observed
    elapsed-wait lower bound" trick does not transfer directly — but a running ITL estimate is
    observable mid-decode (per-token timestamps already exist; `req.ITL` accumulates during the run).
    DESIGN QUESTION first: what is the observable, certain signal for an in-flight ITL miss? Candidate:
    once a decoding request has produced ≥k tokens with running-mean ITL > τ_itl, credit the overage
    incrementally (true-up at completion), mirroring the `z_ttft` credit/true-up structure
    (`sim/edpp.go` `creditAwaiting`/`OnFirstToken`). VALIDATE: re-run the τ_itl 150→50ms ITL-binding
    case (REPRO.md) — expect `z_itl` to activate earlier. NOTE: on uniform decode-bound synth ITL is
    floored by decode capacity (disaggregation moves only prefill), so this helps RESPONSIVENESS, not
    the capacity limit — the real payoff is on the heterogeneous workload (#1).

## DONE — responsive z_ttft (the fix that replaced the wrong "ttft_d accuracy" direction)
10. **(SUPERSEDED then DONE differently.)** The original plan — "make `ttft_d` running-occupancy-aware"
    — was DISPROVED by mining EDPP's decision trace: `ttft_d` enters the rule only as
    `z_ttft·(TTFT_P−TTFT_D)`, so where `z_ttft=0` (81% of decisions) its accuracy is multiplied by zero,
    and where `z_ttft>0` (19%) the term already saturates the decision. `ttft_d`'s accuracy does not move
    the decision. The REAL lever was `z_ttft`'s responsiveness: it updated only at completion, so the
    TTFT-miss signal arrived ~100s+ late (z_ttft 0 for 81% of decisions, first positive 81% through run).
    SHIPPED (feat/edpp-pd-disagg, 2026-06-29): credit `z_ttft` continuously from each waiting request's
    observed elapsed wait (a certain lower bound on its miss), trued up at first token (new
    `OnFirstToken` sim hook + cluster wiring) or completion fallback. Same total contribution per
    request, credited earlier (faithful to the virtual-queue construction). Dropped-after-waiting keeps
    credit. Deterministic sweep (INV-6). RESULT: edpp@2P2D rate2.0 75%→98% TTFT-SLO, p99 518s→180s,
    disagg 52%→81%, z_ttft first-positive 81%→16% through run. EDPP now tracks `always` within a fixed
    topology (still can't beat never@4 — the Q1 provisioning limit). Design:
    `docs/superpowers/specs/2026-06-29-edpp-responsive-z-ttft-design.md`; archived flaw-driven numbers:
    `out/diag/ARCHIVE_lagged-z-ttft-artifact.md`.
12. **Per-instance vs pool-aggregate backlog coherence in TTFT_D / balance (raised 2026-06-29).**
    `Decide` runs after the decode pod D* is picked (state.SelectedInstance) but before P* is picked.
    It uses D*'s own batch/KV for μ_dec/T_iter (per-instance ✓), but `Q_d`/`Q_p` (balance terms) and the
    prefill side are POOL-AGGREGATE. Coherence bug: `TTFT_D = Q_d/μ_dec + …` divides the AGGREGATE
    waiting backlog (all decode pods) by D*'s drain rate — over-counting queue from pods this request
    won't sit on. Negligible on synth (balance term ≈ 0), but becomes first-order on the heterogeneous
    workload (#1) where the balance term drives decisions. FIX DIRECTION: make the backlog terms key off
    D*'s own waiting backlog. Pairs with the deferred running-occupancy TTFT_D work — both make the
    local-TTFT estimate reflect the SPECIFIC chosen pod. Measure on #1 before fixing (measure-first).

11. **Re-measure under the fix — DONE for synth.** Re-measured: 1P3D/2P2D/3P1D rate-2.0 cells (SUMMARY),
    the load knee rates 0.5–3.0 (the old "rate-1.0 cliff" is GONE — EDPP healthy through 1.5), and the
    ITL τ_itl-50ms case (disagg 58→90% but ITL stays 72ms = decode-capacity floored). See FINDINGS.
    REMAINING: RAG (prefill-bound) re-measure is still pre-fix — folded into TODO 8.

## Done (for reference)
- Synth (decode-bound) characterized: out/diag/{SESSION_LOG,FINDINGS,REPRO,SUMMARY}.md. NOTE the
  EDPP-quality numbers were re-measured after the responsive-`z_ttft` fix; load-knee/ITL/means still pending.
- Reservation-gap fix (6a97a2f), weighted-PD default (8728a4f), routing-decision-trace (0869c00).
- Responsive `z_ttft` (2026-06-29): continuous credit from observed elapsed-wait + first-token true-up;
  retired the wrong "ttft_d accuracy" fix direction. edpp@2P2D 75%→98% TTFT-SLO. See item 10 above.
