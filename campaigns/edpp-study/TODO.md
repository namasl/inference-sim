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
   - **Mechanism-check FIRST** (before the 4-arm sweep): `never@4` on A-only vs A+B — does A's ITL/TTFT
     measurably degrade when B is mixed in? If BLIS doesn't model prefill↔decode batch interference,
     this favorable mechanism doesn't exist and we need a different angle (and learned something about
     the sim). Verify the latency model makes t_iter depend on co-batched prefill tokens.
   - Then 4 arms at FIXED split (e.g. 2P2D): never-in-split / always / prefix-threshold / edpp.
     Win condition for EDPP: protect A's ITL/TTFT (disaggregate B) without paying A's transfer cost
     (keep A local) → beat both brackets on A's metrics.
2. **Oracle / hindsight baseline for Q2:** label each request with the decision that *would* have
   minimized its (and neighbors') SLO violation, compare EDPP's choices against it. Quantifies decision
   accuracy independent of topology.
3. **Forced-disaggregation correctness:** in a regime where disaggregation IS the right call for a
   subset, does EDPP identify that subset? (precision/recall of EDPP's disaggregate decisions vs oracle.)

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
8. RAG (prefill-bound) under the corrected weighted-PD default + per-class SLOs — re-measure (earlier
   RAG numbers predate the routing fix). Does the decode-node-count story invert for prefill-bound?
9. Pin the synth knee more precisely (one point at rate ~0.75) — cosmetic.

## TODO — possible EDPP improvement (only after Q2 verdict)
10. HOL/occupancy-aware ttft_d: incorporate running-decode KV/batch occupancy into the local-TTFT
    estimate so EDPP stops over-keeping-local on a saturated decode pool. Prototype + re-test.

## Done (for reference)
- Synth (decode-bound) fully characterized: out/diag/{SESSION_LOG,FINDINGS,REPRO,SUMMARY}.md.
- Reservation-gap fix (6a97a2f), weighted-PD default (8728a4f), routing-decision-trace (0869c00).
