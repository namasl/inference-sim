# EDPP Empirical Study — Findings

Canonical findings doc. Detail/lab-notebook: `out/diag/SESSION_LOG.md`. Headline table:
`out/diag/SUMMARY.md`. Backlog: `TODO.md`. Repro: `out/diag/REPRO.md`.
Superseded artifacts (do NOT cite): `out/diag/ARCHIVE_round-robin-artifact.md` (harness
misconfig) and `out/diag/ARCHIVE_lagged-z-ttft-artifact.md` (the lagged-`z_ttft` flaw — see
"Responsive z_ttft" below).

EDPP is a Lyapunov **time-average** optimizer → report MEANS alongside p99 (means are arguably what it
targets). All results below use the llm-d **weighted PD default** (`precise-prefix-cache:2,queue-depth:1`).

## Framing — two distinct questions
- **Q1: should this workload run disaggregated at all?** Provisioning/topology choice; EDPP doesn't
  control it. Comparing EDPP to `never@4` answers Q1, not EDPP's quality.
- **Q2: given a disaggregated deployment, does EDPP disaggregate the RIGHT requests?** The algorithm's
  actual correctness — baselines hold topology FIXED (always / never-in-split / oracle).
- Q1 for uniform decode-bound synth is settled (below). **Q2 is still the priority** — but the
  responsive-`z_ttft` fix (below) materially changed EDPP's behavior, so prior Q2-flavored verdicts
  ("EDPP actively hurts") are retracted.

## Responsive z_ttft — the fix that changed the EDPP story (2026-06-29)

**The earlier "EDPP is the worst decider / partial disaggregation actively hurts / SLO 0.75" verdict
was an artifact of an engineering flaw, not the algorithm.** `z_ttft` (the TTFT SLO virtual queue)
was updated only at full request completion, so on a saturating decode pool the TTFT-miss signal
arrived ~100s+ late: `z_ttft` was 0 for **81%** of decisions and first went positive **81% through
the run**, leaving the SLO term dormant while EDPP kept loading the failing pod. (The competing
"HOL-blind `ttft_d` predictor" diagnosis was disproved: `ttft_d` enters only as
`z_ttft·(TTFT_P−TTFT_D)`, so it is multiplied by zero exactly when `z_ttft=0`; its accuracy does not
move the decision. Full provenance: `ARCHIVE_lagged-z-ttft-artifact.md`.)

**Fix:** credit `z_ttft` continuously from each in-flight request's observed elapsed wait (a certain
lower bound on its TTFT miss), trued up at first token (new `OnFirstToken` hook) or, as a fallback,
at completion. Same total contribution per request as before — credited earlier — so the virtual
queue's fixed point and the drift-plus-penalty guarantee are unchanged; only the feedback delay is
cut. Design: `docs/superpowers/specs/2026-06-29-edpp-responsive-z-ttft-design.md`.

## Workload: synth (decode-bound), rate 2.0, 2000 reqs, equal 4-node HW

KEY RESULTS (responsive `z_ttft`; pre-fix numbers archived):
- **`never@4` wins** (TTFT p99 0.16s, SLO 100%). Disaggregation does NOT help this decode-bound
  workload at equal hardware — outcome tracks DECODE-CAPABLE NODE COUNT. **(Q1: don't disaggregate;
  add decode nodes.)** Unchanged by the fix (never/always are static deciders).
- **EDPP now tracks `always` within a fixed topology** — it is no longer the worst decider:

  | arm | disagg | decoded | TTFT mean | TTFT p99 | TTFT-SLO@2s |
  |---|---|---|---|---|---|
  | never@4 | 0 | 2000 | 0.07s | 0.16s | 100% |
  | edpp@1P3D | 1625 | 2000 | 1.25s | 19.6s | 99% |
  | edpp@2P2D | 1625 | 2000 | 3.55s | 180s | 98% |
  | edpp@3P1D | 1633 | 1218 | 7.60s | 427s | 98% |
  | always@2P2D | 2000 | — | 0.08s | 0.16s | 100% |

  edpp@2P2D went **75% → 98%** TTFT-SLO (p99 518s → 180s) vs the pre-fix binary, by disaggregating
  breaching requests early. `z_ttft` first positive at **16%** of the run (was 81%), positive for
  **84%** of decisions (was 19%), disagg 52% → 81%.
- **3P1D (1 decode node) is the genuine saturation collapse** — only 1218/2000 decode (nothing to
  balance); the disaggregated prefill still gets fast first-token (SLO 98% on TTFT), but decode is
  capacity-bound. Q1 limit, not an EDPP defect.
- `prefix-threshold` ≈ `always` on synth (tiny inputs trip threshold-16 → ~100% disagg).

## Load-dependence (knee) — RE-MEASURED under the fix; the "rate-1.0 cliff" is GONE
edpp@2P2D across rates (pre-fix → responsive `z_ttft`):

| rate | disagg% | TTFT mean | TTFT p99 | SLO@2s |
|---|---|---|---|---|
| 0.5 | 0.0% | 0.07s → 0.07s | 0.2s → 0.2s | 100% (byte-identical; non-saturated ⇒ untouched) |
| 1.0 | 71.4% | **15.48s → 0.18s** | **162.1s → 0.2s** | 100% (the old sharp cliff is eliminated) |
| 1.5 | 80.4% | 50.74s → 0.54s | 382.0s → 5.1s | 99% |
| 2.0 | 81.2% | 56.32s → 3.55s | 518.3s → 180.2s | 98% |
| 3.0 | 83.2% | 48.29s → 3.46s | 577.4s → 141.0s | 98% |

EDPP now stays healthy through rate 1.5 and degrades gracefully only at 2.0–3.0 (genuine 2-decode-node
saturation). The pre-fix "harmless only at 0.5; sharp cliff at 1.0" finding is RETIRED.

## ITL decision path — RE-MEASURED; conclusion REINFORCED
Tightening `τ_itl` 150→50ms (rate 2.0) now makes EDPP disaggregate MORE under the fix (58% → **90%**),
but **ITL mean stays 72ms** (unchanged; > 50ms target). Disaggregation moves only PREFILL; decode stays
on the same 2 nodes, so ITL is floored by decode capacity. **EDPP's sole lever (prefill placement) is
matched to TTFT/HOL but mismatched to ITL/decode-capacity** — the fix sharpens the response but cannot
beat the capacity floor. NOTE `z_itl` still has the SAME completion-lag flaw `z_ttft` just had
(TODO 10b: apply the responsive-update treatment), though it won't move this capacity-bound ITL.

## Still pre-fix (lower priority)
- **RAG (prefill-bound)** under the corrected default + per-class SLOs — never re-measured post-fix
  (TODO 8). Does the decode-node-count story invert for prefill-bound?

## Anomalies & instrument audits (kept)
- The round-robin "collapse/161s/1192" numbers were a harness misconfiguration (default
  `--routing-policy round-robin` + no per-pool scorers → decode pool unbalanced). Fixed by defaulting
  PD pools to llm-d weighted. **Lesson: audit the routing config (the "knob") before blaming the
  algorithm.** (Archived: `ARCHIVE_round-robin-artifact.md`.)
- The "EDPP actively hurts" verdict was a control-law flaw (lagged `z_ttft`), not the algorithm.
  **Lesson: when a feedback controller looks bad, check WHEN/from-what its feedback updates.**
- `--num-instances N` alone is fatal for PD (needs role'd instances) — use explicit P/D splits.
- per-request ITL is recorded unconditionally on replay (`itl_mean_us`); `--record-itl` is observe-only.
- `slo_class` valid set: critical/standard/sheddable/batch/background (NOT "interactive").

## Q2 — heterogeneous favorable workload: EDPP FAILS (externality-blind) [2026-06-29]
First real Q2 test (requests DIFFER, so there's a right subset to disaggregate). Workload
(`specs/hetero/hetero.yaml`): A (85%, small prefill / substantial decode / tight 200ms TTFT) should stay
LOCAL; B (15%, ~16k prefill / tiny decode / batch SLO) should DISAGGREGATE (its prefill interferes with
co-resident A's TTFT — mechanism verified). 2P2D, decode ADEQUATE (bottleneck = interference, not capacity).

| arm | A TTFT p99 | A SLO viol@200ms |
|---|---|---|
| never-in-split (B local) | 259ms | 2.7% |
| always / prefix-16 (all disagg) | 271ms | 2.5% |
| **edpp** | **256ms** | **2.5%** |
| **ORACLE (disagg B only, A local)** | **51ms** | **0.0%** |

**A 5× tail win (259→51ms) + 100% SLO is achievable; EDPP captures none of it.** EDPP disaggregates
A 0% (correct) but B only 2% — it keeps the interfering big-prefill requests LOCAL, so A is no better
than never-in-split. **Structural, not tuning:** B's huge `W_p` enters only via `balance_term_d =
q_d·(W_p/W*_d)`, and `q_d≈0` when decode is adequate; B's own TTFT is far under its loose 5s SLO so
`z_ttft(batch)=0`. Worse, B's loose SLO inflates `W*_d` (∝τ_ttft), making B's balance term *smaller*
than A's — so lowering `V` disaggregates A before B (backwards). **Root cause: EDPP judges each request
by its OWN class SLO pressure + shared backlog; B's prefill harming A's TTFT is an externality on
neighbors that the rule cannot express.** Detail + fix direction in SESSION_LOG / TODO.

## Prefill-bound RAG (inference-perf batch-summarization) — agg wins; EDPP over-disaggregates [2026-06-29]
Real catalog workload (prefill_decode_ratio 30): vector-qa (78%, ~2k in, standard τ 500ms) + doc-read
(22%, ~60k in, batch). GPU-matched 16 GPUs (4×TP4). vector-qa TTFT SLO-violation@500ms:

| arm | r=1.0 | r=3.0 | r=6.0 |
|---|---|---|---|
| **agg-4 (NO disagg)** | **0%** | **9%** | **53%** |
| prefix-thresh 2P2D | 15% | 76% | 98% |
| edpp 2P2D | 11% | 36% | 90% |
| edpp 1P3D | 11% | 35% | 69% |

**Q1: agg-4 wins at every rate** — GPU-matched dedicated-role splits steal flexible capacity (agg's 4
nodes all prefill via chunked interleaving ≈ 2× a 2P-pool's throughput) and short prefills wait behind
doc-read 60k prefills in the prefill pool. **Q2: EDPP over-disaggregates short vector-qa (60-76%)** —
better than prefix-threshold (≈100%) but far worse than agg. MECHANISM: EDPP predicts `ttft_p < ttft_d`
(disagg faster) for short reqs, but `ttft_p` under-predicts prefill-pool congestion because the clogging
doc-read prefills are RUNNING, not WAITING (`q_p≈0`) — the SAME waiting-vs-running blindness as `ttft_d`
on synth, now on the prefill side. Caveats: GPU-matched framing, single seed.

## SYNTHESIS — EDPP never beats no-disaggregation at equal HW (4 workloads)
- **synth (decode-bound):** `never@4` wins; EDPP ranges harmless→harmful (responsive-`z_ttft` fix makes
  it *behave* sensibly but it still can't add decode nodes).
- **hetero (mixed, decode-adequate):** EDPP *under*-disaggregates the big-prefill B (externality-blind);
  oracle shows a 5× A-TTFT win EDPP misses.
- **RAG (prefill-bound):** `agg-4` wins; EDPP *over*-disaggregates short requests (`ttft_p` blind to
  running prefill-pool congestion).
- **vector-qa-only (clean single class):** `agg` wins on COMPLETIONS (893/1000 @ 0% viol vs PD splits
  ≤532); EDPP completes fewest (294) + adds violations — worst arm. Role dedication starves the
  under-provisioned side; "0% TTFT" on PD splits is a truncation mirage (read completions, not
  TTFT-of-completed).

Common roots: (1) role dedication wastes flexible capacity → lower completion throughput at equal HW;
(2) predictors see only WAITING backlog, blind to RUNNING occupancy (`ttft_d` and `ttft_p`); (3) the
own-class SLO + pool-backlog rule misjudges *when* to disaggregate and the cross-request *externality*.
**CAVEAT (untested — the real open door):** all comparisons are GPU-MATCHED (same 16 GPUs repartitioned).
Disaggregation's production value is INDEPENDENT SCALING (add prefill nodes without stealing from decode),
which equal-HW cannot show. Fair claim: *at equal HW, repartitioning into P/D roles (and EDPP's routing
within it) does not pay off on these workloads* — NOT *disaggregation never helps*. The honest next test,
if pursued, is a non-GPU-matched (independent-scaling) comparison.

## Open (priority: Q2)
- **The externality term** (fix direction from the result above): weight a request's prefill
  interference cost by the *co-resident decode pool's* SLO pressure (the victims'), not the deciding
  request's own `z`. This is the missing ingredient for EDPP to capture the favorable mechanism.
- Robustness: re-run the hetero workload with amplified B (frequency/size) — the structural finding is
  robust but the magnitude (2.7% interference signal) is one operating point.
- Still open: `z_itl` responsive-update (TODO 10b), per-instance backlog coherence (TODO 12),
  RAG re-measure (TODO 8).
