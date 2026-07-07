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

## Stage A — estimator-validation harness (2026-07-01)

New `--pd-outcome-trace` CSV (run + replay) emits one row per request pairing realized
admission delay (per prefill/decode sub-req, `T_adm = schedule − enqueue`), TTFT, mean ITL,
E2E, and completion, keyed by `request_id`. `analyze/estimator_validation.py` joins it against
`--edpp-decision-trace` on `request_id` and reports predicted-vs-realized bias, split by
disaggregated × slo_class over completed requests. This validates the *shipped* (waiting-only,
occupancy-blind) forward estimators; it is the baseline Stage C's occupancy-aware roll-forward
will be re-measured against. Measurement-only — no decider/routing change.

**Anchor run** — synth @ 2P2D, rate 2.0, 5000 reqs (bake `blis run --num-instances 4
--trace-output`; replay `--num-instances 4 --prefill-instances 2 --decode-instances 2
--pd-decider edpp --edpp-coeffs coeffs-llama70b-h100-tp4.json --edpp-tau-ttft 2s
--edpp-tau-itl 150ms`). EDPP disaggregated 4545/5000 (91%); 455 kept local. Bias
(median ratio realized/predicted, and tail):

- **ttft_p (disagg, batch):** median 2.08× under-pred; predicted p50 63ms vs realized p50 120ms;
  tail tight (p99 pred 93ms / real 190ms). For disaggregated requests TTFT is prefill-dominated,
  so the decode queue does not enter TTFT — it lands in E2E/ITL.
- **ttft_d (local path, HOL-blind):** median 2.74× but **tail catastrophic — p90 realized 324s vs
  predicted 0.55s (~590×), p99 real 366s vs pred 0.68s**. This is the archived "predicted 14.7s vs
  realized 542s" HOL-blindness (`ttft_d` built from waiting backlog, ignoring RUNNING decode
  occupancy) — reproduced, confirming the harness.
- **Decode admission delay (waiting-only estimate qd_raw/mu_d_nom):** **median ~905× under-pred —
  predicted p50 0.31s vs realized p50 285s** under 2-decode-node saturation. The clearest single
  quantitative case for Stage C (occupancy-aware admission-delay roll-forward).
- **Prefill admission:** median 1.33× (predicted ~0 vs realized ~15.6ms) — the estimator predicts
  no prefill wait; real wait is small.

**Conclusion:** the harness works end-to-end and reproduces the qualitative, tail-heavy
under-prediction of the shipped occupancy-blind estimators. The waiting-only decode signal is
optimistic by ~3 orders of magnitude under saturation. This is the Stage-A baseline.

**Reproduction (one command).** `bash campaigns/edpp-study/repro_stage_a.sh` (tracked) rebuilds
`blis` if needed, bakes the trace, replays 2P2D@edpp with both traces, and writes the bias report to
`campaigns/edpp-study/out/stage_a/bias.json` (out/ gitignored). Deterministic (INV-6): the run above
is byte-identical across invocations. Inputs pinned by the script: model
`meta-llama/llama-3.3-70b-instruct`, coeffs `scripts/calibration/coeffs-llama70b-h100-tp4.json`, spec
`campaigns/edpp-study/specs/synth_rate2.0.yaml` (5000 reqs). Runtime ~110s. **Checkpoint —
`bias.json` must show** (else the harness, not the estimator, regressed): total 5000 / completed 5000;
`disagg=True,class=batch` n=4545 with ttft median-ratio ≈2.085, decode-admission median-ratio ≈904.8
(pred p50 ≈0.312s, real p50 ≈285.8s), prefill-admission median-ratio ≈1.326; `disagg=False,
class=unknown` n=455 with ttft median-ratio ≈2.740, real p90 ≈324s vs pred p90 ≈0.553s. The two traces
the analysis joins are produced by `--edpp-decision-trace` (needs `--trace-level decisions`) and
`--pd-outcome-trace` (no trace-level needed) — both emitted by the same replay invocation.

**Known limitations / follow-ups (not blockers):**
1. Local (non-disaggregated) requests carry no `ParentRequest`, so their `slo_class`/`input_tokens`
   are empty/0 in the outcome CSV (populated only from a parent's `OriginalRequest`). The analysis
   buckets them as `class=unknown` (was: silently dropped by NaN groupby — fixed). A proper fix
   populates class/tokens for local rows from the original request stream.
2. The outcome `completed` flag uses the `RequestE2Es[id] > 0` proxy, which marked 5000/5000 here
   vs the sim's `completed_requests` = 4545 — the proxy is looser than the sim's completion
   definition; reconcile before using `completed` as a hard filter.
3. Realized decode `T_adm` (p50 285s) coexists with realized TTFT p50 120ms on the same disagg
   requests — the decode-side TTFT is measured from decode arrival (omits the pre-decode wait),
   the long-standing "decode-side TTFT understated" issue. Stage A surfaces it; it is orthogonal
   to this harness.
4. PD+EDPP *replay* on this full config completed 4545/5000 (2-decode-node saturation, expected) —
   NOT a parity bug. An earlier smoke run's 23/50 was a small-config horizon artifact, not an
   engine defect. `--pd-outcome-trace` admission-time columns match run byte-for-byte.

## Stage B — corrected work model + per-request validation (2026-07-02)

Corrected the EDPP work formulas to equal the work the **active (trained-physics) latency model
actually charges**, and validated per-request against realized trajectory work via a new
`--edpp-work-trace` CSV (run + replay). Same frozen coeffs, no refit.

**Corrected formulas** (`sim/edpp_coeffs.go`):
- `Wp(a_p, a_r) = C_pf·a_p + C_attn·a_p·(a_r + a_p/2)` — the trained-physics basis (**+ a_p/2**), matching
  `trained_physics_model.go`'s per-step attention charge `ti·(si + ti/2)` (si = full prompt) that
  `C_attn` was calibrated to. Replaces the old no-cache `C_pf·a_p + (C_attn/2)·a_p²`.
- `Wd(a_r, o) = C0·o + C1·o·(a_r + (o−1)/2)` — exact discrete decode sum `Σ_{k=0}^{o−1}(C0+C1(a_r+k))`,
  matching the model's per-decode-step charge `C0 + C1·ProgressIndex`. Replaces the old
  `N̂_out·(C0 + C1·NomDecodeCtx)` fixed-nominal form (`NomDecodeCtx` retained only for the
  `selectedDecodeState` fallback).

**Validation (synth@2P2D and rag@2P2D, rate 2.0, 5000 reqs each; `bash campaigns/edpp-study/repro_stage_b.sh`):**

| workload | single-chunk prefill `max_abs_rel_err` | decode `max_abs_rel_err` | chunked prefill (n, max err) |
|---|---|---|---|
| synth | 5.6e-16 (n=4701) | median 0, p90 2.5e-4 (n=4456; small preemption-tail, max ~10%) | n=299, max 0.28 |
| rag   | 5.8e-16 (n=1418) | 1.2e-15 (n=5000, fully exact) | n=3582, max 0.22 |

**Model is exact.** Single-chunk prefill and decode work match realized to float precision on both
workloads — the corrected `W_p`/`W_d` ARE the closed forms of the active latency model's per-request
charge. The chunked-prefill residual (realized < closed, median ~−0.9%) is the **documented,
expected** `C_attn·(a_p² − Σs_r²)/2` term: the closed form assumes single-chunk prefill, so chunked
requests accrue slightly less attention work. This is reported, not an error.

**Correction effect:** per-request `wp_closed / wp_closed_nocache_old` median ≈ 1.04 on both (the linear
`C_pf·a_p` term dominates at these operating points; the ~3× attention-basis change is only visible
where `a_p ≈ a_r`). NOTE: synth here shows median `cache_hit_frac` 0.91 (heavy prefix reuse — synth is
NOT a no-cache workload as earlier assumed), rag 0.18. The corrected cross term (`a_r` dependence) is
what the old no-cache form dropped.

**Expected decision shift (NOT a regression):** the corrected `W_p` attention term differs ~3× from the
old form at no-cache, so EDPP routing decisions shift vs pre-Stage-B. This is the correction working.
Precise pre/post disagg-fraction quantification is a deferred follow-up (requires a pre-Task-1 rebuild);
the per-request work exactness above is the correctness gate, not the decision delta.

### Documented latency-model fidelity gap (deferred, NOT fixed here)

The default trained-physics latency model charges prefill attention on a **full-input-length** basis
(`ti·(a_r + ti/2)`), which over-counts physically-causal (triangular, `≈ N²/2`) attention by up to ~3×
at single-chunk. `C_attn` is calibrated to this basis, and Stage B's `W_p` matches it **deliberately**
so EDPP's congestion estimate is consistent with the simulator it runs in — the correct choice for
routing-decision fidelity. The **roofline** backend (`roofline.go`) uses the physically-causal
prior-context basis (`≈ a_r − a_p/2`), which matches design-doc §3.6. So §3.6's causal form is
physics-pure but inconsistent with the default model; Stage B's `+a_p/2` matches the default model.
Fixing the trained-physics basis to causal + refitting `C_attn` (invalidating the frozen coeffs and all
prior findings) is a separate, deferred task. If a future study runs the roofline backend, `W_p` must
switch to the causal basis; the accumulator already reads whichever model is active.

**Reproduce:** `bash campaigns/edpp-study/repro_stage_b.sh` → `campaigns/edpp-study/out/stage_b/{synth,rag}_bias.json`
(~4 min, deterministic). Checkpoint: single-chunk prefill and decode `max_abs_rel_err` < 1e-6 on both;
chunked residual is the `Σs²` term.

## Stage C — occupancy-aware admission estimator + fidelity ablation (2026-07-02)

Made EDPP's admission-delay estimate pluggable (replacing the occupancy-blind `qD/muDec`·`qP/muPf`
terms in `ttft_d`/`ttft_p`) with four deployable variants — `waiting` (shipped strawman), `little`
(Little's law, aggregate), `fluid` (occupancy-conditioned mean-field `N_ahead/X̂_dep`), `rollforward`
(true per-request deterministic batch look-ahead) — plus `fluid_oracle`/`rollforward_oracle`
(true-`o_r`, logging-only; INV-9 guard forbids them as routing drivers). New `--edpp-admission-trace`
logs, per request×pool, the realized admission delay + all six predictions. `local_t_adm` capture
(Stage A gap) closed. Default `waiting` → decider byte-identical.

**Microbenchmark** (`repro_stage_c.sh`): 1P1D, `--pd-decider edpp`, synth rate 2.0; T1 forces all-local
(`--edpp-c-xfer 100s`) to isolate `ttft_d`; T2 forces all-disagg (`--edpp-c-xfer 0s`). Routing is
forced via the transfer-penalty knob (EDPP's `Decide` must run to assemble the estimator contexts, so
`never`/`always` deciders can't be used). Median ratio realized/predicted (>1 = under-prediction):

**T1 local pool — `ttft_d` isolation, saturated (realized p50 ≈ 560s):**
| estimator | median ratio realized/pred |
|---|---|
| waiting (shipped) | **57.3×** under-predicts |
| little | n/a (predicts 0) |
| fluid | 2.6e6× (anomalous — see below) |
| **rollforward** | **1.29× — near-exact** |

**Headline: the true `rollforward` estimator collapses the shipped estimator's ~57× admission
under-prediction to ~1.3× on the saturated local decode pool** — the direct fix for the mechanism
Stage A measured (~905×). N̂_out-prediction error was ~0 here (`rollforward` ≈ `rollforward_oracle`),
so the residual 1.3× is estimator *form*, not output-length prediction.

**Open issues the ablation surfaced (the clean 4-point monotonic collapse is NOT yet achieved):**
1. **`fluid` is anomalous** — it under-predicts by ~1e6× (predicts ~µs where reality is ~100s). The
   mean-field `N_ahead/X̂_dep` with the current `RemainingStepsEst`/free-slot inputs collapses to
   near-zero on the local path. Needs debugging (likely `RemainingStepsEst` mis-scaled or the free-slot
   early-return misfiring). It should sit between `waiting` and `rollforward`, not below both.
2. **`little` is inert** — predicts 0 (ratio n/a). Despite the Task-3 `DispatchRate→AdmissionRate`
   wiring, the admission rate is effectively unavailable at decision time in this run (DispatchRate 0
   until first completion, and/or not surfacing). Needs a reliable per-instance admission-rate signal.
3. **Prefill-pool estimators are broken** — realized prefill admission is a constant ~15.5ms and no
   estimator tracks it; the snapshot enrichment added only the *decode* running batch (`RunningDecode`),
   so the prefill-pool context has no occupancy. `ttft_p` cannot be validated until the prefill pool is
   enriched symmetrically.
4. **`rollforward` OVER-predicts on the disagg decode path** (T2 decode: 0.41×, i.e. predicts ~2.5×
   too high). In disaggregation the decode sub-request is admitted quickly after transfer, so the
   full-batch-drain assumption over-estimates; the 905×-analog under-prediction lives on the *local*
   (kept-local-under-saturation) path, which T1 captures and `rollforward` fixes.

**Status:** the estimator infrastructure (pluggable interface, four variants + oracle, INV-9 guard,
`--edpp-admission-trace`, `local_t_adm`) is complete and reviewed. The **proposed `rollforward`
estimator is validated on its target scenario** (57×→1.3× on the saturated local pool). The `fluid`
under-prediction, `little` inactivity, and prefill-pool enrichment are **follow-ups required before the
full monotonic ablation and the paper's fidelity figure are publishable**.

**Reproduce:** `bash campaigns/edpp-study/repro_stage_c.sh` → `out/stage_c/{t1,t2}_ablation.json`.
Checkpoint: T1 local `waiting` median ratio ≈ 57×, `rollforward` ≈ 1.3×.
Limitations carry Stage B's (trained-physics attention basis); single saturating operating point.

### Stage C follow-up: utilization sweep (validity of the saturating operating point)

The T1/T2 microbenchmark uses a single saturating operating point (synth rate 2.0 on one decode
engine). Under overload the backlog grows and admission delay is non-stationary (each request waits
longer than the last), so the reported median ratios summarize a moving target — a deliberate stress
test, not a steady-state claim. The per-request *conditional* comparison (predicted vs realized at each
request's own decision instant) is valid regardless, and is precisely why the conditional `rollforward`
tracks the growing delay while the aggregate `little` cannot. REQUIRED FOLLOW-UP before the paper's
fidelity figure: a **utilization sweep** at loads approaching but below single-engine capacity, where
admission delay is large but bounded/stationary, so predicted-vs-realized has a well-defined
steady-state meaning. Pair it with the fluid/little/prefill fixes.

### Stage C CORRECTION (2026-07-05): the 57×→1.3× headline is oracle-contaminated

A root-cause investigation of `fluid`'s under-prediction surfaced a **measurement-validity bug**: the
deployable `rollforward`/`fluid` and their `_oracle` variants are the same impl and read
`RunningReqState.TrueRemaining` whenever it is ≥0. Enabling `--edpp-admission-trace` turns on oracle
mode, which POPULATES `TrueRemaining`. So in the T1/T2 microbenchmark the "deployable" `rollforward`
was actually reading oracle remaining — which is why `rollforward` and `rollforward_oracle` were
identical (both 1.29×) and the decomposition showed ~0 N̂_out error. **The 1.29× "rollforward fixes
57×→1.3×" result is oracle-fed; the true deployable-`N̂_out` number is UNKNOWN pending a fix that
prevents deployable estimators from reading `TrueRemaining`.** Also found: `RemainingStepsEst` collapses
to 1 (RemainingDecodeWork never populated + a mean-based fallback that goes negative under saturation),
and `N̂_out` is biased low under saturation (survivorship — only short requests have completed), which
inherently limits the *deployable* estimators. These are being addressed as a fix-cluster (fluid wave
mean-field + oracle/deployable separation + N̂_out handling + little admission-rate + prefill enrichment
+ CLI flag). Treat all Stage C ablation numbers above as PROVISIONAL until the fix-cluster lands.

### Stage C — DE-CONFOUNDED RESULTS after fix-cluster (2026-07-05, supersedes the provisional/1.29× numbers)

All six fix-cluster items landed (fluid wave form; censored per-request remaining-steps; windowed
admission-rate; oracle/deployable separation; prefill enrichment; `--edpp-tadm-estimator`) plus the
second round (rollforward queue-waves; `little` pool-filtered-snapshot fix). `repro_stage_c.sh` re-run,
median ratio realized/predicted (>1 = under-prediction, <1 = over-prediction; ~1 = accurate):

| pool (workload) | waiting | little | fluid | rollforward | rollforward_oracle |
|---|---|---|---|---|---|
| **local (T1, realized p50 ≈ 560s)** | **57.3×** | 1.30× | **1.16×** | **1.16×** | 1.25× |
| decode (T1) | 12.9× | 1.81× | 0.98× | 0.98× | 0.98× |
| decode (T2) | 4.05× | 2.70× | 0.34× | 0.34× | 0.34× |
| local (T2) | 1430× | 8.4× | 0.35× | 0.35× | 0.35× |

**Findings:**
- **Occupancy-blindness is the dominant error.** `waiting` (waiting-backlog ÷ drain) under-predicts
  admission delay 57×–1430× on saturated pools — the mechanism Stage A first measured.
- **The occupancy-aware estimators fix it.** On the target T1 local pool, `fluid`/`rollforward` reach
  1.16× (near-exact) and `little` 1.30×. The key ingredient is accounting for the queue-ahead
  (`fluid`'s wave form; `rollforward`'s queue-depth-aware departure walk).
- **`fluid` and `rollforward` converge** (identical on T1 local/decode): once `rollforward` rolls
  through `⌈(QueueDepth+1)/BatchSize⌉` waves, its deep-queue behavior IS the fluid wave form — so the
  "true per-request" estimator and the mean-field agree where the queue dominates. The per-request
  walk's extra fidelity matters only for shallow queues.
- **Oracle ≈ deployable** (T1 local 1.16 vs 1.25; decode identical): after the **censored `N̂_out`
  floor** (a request that produced `k` tokens has `o_r ≥ k`), the deployable estimate closes almost all
  of the gap to the true-remaining oracle. The `N̂_out`-prediction residual is small here.
- **T2 (disaggregated) decode/local slightly OVER-predict (0.34–0.35×):** a disaggregated decode
  sub-request is admitted quickly after transfer, so realized decode admission is smaller than the
  full-queue-drain the occupancy estimators assume. Still far better than `waiting`.
- **Prefill pool reads ~0 (nan ratio) — correct physics, not a bug.** The prefill pool is unsaturated
  at this operating point (QueueDepth=0, free slot), so the occupancy estimators correctly return ~0
  (`waiting` is live on 2086 prefill rows). Validating prefill estimators requires a prefill-saturating
  workload (follow-up).

**Retraction:** the earlier "rollforward fixes 57×→1.3×" headline was oracle-contaminated (deployable
estimators were reading the oracle `TrueRemaining`). De-confounded, the honest headline is:
**occupancy-blind `waiting` (57×–1430×) → occupancy-aware `fluid`/`rollforward` (~1.16× on the saturated
local pool, converging), with `little` a decent aggregate (1.3×) and the censored-`N̂_out` deployable
estimate ≈ the oracle.** Remaining follow-ups: the utilization sweep (bounded/stationary operating
points), prefill-saturating validation, and the prefill oracle-semantics nuance (prefill "remaining" is
known input length, not a hidden variable — the oracle/deployable split is decode-specific).

---

## Utilization sweep — decode-pool admission fidelity vs load (2026-07-06)

**Purpose.** Harden the single-point Stage C result (measured at forced heavy overload — a
*non-stationary* stress test) by measuring decode-pool admission-estimator fidelity across a range of
**bounded, stationary** operating points below capacity. Topology = Stage C T1 (1P1D, EDPP,
`--edpp-c-xfer 100s` ⇒ all-local on a single decode engine), synth, measurement-only. Reproduce:
`bash campaigns/edpp-study/repro_utilization_sweep.sh` → `out/utilization_sweep/sweep.json` (+`sweep.png`).

**Capacity.** λ* = **0.75 req/s** (composite saturation detector; lowest coarse-scan rate not STABLE —
0.1/0.25/0.5 STABLE, 0.75 first non-STABLE). The ρ grid `{0.5,0.7,0.85,0.9,0.95,0.98}·λ*` yields the
points below; the achieved ρ̂ is measured per point (`responses_per_sec/λ*`).

**Result — the sweep worked, and it reveals a STEP FUNCTION, not a smooth fidelity curve.**
Admission delay is bounded and **stationary** at every point (admission-delay drift = median(2nd
half)/median(1st half) ≈ 1.0 throughout), and grows only gently with load:

| ρ̂ | verdict | realized adm p50 | ITL mean | `waiting` | `fluid`/`rollforward` | `little` |
|------|-----------|-----------------|----------|-----------|-----------------------|----------|
| 0.50 | STABLE    | 29.0 ms | 27.2 ms | ≈0 (1e14× off) | **predict 0** (nan) | over-pred ~17× |
| 0.69 | STABLE    | 33.1 ms | 36.1 ms | ≈0            | **predict 0** (nan) | over-pred ~17× |
| 0.84 | STABLE    | 38.8 ms | 47.4 ms | ≈0            | **predict 0** (nan) | over-pred ~25× |
| 0.88 | STABLE    | 40.8 ms | 52.8 ms | ≈0            | **predict 0** (nan) | over-pred ~16× |
| 0.93 | STABLE    | 44.2 ms | 59.5 ms | ≈0            | **predict 0** (nan) | over-pred ~18× |
| 0.95 | OVERLOADED| 46.9 ms | 64.4 ms | ≈0            | over-pred ~90×      | over-pred ~2× |

(Ratios are `realized/predicted`; a `predict 0` yields nan/astronomical ratios, so read the **signed
error** at these points — it ≈ the full realized floor, i.e. the estimators contribute ~0.)

**Mechanism (why the occupancy estimators predict 0 below the cliff).** Every occupancy-aware estimator
opens with the same free-slot early-return (`fluid` at `sim/admission_estimator.go:66`, `rollforward` at
`:84`): `if BatchSize < MaxBatchSize && FreeKVBlocks >= ReqKVNeed { return 0 }`. Below saturation a slot and KV are free, so it fires and
returns **exactly 0** — modelling "a slot is free ⇒ admitted instantly." But the next `FormBatch` only
runs after the **current in-progress decode step finishes**, so a request enqueued mid-step still waits
the residual of that step (≈ one `T_iter`) plus dispatch-tick cadence before admission. The estimators
model **slot/KV availability**, not the **time to the next admission opportunity**. `waiting` (QWork/Mu)
also predicts ≈0 because the waiting queue is near-empty below capacity.

**This dropped term is confirmed to be ≈ one decode iteration.** The realized sub-saturation admission
floor tracks ITL (inter-token latency ≈ decode step time `T_iter`) closely and climbs with it as the
batch fills: 29↔27 ms at ρ̂=0.5, 44↔60 ms at ρ̂=0.93. So the floor the estimators omit **is** the
residual-step / `FormBatch`-cadence latency, ≈ `T_iter`.

**Interpretation (the honest headline).** For this decode-bound long-job workload the admission-delay
curve is a **step function**: a small (~30–47 ms), routing-irrelevant floor ≈ one decode step for all
ρ̂ below the saturation cliff, then an explosion above it (Stage C's 560s regime). There is **no wide
"large-but-bounded" band** to draw a smooth fidelity curve through — the transition is sharp, and
refining λ*/the grid cannot manufacture a band the physics does not produce. Consequences:
- **Stage C's 57×→1.16× win is real but regime-specific** — it validates the estimators where slot-wait
  dominates (heavy overload). Below the cliff the estimators correctly report ~0 *slot-wait*, but omit
  the O(`T_iter`) admission-opportunity latency that is the entire admission delay there.
- **This gap is routing-irrelevant at these SLOs:** the floor (~30–47 ms) is ≪ τ_ttft = 2 s, so z_ttft
  does not engage on it and it cannot change an EDPP routing decision. It is a documented modelling gap,
  not a routing defect.
- The occupancy-aware estimators are therefore *sufficient for routing* (accurate exactly in the regime
  that crosses the SLO and drives decisions) while structurally incomplete as a general
  admission-latency model (they miss the sub-saturation step-cadence floor).

**Reproduction checkpoint** (a correct re-run must reproduce): λ* = 0.75; admission-delay drift ≈ 1.0 at
every retained point (stationary); realized adm p50 rising ~29→47 ms across ρ̂ 0.5→0.95 and tracking ITL
mean; occupancy-aware estimators predict 0 (nan ratio, signed error ≈ realized) at all STABLE points;
the sole OVERLOADED point at ρ̂≈0.95. If instead a smooth monotonic bias curve appears, the harness or
the topology changed. **Numerical caveat:** below the cliff the realized delay is a small floor and the
estimators predict 0, so the *ratio* is meaningless — read the **signed error** (ms), not the ratio, in
this regime (the analyzer flags `ratio_meaningful` against a `ratio_floor_us` for exactly this reason).

**FLOOR IMPLEMENTED (2026-07-06, commit 1f2d4bd; design
`docs/superpowers/specs/2026-07-06-edpp-admission-floor-design.md`).** The occupancy-aware estimators
(`fluid`/`rollforward`/`little`, + oracles) now lower-bound their admission estimate by one `T_iter` via
`flooredTAdm(est,ctx)=max(est,TIter)` (the wait for the current decode step to finish before the next
`FormBatch`). `waiting` is left unfloored — it is the occupancy-blind strawman, and leaving it untouched
preserves the byte-identical default driver.

Re-run result (`repro_utilization_sweep.sh`) — the floor closes the sub-saturation gap:

| ρ̂ | verdict | realized p50 | `waiting` | `fluid`/`rollforward`/`little` (floored) |
|------|-----------|-------------|-----------|------------------------------------------|
| 0.50 | STABLE    | 29.0 ms | ≈0 (huge ratio) | **1.26×** |
| 0.69 | STABLE    | 33.1 ms | ≈0 | **1.17×** |
| 0.84 | STABLE    | 38.8 ms | ≈0 | **1.12×** |
| 0.88 | STABLE    | 40.8 ms | ≈0 | **1.06×** |
| 0.93 | STABLE    | 44.2 ms | ≈0 | **1.06×** |
| 0.95 | OVERLOADED| 46.8 ms | ≈0 | **1.05×** |

The occupancy-aware estimators now track the realized floor to ≈1.05–1.26× across the whole STABLE band
(best near ρ→1; slightly high at low load because one `T_iter` marginally under-shoots the realized
floor); `waiting` still predicts ≈0 (unfloored strawman, still the occupancy-blind baseline).

**Stage C overload headline preserved, with one honest nuance.** Re-running `repro_stage_c.sh`: T1 local
`waiting` = **57.3×** (unchanged), occupancy-aware `fluid`/`rollforward` = **~1.24×** (`little` 1.30×,
oracles ≈ deployable). The occupancy-aware figure shifted slightly from the pre-floor **1.16×** — this is
NOT the floor leaking into the saturated path: `max(est, T_iter)` is a no-op for genuinely saturated rows
(est ≈ 560 s ≫ `T_iter`), so no saturated prediction changed. The T1 run aggregates over the whole
trajectory including the sub-saturation ramp-up, and those early free-slot rows now floor to `T_iter`
instead of 0, nudging the median. Per-row the saturated predictions are identical; the headline (blind
`waiting` 57× vs occupancy-aware ~1.2×) stands, and the occupancy-aware estimators are now consistent
(~1.05–1.26×) across the entire load range instead of collapsing to 0 below the cliff.

The floor is estimator-fidelity only — routing-irrelevant (the floored delay ~30–47 ms ≪ τ_ttft = 2 s, so
`z_ttft` never engages on it) — and `waiting` (the default driver) is byte-identical.

## Counterfactual regret — per-decision hindsight diagnosis of reduced-EDPP (2026-07-07)

**Purpose.** An *exact* per-decision one-step-deviation regret for the shipped (reduced) EDPP
decider. Capture the policy's realized per-request `(decode, prefill)` plan, then for each sampled
request replay the *entire* plan with only that one request's action changed to each alternative
`a ∈ 𝒜 \ {plan(r)}`, and measure the change in aggregate goodput (`slo_attainment`). Regret(r) =
`max_a goodput(dev_{r,a}) − goodput(baseline)`, clamped at 0. Positive regret ⇒ some single
alternative decision would have improved total goodput in hindsight, i.e. EDPP left goodput on the
table at that decision. This is a *local* diagnostic (one decision moves at a time), **not** the
global P/D optimum — that is the later MILP yardstick's job.

**Setup.** Topology **1P2D** (`--num-instances 3 --prefill-instances 1 --decode-instances 2`),
decider `edpp` with the frozen Llama-70B/H100-TP4 coeffs, `τ_ttft=2s`, `τ_itl=150ms`,
SLO `ttft batch=2s, itl batch=150ms`. Trace: `specs/synth_cf.yaml` — the decode-bound synthetic-data
workload, shrunk to **800 requests at aggregate_rate 2.0** so the cluster saturates
(baseline goodput 0.9775, not 1.0) while each single-request deviation still has measurable leverage
on the aggregate (with the full 5000-req trace one row is diluted to ≈0). `|𝒜| = 2 decode × (1 prefill
+ local) = 4`, so 3 deviations per sampled request. First end-to-end run used **K=10** (30–36 sim
runs); scale K up for tighter statistics.

**Self-consistency gate (REQUIRED, passed).** Replaying the captured plan via `--pd-plan` reproduced
the baseline exactly: replay `slo_attainment = 0.9775` == baseline `0.9775` (INV-6/INV-13). The
fixed-plan decider is a faithful record/replay of EDPP's decisions, so the regret below is meaningful.

**Result (K=10).** baseline_goodput **0.9775**, mean_regret **0.0139**, total_regret **0.1387**,
frac_positive **0.70**. Where it concentrates:

| baseline decision | # sampled | regret | hindsight-best |
|---|---|---|---|
| kept **local** (decode unassigned, left to default routing) | 6 | **0.0225 each** (→ goodput 1.0) | pin decode to `instance_1` (local or disagg) |
| **disaggregated** (explicit decode + prefill=instance_0) | 4 | 3×`0.000`, 1×`0.0037` | mostly `baseline` |

**Interpretation.** *Nearly all* of reduced-EDPP's positive one-step regret sits on its **kept-local**
decisions — the requests it declined to disaggregate (6 of the 7 positive-regret decisions, 0.135 of the
0.1387 total; the one disaggregation decision with positive regret contributes just 0.0037). On those,
EDPP records no decode instance and
leaves decode placement to the default weighted router; pinning the decode instead (to `instance_1`)
recovers the last ~0.0225 of goodput and reaches 1.0. Its **explicit disaggregation** decisions are
locally near-optimal (regret ≈0; the largest is 0.0037). So the goodput reduced-EDPP leaves on the
table here is **not** a wrong disaggregate-or-not call — it is *unresolved decode-instance placement on
the local path*. This is consistent with the pool-average structure critique: EDPP scores against
pool aggregates and does not commit a per-instance decode target when it keeps a request local, and
that placement is exactly where a hindsight-better single decision exists. The effect is small in
absolute goodput (mean 0.0139) but systematic (0.70 of sampled decisions), and it is a decode-placement
gap rather than a P/D-split gap.

*Caveat:* because EDPP's local decisions carry an empty decode instance in the outcome trace, a
"deviation" on a local request also *pins* a decode target the baseline had left free — so this
regret blends "should have disaggregated" with "should have pinned a better decode". The gate passing
(empty-override replay == baseline) confirms the baseline is captured faithfully; the deviations are
genuine alternatives to it.

**Hand-case validation.** On a tiny idle 2-request 1P2D trace the harness reproduces the known answers
(recorded in `repro_counterfactual.sh`): all-local baseline under a loose SLO ⇒ regret 0 (local is
already optimal, no invented gains); all-disaggregate baseline under a tight `ttft=40ms` SLO (so the
KV-transfer hop misses while local meets) ⇒ positive regret with a *local* hindsight-best.

**Reproduce:** `bash campaigns/edpp-study/repro_counterfactual.sh` (`K=…`, `TARGET_POLICY=…`,
`SPEC=…` overridable). Diagnostic only (local one-step deviation), not the global optimum; it is the
shared fixed-plan (`--pd-plan`) infrastructure for the later full-joint decider and the MILP yardstick.
