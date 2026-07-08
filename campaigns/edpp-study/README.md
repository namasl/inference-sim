# EDPP Study — Orientation & Operator Guide

Start here. This is the durable onboarding doc for the EDPP disaggregation-routing study: what the
system is, which files to read, how to reproduce each result, and how to read the trace CSVs. Results
themselves live in `FINDINGS.md`; design rationale lives in the specs (see "Where things live").

---

## 1. First principles — what this is about

**BLIS** (repo root `inference-sim/`) is a deterministic discrete-event **simulator** for LLM inference
serving. It predicts request timings for a fleet of GPUs without real hardware.

Serving one request has **two phases**:
- **Prefill** — read the whole prompt, build the KV cache. One compute burst. Sets **TTFT**
  (time-to-first-token).
- **Decode** — generate output tokens one at a time, each re-reading the KV cache. Many small steps.
  Sets **ITL** (inter-token latency) and total time.

**Disaggregation ("PD")** runs prefill and decode on *separate* instances: a request is prefilled on a
prefill (P) node, its KV cache is transferred over the network to a decode (M) node that generates the
answer. The alternative is **local** — one node does both.

**EDPP** is our routing policy that decides, per request, **disaggregate vs local**. The new
formulation is a Lyapunov **drift-plus-penalty** rule: for each request it computes a cost balancing
(a) the **work** each action puts on each node (`W_p` prefill, `W_d` decode), (b) **TTFT/ITL SLO
deficits**, and (c) a **transfer penalty**, then picks the cheaper action. To act, it must *estimate*
two things before it can measure them: per-request **work** and **admission-delay/TTFT** (`T̂`).

The study asks two separate questions (see `FINDINGS.md` "Framing"):
- **Q1:** *should* a workload run disaggregated at all? (provisioning/topology — EDPP doesn't control it)
- **Q2:** *given* a disaggregated deployment, does EDPP disaggregate the **right** requests? (the algorithm's correctness)

---

## 2. Where things live

| What | Path | Tracked? |
|---|---|---|
| **Results / findings log** | `campaigns/edpp-study/FINDINGS.md` | ✅ tracked |
| **This orientation guide** | `campaigns/edpp-study/README.md` | ✅ tracked |
| **Backlog** | `campaigns/edpp-study/TODO.md` | ✅ tracked |
| **Reproduce scripts** | `campaigns/edpp-study/repro_stage_a.sh`, `repro_stage_b.sh` | ✅ tracked |
| **Paper-figure scripts** | `repro_work_model_sweep.sh` (Fig 1), `repro_ttft_d_local.sh` (ttft_d Figs A/B/C) | ✅ tracked |
| **Analysis scripts** | `campaigns/edpp-study/analyze/*.py` | ✅ tracked |
| **Workload specs** | `campaigns/edpp-study/specs/*.yaml` | mostly tracked |
| **Run outputs (traces, bias.json)** | `campaigns/edpp-study/out/` | ⛔ gitignored (reproducible) |
| **Design specs (rationale)** | `docs/superpowers/specs/2026-07-01-edpp-*.md` | ⛔ gitignored (on-disk only) |
| **Implementation plans** | `docs/superpowers/plans/2026-07-0*-edpp-*.md` | ⛔ gitignored (on-disk only) |
| **Task-by-task execution ledger** | `.superpowers/sdd/progress.md` | ⛔ gitignored (dies on `git clean -fdx`) |
| **Canonical joint-routing design** | `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` | ✅ tracked (design branch) |
| **Frozen coefficients** | `scripts/calibration/coeffs-llama70b-h100-tp4.json` | ✅ tracked |

**Code to read** (the policy + validation instrumentation):
- `sim/edpp.go` — `Decide()` (the drift-plus-penalty decision) and `OnRoute()` (books work into backlogs).
- `sim/edpp_coeffs.go` — the work/time formulas: `Wp`, `Wd`, `tIterDecode/Prefill`, coeff struct.
- `sim/simulator.go` — `executeBatchStep` (~line 749): the per-step engine loop; the per-request **work
  accumulator** and the `BLIS_STEP_CSV` calibration tap hook here.
- `sim/latency/` — the latency models that define "work" (default **trained-physics**; **roofline** is
  the physically-causal alternative — see the fidelity note in `FINDINGS.md` Stage B §7).
- `sim/cluster/cluster.go` — `BuildPDOutcomeRecords` / `BuildWorkTraceRecords` (correlate sub-requests
  to parents), the trace flag wiring.
- `sim/trace/` — the CSV writers (`pd_outcome_csv.go`, `work_trace_csv.go`, `edpp_csv.go`).

---

## 3. Reproduce a result (one command each)

Build first: `go build -o blis main.go` (the repro scripts do this automatically if `./blis` is absent).

- **Stage A — estimator validation** (does the *shipped* TTFT/admission estimator match reality?):
  ```
  bash campaigns/edpp-study/repro_stage_a.sh
  ```
  → `out/stage_a/{decisions,outcome}.csv` + `out/stage_a/bias.json`. ~2 min, deterministic.
  Headline: the shipped occupancy-blind estimator under-predicts decode admission delay ~900×.

- **Stage B — work model validation** (does the corrected `W_p`/`W_d` equal the work the simulator charges?):
  ```
  bash campaigns/edpp-study/repro_stage_b.sh
  ```
  → `out/stage_b/{synth,rag}_work.csv` + `out/stage_b/{synth,rag}_bias.json`. ~4 min, deterministic.
  Headline: single-chunk prefill and decode work match realized to float precision.

Each script **bakes** a trace (`blis run --num-instances 4 --trace-output …`), then **replays** it at a
2P2D split under EDPP (`blis replay --prefill-instances 2 --decode-instances 2 --pd-decider edpp
--edpp-coeffs …`) with the trace flags, then runs the analysis script. FINDINGS carries the exact
checkpoint numbers a correct run must reproduce (if a rerun disagrees, the *harness* regressed, not the
estimator/model).

**Key flags** (see `docs/design/2026-07-01-joint-routing-session-handoff.md` for the full list):
`--pd-decider edpp`, `--edpp-coeffs <json>` (required with EDPP), `--edpp-tau-ttft/-itl`,
`--edpp-decision-trace <csv>` (needs `--trace-level decisions`), `--pd-outcome-trace <csv>`,
`--edpp-work-trace <csv>`. `blis --rate` is ignored with `--workload-spec` (edit `aggregate_rate` in the spec).

---

## 4. The trace CSVs — column dictionary

All three join on `request_id`. Times/work in **microseconds**; `disaggregated`/`completed` are the
strings `"true"`/`"false"` (Go bools). Zero in a timing/`t_adm` column means "phase not reached".

### `--edpp-decision-trace` (what the policy *predicted*, one row per decision)
`request_id, clock, class, skip_reason` · `ap` (uncached prefill tokens), `wp` (prefill work),
`delta_pf_chunk` · `qd_raw`/`qp_raw` (raw decode/prefill backlog work), `qd`/`qp` (normalized) ·
`mu_d_nom`/`mu_p_nom` (nominal drain rates), `w_star_d`/`w_star_p` (work normalizers), `tau_ttft`/`tau_itl`
(SLO targets) · `ttft_p`/`ttft_d` (predicted TTFT if disagg / if local), `itl_p`/`itl_d` ·
`z_ttft`/`z_itl` (SLO virtual-queue deficits) · `balance_term_d`/`balance_term_p`/`transfer_term`/
`ttft_term`/`itl_term` (the rule's terms) · `lhs`/`rhs` (decision compares these) · `disaggregate` (lhs>rhs).

### `--pd-outcome-trace` (what actually *happened*, one row per request — Stage A)
`request_id, slo_class, input_tokens, disaggregated, prefill_instance, decode_instance` ·
`prefill_enqueue`/`prefill_schedule`/`prefill_t_adm` (realized prefill admission delay = schedule−enqueue) ·
`decode_enqueue`/`decode_schedule`/`decode_t_adm` · `local_enqueue`/`local_schedule`/`local_t_adm`
(non-disagg path; `local_t_adm` currently always 0 — see FINDINGS Stage A limitation) ·
`realized_ttft`/`realized_mean_itl`/`realized_e2e` · `completed` (RequestE2Es>0 proxy — looser than the
sim's completed count; see FINDINGS).

### `--edpp-work-trace` (realized vs closed-form work, one row per request — Stage B)
`request_id, slo_class` · `a_r` (full prompt length), `a_p_realized` (Σ prefill tokens actually
processed, excludes cached prefix), `o_r_realized` (decode steps) · `prefill_chunks` (1 = single-chunk),
`cache_hit_frac` (=1−a_p/a_r) · `realized_prefill_work`/`realized_decode_work` (Σ per-step δ, active
latency-model basis) · `wp_closed`/`wd_closed` (corrected closed forms with **realized** inputs) ·
`wp_closed_nocache_old` (old shipped form, for the correction-effect delta).

---

## 5. How to analyze

`analyze/estimator_validation.py` (Stage A) and `analyze/work_model_validation.py` (Stage B) both take
`--<trace> <csv> [--decision <csv>] [--out <json>] [--plots <png>]` and emit a JSON bias report.

- **Stage A:** `estimator_validation.py --outcome out/stage_a/outcome.csv --decision out/stage_a/decisions.csv`.
  Joins predicted (`ttft_p`/`ttft_d`, reconstructed admission = `qp_raw/mu_p_nom`) vs realized
  (`realized_ttft`, `*_t_adm`), split by `disaggregated`×`slo_class`. Read `median_ratio_real_over_pred`
  (the under-prediction factor) and the p90/p99 tail. Read **completions**, not just TTFT-of-completed
  (overload truncation flatters arms that finish fewer requests).
- **Stage B:** `work_model_validation.py --work out/stage_b/synth_work.csv`. Reports relative error of
  realized-vs-closed prefill (split single-chunk / chunked) and decode work. **"Model exact"** =
  `max_abs_rel_err < 1e-6` for single-chunk prefill and for decode; the chunked residual is the expected
  `C_attn·(a_p²−Σs²)/2` term.

Other scripts: `analyze/window.py` (steady-state window detection), `analyze/report.py` (sweep summary +
plots), `make_specs.py` (generate workload specs), `sweep.sh` (the full grid — bakes once per (workload,
rate), replays across decider×split cells).

---

## 6. Gotchas (learned the hard way — see FINDINGS "Anomalies")

- **`--num-instances N` alone is fatal for PD** — needs role'd instances; use explicit
  `--prefill-instances`/`--decode-instances`.
- **`--edpp-decision-trace` needs `--trace-level decisions`**; `--pd-outcome-trace`/`--edpp-work-trace` do not.
- **Audit the routing config before blaming the algorithm** — an early "collapse" was a round-robin
  misconfig; a "worst decider" verdict was a lagged-`z_ttft` control-law flaw (both retired).
- **`slo_class` valid set:** critical / standard / sheddable / batch / background (NOT "interactive").
- **synth is NOT no-cache** (heavy prefix reuse); **run vs replay decode length can differ** (a
  pre-existing DES quirk) — neither breaks the work-model exactness check (each run is internally consistent).
- **Latency-model basis:** the default trained-physics model over-counts causal attention ~3× vs the
  roofline backend; `W_p` deliberately matches the *active* model. See FINDINGS Stage B §7.

---

## 7. Stage C — occupancy-aware admission-delay estimator (walkthrough)

### 7.1 The problem
EDPP decides disaggregate-vs-local per request; to decide it must *predict* the **admission delay** —
how long the request waits to enter a serving engine — which feeds its TTFT estimate. The shipped
predictor was occupancy-BLIND: admission delay ≈ *waiting-queue work ÷ drain rate* (`qD/muDec`). When
the waiting queue is empty but the running batch is FULL, it predicts ≈0 while reality is "wait for a
running request to finish and free a slot." Stage A measured this at ~905× under-prediction. Stage C
makes the predictor pluggable and adds fidelity levels that model the running batch.

### 7.2 What was built (files)
- `sim/admission_estimator.go` — `AdmissionDelayEstimator.EstimateTAdm(AdmissionContext) float64` (pure
  function) + variants:
  - **deployable** (may drive routing): `waiting` (shipped `QWork/Mu`, the default), `little`
    (Little's law `QueueDepth/AdmissionRate`), `fluid` (occupancy-conditioned mean-field
    `N_ahead/X̂_dep`, `X̂_dep=BatchSize/(RemainingStepsEst·TIter)`), `rollforward` (true per-request
    deterministic batch look-ahead — the proposed method).
  - **oracle** (logging-only, use true `o_r`, forbidden as routing driver by an INV-9 guard in
    `NewEDPPDecider`): `fluid_oracle`, `rollforward_oracle`.
- `sim/edpp.go` — `Decide` calls the estimator for the `ttft_d`/`ttft_p` admission terms (keeping the
  `+compute` part); default `waiting` is byte-identical to pre-Stage-C. Holds the INV-9 guard.
- `sim/routing.go` + `sim/cluster/snapshot.go` — `RoutingSnapshot` enriched with the running decode
  batch (`RunningDecode`, `RemainingDecodeWork`, `AdmissionRate`); populated only when admission detail
  is enabled (zero cost otherwise).
- `sim/trace/admission_csv.go` + `--edpp-admission-trace <path>` — logs realized admission + all six
  predictions per request×pool (so ONE run yields the whole comparison).
- `local_t_adm` capture (closes Stage A's gap for locally-routed requests).

### 7.3 How it is evaluated — isolation microbenchmark
To measure a *predictor* with zero routing noise: one engine per role (1P1D), routing FORCED so the
estimator's value doesn't change where requests go.
- **T1 (isolate the decode-pool admission delay):** force all-local → every request queues on the
  single decode engine; one `local` admission event per request.
- **T2 (isolate BOTH pools):** force all-disaggregate → single P engine + single D engine. Each
  disaggregated request has TWO admission events — a `prefill` sub-req on P and a `decode` sub-req on
  D (after KV transfer) — so T2 validates the estimator on both. The decode-pool wait is part of TTFT
  because the first token comes from the D engine.

> **Naming note.** A `pool` (`local`/`prefill`/`decode`) is an *admission event*, not an instance
> role — see §7.5. The decode-pool admission delay is NOT Stage A's `ttft_d` (which meant "TTFT if
> routed local"); it's the wait for a disaggregated request's decode sub-request to get a batch slot
> on the decode engine. This doc says "decode-pool / prefill-pool admission delay," not `ttft_d`/`ttft_p`.

Routing is forced with the transfer-penalty knob while EDPP still runs (see FAQ Q2): `--edpp-c-xfer
100s` ⇒ all-local (T1); `--edpp-c-xfer 0s` ⇒ all-disagg (T2).

### 7.4 Exact commands (every flag)
Build: `go build -o blis main.go`. T1 (T2 = same but `--edpp-c-xfer 0s` and `t2_admission.csv`):
```
./blis run \
  --model meta-llama/llama-3.3-70b-instruct \                 # calibrated model
  --workload-spec campaigns/edpp-study/specs/synth_rate2.0.yaml \  # synth, aggregate_rate 2.0, 5000 reqs
  --num-instances 2 --prefill-instances 1 --decode-instances 1 \  # 1P1D (num = P+D)
  --pd-decider edpp \                                          # REQUIRED: only edpp runs Decide (Q2)
  --edpp-coeffs scripts/calibration/coeffs-llama70b-h100-tp4.json \  # frozen α/C0/C1/C_pf/C_attn (Q4)
  --edpp-tau-ttft 2s --edpp-tau-itl 150ms \                    # EDPP control-law targets (Q1)
  --slo-ttft "batch=2s" --slo-itl "batch=150ms" \              # goodput yardstick (Q1)
  --edpp-c-xfer 100s \                                         # 100s ⇒ all-local (T1); 0s ⇒ all-disagg (T2)
  --edpp-admission-trace /tmp/t1_admission.csv                 # realized + 6 predictions per request×pool
```
Analysis: `python3 campaigns/edpp-study/analyze/admission_ablation.py --admission /tmp/t1_admission.csv`.
One-command reproduction (writes to `out/stage_c/`): `bash campaigns/edpp-study/repro_stage_c.sh`.

### 7.5 Reading the outputs
**What `pool` is:** an *admission event*, not an instance role. Instances are typed P (prefill-only)
or M (mixed prefill+decode); there is no decode-only instance. A request emits one row per admission
it undergoes: `local` (whole request admitted on an M instance — one event), or `prefill` **+**
`decode` (the two sub-requests of a disaggregated request; the `decode` row is the decode sub-req's
admission onto an M instance after KV transfer). So `pool="decode"` means "decode-side admission of a
disaggregated request," not "a decode-only engine."

`--edpp-admission-trace` columns: `request_id, pool, realized_t_adm, t_adm_pred_{waiting, little, fluid,
rollforward, fluid_oracle, rollforward_oracle}` (µs; `pool` ∈ decode/prefill/local). `admission_ablation.py`
JSON: per pool×estimator `median_ratio_real_over_pred` (>1 = under-prediction — the headline metric) +
`decomposition` (`rollforward` residual split into estimator-*form* error vs `N̂_out`-*prediction* error).

### 7.6 Result (DE-CONFOUNDED — see FINDINGS "Stage C — DE-CONFOUNDED" for the full table + checkpoint)
T1 local pool (saturated, realized p50 ≈ 560s), median ratio realized/pred: **`waiting` 57.3×**
(occupancy-blind — the Stage A mechanism) → **`little` 1.30×**, **`fluid` 1.16×**, **`rollforward`
1.16×** (occupancy-aware estimators fix it; `fluid` ≈ `rollforward` because the per-request walk's
deep-queue behavior collapses to the wave form), **`rollforward_oracle` 1.25×** (oracle ≈ deployable —
the censored `N̂_out` floor closed the gap). T2 disagg decode over-predicts slightly (~0.34×, decode
admitted fast post-transfer). Prefill pool reads ~0 = **correct physics** (unsaturated: `QueueDepth=0`,
free slot), not empty occupancy.

> **History.** An earlier version of this section reported `rollforward` → 1.29× as the headline. That
> number was **oracle-contaminated** (the "deployable" estimator was reading true remaining output via
> the admission-trace's oracle mode) and is **RETRACTED**. The fix-cluster (2026-07-05) separated
> oracle from deployable on both the logging and routing paths, fixed `fluid` (wave form, was ~1e6×
> off), activated `little` (`buildPoolFilteredSnapshots` never populated `AdmissionRate`), and added
> the `--edpp-tadm-estimator` CLI flag. The numbers above are the de-confounded result.

Open follow-ups before the paper's fidelity figure (in FINDINGS): **prefill-saturating validation**
(prefill estimators unproven under a prefill queue), and the parallel **Layer-2** analytical track.

### 7.7 Utilization sweep — fidelity vs load (done; see FINDINGS "Utilization sweep")
`bash campaigns/edpp-study/repro_utilization_sweep.sh` extends the single-point ablation across bounded,
stationary operating points below capacity: it locates λ* (composite saturation detector, coarse scan),
sweeps ρ = {0.5…0.98}·λ* on the T1 topology, and aggregates via `analyze/utilization_sweep.py` (measured
ρ̂ = `responses_per_sec/λ*`; two-layer stationarity = detector verdict + admission-delay drift).
**Result:** the admission-delay curve is a **step function**, not a smooth fidelity curve — a small
(~30–47 ms ≈ one decode step / `T_iter`, tracks ITL), routing-irrelevant floor for all ρ̂ below the
saturation cliff, then the Stage C explosion above it. Below the cliff the occupancy-aware estimators
predict 0 (their free-slot early-return fires (`fluid` at `admission_estimator.go:66`, `rollforward` at `:84`) — a slot is free — so they
model slot/KV wait but not the residual-step wait for the next `FormBatch`). This is why Stage C's
57×→1.16× win is **regime-specific** (heavy overload, where slot-wait dominates) and why the gap is
routing-irrelevant here (floor ≪ τ_ttft = 2 s ⇒ z_ttft never engages on it). Read the **signed error**,
not the ratio, below the cliff (the analyzer's `ratio_meaningful`/`ratio_floor_us` flags this).
**Follow-up DONE (commit 1f2d4bd):** the occupancy-aware estimators (`fluid`/`rollforward`/`little`) are
floored by one `T_iter` (`flooredTAdm=max(est,TIter)`) so they track the sub-saturation delay — the sweep
STABLE band now reads ≈1.05–1.26× (was `predict 0`/NaN); `waiting` stays unfloored as the strawman
(≈0). Stage C overload headline preserved (`waiting` 57×, occupancy-aware ~1.2×); the floor is a per-row
no-op above saturation (routing-irrelevant, ≪ τ_ttft). Spec:
`docs/superpowers/specs/2026-07-06-edpp-admission-floor-design.md`.

## 7.8 Counterfactual-regret harness — per-decision hindsight diagnosis

`bash campaigns/edpp-study/repro_counterfactual.sh` measures the **exact one-step-deviation regret**
of the shipped (reduced) EDPP decider: it captures EDPP's realized per-request `(decode, prefill)`
plan, then re-runs the whole plan with one request's action changed to each alternative and reads the
aggregate-goodput delta. Positive regret ⇒ a single different decision would have improved total
goodput in hindsight.

- **Fixed-plan decider + `--pd-plan`.** The infra is `blis --pd-plan <csv>` (columns
  `request_id,decode_instance,prefill_instance`; empty/`local` prefill ⇒ prefill on the decode
  instance). It **overrides `--pd-decider`** and forces the supplied per-request routing; the plan must
  be *total* (a missing request is fatal — R1, no silent fallback). It reads only the plan (INV-9) and
  is wired on both `run` and `replay` (INV-13). This is the shared substrate for (a) recording/replaying
  any decider's plan, (b) this regret diagnostic, and (c) the later **full-joint decider** and **MILP
  yardstick** (which emit plans the same decider replays).
- **Pipeline.** capture-plan (`analyze/counterfactual_regret.py capture-plan`) → **self-consistency
  gate** (replay the captured plan; `slo_attainment` must equal the baseline, else STOP) → K-sampled
  deviation sweep → `regret` aggregation. Cost `K·(|𝒜|−1)` sims; `|𝒜|=4` at 1P2D.
- **Filename contract (for future deciders reusing this infra).** The `regret` subcommand parses
  deviation files named `dev_<reqid>_<action>.json` with the regex `dev_(.+)_([^_]+)\.json` — it splits
  on the *last* underscore because `request_id`s themselves contain underscores. The **action token must
  therefore be underscore-free**; instance names like `instance_1` are not, so the repro script strips
  the underscore (e.g. `instance1`) when composing dev-file names. Any future full-joint/MILP driver
  emitting dev-files must observe this same underscore-free-action-token convention.
- **Config knobs (env):** `K`, `TARGET_POLICY` (default `edpp`), `SPEC` (default the small saturating
  `specs/synth_cf.yaml`), `OUT`, `MODEL`, `COEFFS`.
- **Driver = occupancy-aware.** The script defaults to `--edpp-tadm-estimator rollforward` — we measure
  EDPP's routing *quality*, so it must route with the occupancy-aware estimator, not blis's default
  occupancy-blind `waiting` (`ESTIMATOR=waiting` reproduces the blind-driver comparison). The
  estimator-BIAS scripts (`repro_stage_a`/`_b`) keep `waiting` on purpose.
- **Result & interpretation:** see FINDINGS "Counterfactual regret". Headline: gate passes; with the
  occupancy-aware driver, baseline goodput rises (0.9775→0.99) and leftover regret roughly halves
  (0.1387→0.06) vs the blind driver, but the residual regret **survives and is still decode-node
  placement** — every positive-regret decision is improved by pinning decode to `instance_1` (local or
  disagg), the choice EDPP delegates to the scorer. Direct motivation for the full-joint rule [C]
  (choose `d` by the drift objective). Diagnostic only (local deviation), **not** the global optimum.

## 7.9 Joint P/D mechanism — reduced vs `--edpp-joint` sweep

`bash campaigns/edpp-study/repro_joint.sh` runs the joint-mechanism study: **correctness gates first**,
then an exploratory topology×workload sweep comparing the shipped **reduced** EDPP decider (fixed decode
`d` from the scorer, local-vs-disagg only) against the **joint** decider (`--edpp-joint`: enumerate all
`(decode, prefill)` candidates, pick the drift-plus-penalty argmin).

- **Gates (pass/fail, stage 1, fail loudly).** (a) two `--edpp-joint`-OFF runs are byte-identical in
  metrics + decision trace (joint plumbing doesn't perturb the reduced default); (b) the §5.5 reduction
  unit test (`go test ./sim -run TestJoint_ReducesToScorerSliceMatchesReduced`); (c) the joint plan
  captured from `--edpp-joint-trace` replays via `--pd-plan` to the joint baseline goodput exactly. Note
  the joint plan is captured from the **joint trace**, not `--pd-outcome-trace` — the outcome trace
  leaves `decode_instance` empty for local requests, which replays faithfully for reduced but not for
  joint (whose point is to override the *local* decode). See FINDINGS gate (c) caveat.
- **Sweep.** cells = {1P2D, 2P2D} × {`synth_cf` cache-uniform, `synth_asym` cache-asymmetric (unique
  large prompts, no shared prefix — generated by `make_specs.py`; both regenerated automatically if
  `specs/` is absent since `specs/` is gitignored)}. Per cell/policy: run with `--pd-outcome-trace` +
  `--metrics-path` (+ `--edpp-joint-trace` for joint), then `counterfactual_regret.py` for one-step regret
  and `joint_divergence.py` for the d/p divergence rate + direction. Small `K` (default 4) bounds runtime.
- **Divergence analyzer.** `analyze/joint_divergence.py summary --trace <joint-trace.csv>` reports
  d/p-divergence rate and, on divergent rows, the direction — share where joint picked a strictly
  **lower-J** candidate vs a **J-tie** (deterministic lower-index/lower-occupancy break); `dir_higher_J`
  must be ~0 (argmin invariant). A float-dust tolerance keeps ~1e-21 summation noise classified as a tie.
  Self-test: `joint_divergence.py selftest` or `analyze/test_joint_divergence.py`.
- **Result & interpretation:** see FINDINGS "Joint mechanism (sub-project 1)". Headline: gates pass;
  joint **cuts reduced-EDPP's decode-placement regret ~25% on cache-uniform** (confirming the regret
  hypothesis) while shaving a hair of goodput, and **ties-to-loses on cache-asymmetric** (reduced already
  optimal at the loose SLO). Its prefill-rerouting lever only fires at 2P2D under cache asymmetry
  (`p_div=0.26`). Homogeneous-hardware LOCAL diagnostic; per-instance `θ_i` heterogeneity deferred.

## 8. FAQ (why things are the way they are)

**Q1. Why are there separate `--slo-*` and `--edpp-tau-*` flags?** They are mechanically distinct.
`--slo-ttft`/`--slo-itl` are the **measurement yardstick** — they define goodput (a request is "good"
if realized TTFT/ITL ≤ target), apply to *every* decider, and affect only reported metrics, never
routing. `--edpp-tau-ttft`/`--edpp-tau-itl` are an **EDPP control-law parameter** — the τ normalizers
inside the drift-plus-penalty rule (`z_ttft`/`z_itl` accumulate deficit relative to τ; term scalings
divide by τ), shaping *what the policy optimizes*. Keeping them separate lets you score against the
real SLO while the policy chases a different internal target, and lets non-EDPP deciders be scored
uniformly. In these studies τ is set equal to the SLO by choice. (Per-class variant: `--edpp-tau-*-classes`.)

**Q2. Why must the study use `--pd-decider edpp`, not `always`/`never`?** The estimator predictions and
the `AdmissionContext` they read are computed inside EDPP's `Decide()`. `always`/`never` are separate
code paths that force routing WITHOUT running EDPP's drift-rule machinery, so they assemble no context
and `--edpp-admission-trace` would log nothing. We need both forced routing AND context assembly, so we
run `edpp` and force the decision with `--edpp-c-xfer` (100s ⇒ local, 0s ⇒ disagg).

**Q3. If the system is saturated, the queue grows unboundedly, so `t_admit` keeps increasing — is the
measurement valid?** Correct: under overload there is no steady-state `t_admit`; each request waits
longer than the last. This is handled three ways, with one honest limitation: (a) the harness compares
**per-request, conditional on the batch state at that request's decision instant** — not against a
steady-state average — so a growing delay is the signal, not a bug (this is also *why* the conditional
`rollforward` tracks it while the aggregate `little` cannot — there is no steady-state mean for
`little` to use); (b) the workload is finite (5000 requests), so saturation is transient/bounded, not
infinite; (c) LIMITATION: a forced-overload point is a deliberate stress test over a non-stationary
transient, so the "median ratio" summarizes a moving target. The rigorous complement — a **utilization
sweep approaching but below capacity**, where admission delay is large but bounded/stationary — is a
noted follow-up (see FINDINGS) before claiming steady-state accuracy.

**Q4. The Stage B work equations changed — do the frozen `--edpp-coeffs` still hold?** Yes, by design,
not luck. The coeffs (α, C0, C1, C_pf, C_attn) were fit to the latency model's per-iteration law
`T_iter = α + C0·B_dec + C1·KV + C_pf·S_pf + C_attn·pf_ctx`. Stage B changed the *form* of `W_p`/`W_d`
but deliberately chose it to be the **trajectory integral of the exact same per-step δ the coeffs
describe** (that was the "Option 1" decision — the trained-physics `+a_p/2` basis, matching the
`nt·(si+nt/2)` the latency model charges and `C_attn` was fit to). No refit. And Stage B *proved* it:
realized per-request work matched the closed-form `W_p`/`W_d` to float precision (~5e-16) using these
exact frozen coeffs. Stage C's `rollforward` builds `T_iter` from the same coeffs, inheriting that
validity. Caveat (fidelity, not validity-within-sim): the coeffs are tied to the trained-physics model,
which over-counts causal attention ~3× vs roofline/physical; a roofline or real-server backend would
recalibrate `C_attn`. See FINDINGS Stage B §7.
