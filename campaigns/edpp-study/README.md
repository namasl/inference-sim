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
