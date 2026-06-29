# EDPP Empirical Study — Findings

Canonical findings doc. Detail/lab-notebook: `out/diag/SESSION_LOG.md`. Headline table:
`out/diag/SUMMARY.md`. Backlog: `TODO.md`. Repro: `out/diag/REPRO.md`.
Superseded round-robin artifacts: `out/diag/ARCHIVE_round-robin-artifact.md` (do not cite).

EDPP is a Lyapunov **time-average** optimizer → report MEANS alongside p99 (means are arguably what it
targets). All results below use the llm-d **weighted PD default** (`precise-prefix-cache:2,queue-depth:1`).

## Framing — two distinct questions
- **Q1: should this workload run disaggregated at all?** Provisioning/topology choice; EDPP doesn't
  control it. Comparing EDPP to `never@4` answers Q1, not EDPP's quality.
- **Q2: given a disaggregated deployment, does EDPP disaggregate the RIGHT requests?** The algorithm's
  actual correctness — baselines hold topology FIXED (always / never-in-split / oracle).
- Everything below answers **Q1** for uniform decode-bound synth. **Q2 is NOT yet determined — priority.**

## Workload: synth (decode-bound), rate 2.0, 2000 reqs, equal 4-node HW

KEY RESULTS (weighted default; full table with means in `out/diag/SUMMARY.md`):
- **`never@4` wins** (goodput 1.68 rps). Disaggregation does NOT help this decode-bound workload at
  equal hardware. Outcome tracks DECODE-CAPABLE NODE COUNT monotonically: never@4(4) > *@1P3D(3) >
  *@2P2D(2) > *@3P1D(1) on goodput AND completion. **(Q1: don't disaggregate; add decode nodes.)**
- **EDPP is the WORST decider** at every split (TTFT mean 56s / p99 518s @2P2D; SLO 0.75). It keeps
  ~48–51% of requests LOCAL; on a saturated decode pool those locals HOL-block (their prefill queues
  behind running decode). `always`/`prefix-threshold` disaggregate 100% → no locals → fast TTFT (the
  decode-queue wait appears only in E2E). EDPP's PARTIAL disaggregation actively hurts here.
- 3P1D (1 decode node) is the only REAL saturation collapse: always/prefix complete 1192/2000 with
  10,157 preemptions (nothing to balance).
- `prefix-threshold` ≈ `always` on synth (tiny inputs trip threshold-16 → ~100% disagg).

## WHY EDPP keeps ~half local (RESOLVED) — HOL-blind local-TTFT predictor
`ttft_d` (predicted TTFT if local) is built from the decode WAITING-backlog (`qd`) + nominal μ; it
ignores RUNNING decode occupancy. So on transient `qd` dips EDPP predicts ~2s, keeps a request local,
and it then waits 24–542s behind running decodes. Predicted-vs-realized (edpp@2P2D):
kept-local realized TTFT mean **117.8s** / p99 **541.9s** vs predicted p99 14.7s (174× under); the
disaggregated side is predicted conservatively. The fix candidate (TODO #10): make `ttft_d`
running-occupancy/KV-aware.

## Load-dependence (knee)
`never@4` healthy across rates 0.5–3.0 (TTFT mean ≤0.09s). EDPP@2P2D harmless ONLY at rate 0.5
(0% disagg); SHARP CLIFF at rate 1.0 — TTFT mean 0.07s→15.5s (p99 162s) with just 7.2% disagg = the
2-decode-node saturation onset. (Full table in SESSION_LOG.)

## ITL decision path
Tightening `τ_itl` 150→50ms first ACTIVATED `z_itl` (388 decisions). EDPP correctly disaggregated more
(52→58%) but ITL mean stayed 72ms — disaggregation moves only PREFILL; decode stays on the same nodes,
so ITL is floored by decode capacity. **EDPP's sole lever (prefill placement) is matched to TTFT/HOL
but mismatched to ITL/decode-capacity.**

## Anomalies & instrument audits (kept)
- The round-robin "collapse/161s/1192" numbers were a harness misconfiguration (default
  `--routing-policy round-robin` + no per-pool scorers → decode pool unbalanced). Fixed by defaulting
  PD pools to llm-d weighted. Full episode archived. **Lesson: audit the routing config (the "knob")
  before blaming the algorithm.**
- `--num-instances N` alone is fatal for PD (needs role'd instances) — use explicit P/D splits.
- per-request ITL is recorded unconditionally on replay (`itl_mean_us`); `--record-itl` is observe-only.
- `slo_class` valid set: critical/standard/sheddable/batch/background (NOT "interactive").

## Open (priority: Q2)
EDPP's per-request decision correctness is undetermined. See `TODO.md`: mine the decision/routing
traces we already produced (per-term lhs/rhs across load, predicted-vs-observed across load, routing
queue/latency/scores), then a heterogeneous favorable-regime workload + oracle baseline.
