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

## NOT yet re-measured under the fix (pre-fix numbers retired)
The following were measured under the lagged-`z_ttft` binary and need re-running (TODO):
- **Load-dependence (knee)** across rates 0.5–3.0 (the old "sharp cliff at rate 1.0").
- **ITL decision path** (tightening τ_itl; `z_itl` activation) — and note `z_itl` has the SAME
  completion-lag flaw `z_ttft` just had (TODO: apply the responsive-update treatment to `z_itl`).
- **Time-average MEANS** table in `SUMMARY.md`.

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

## Open (priority: Q2)
EDPP's per-request decision correctness on a workload where requests DIFFER is still undetermined
(uniform synth gives no per-request signal). See `TODO.md`: heterogeneous favorable-regime workload +
oracle baseline; apply the responsive-update fix to `z_itl`; re-measure the load knee / ITL / means
under the fix.
