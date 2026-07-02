# Stage A — EDPP Estimator Validation Harness (Design)

Date: 2026-07-01
Branch: `design/pd-joint-routing`
Related: `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (§3.8, §5.4);
`docs/design/2026-07-01-joint-routing-session-handoff.md` (Stage A); memory
`edpp-pd-routing-reservation-gap` (the "predicted 14.7s vs realized 542s" under-prediction).

## 1. Goal & scope

Quantify the bias of the **currently shipped** EDPP forward TTFT estimators (`ttft_p`, `ttft_d`,
computed at `sim/edpp.go:420-421`) against realized outcomes, per request, on a disaggregated
workload.

This is Stage A of the joint-routing experimental plan and is **measurement-only**: it does **not**
change the decider, the estimator, or any routing behavior. Its output is a predicted-vs-realized
dataset plus a bias report. It establishes the baseline that Stage C (occupancy-aware admission-delay
roll-forward) will later be re-measured against, and it tests the guarantee's one soft dependency
(§5.4: the forward-estimator bias).

Non-goals (explicitly deferred):
- Implementing the §3.8 occupancy-aware roll-forward `T̂` (Stage C).
- Any heterogeneous workload comparison or baseline bake-off (Stage D).
- Changing `m_pf`/`m_dec` or the ITL responsiveness path (open items in the handoff).

## 2. Data captured — new `--pd-outcome-trace <path>` CSV

One row per request, sorted by `request_id` at emit (INV-6 determinism). All times in microseconds,
absolute. Zero means the phase was not reached (mirrors the existing `ParentRequest` timestamp
convention).

| column | source |
|---|---|
| `request_id` | join key to `--edpp-decision-trace` |
| `slo_class` | `req.SLOClass` |
| `input_tokens` | `len(req.InputTokens)` (`a_p`; lets analysis bucket bias by prompt size without a third join) |
| `disaggregated` | parent exists and request took the disaggregated path |
| `prefill_instance`, `decode_instance` | `ParentRequest.PrefillInstanceID` / `DecodeInstanceID` (empty for non-disagg) |
| `prefill_enqueue`, `prefill_schedule`, `prefill_t_adm` | `PrefillEnqueueTime`, **new** `PrefillScheduleTime`, and their difference |
| `decode_enqueue`, `decode_schedule`, `decode_t_adm` | `DecodeEnqueueTime`, **new** `DecodeScheduleTime`, and their difference |
| `local_enqueue`, `local_schedule`, `local_t_adm` | single-admit path for non-disagg (local) requests |
| `realized_ttft` | `Metrics.RequestTTFTs[id]` (relative first-token time) |
| `realized_mean_itl` | `Metrics.RequestITLs[id]` |
| `realized_e2e` | `Metrics.RequestE2Es[id]` |
| `completed` | terminal state flag (truncated/dropped requests remain visible — "read completions" lesson) |

`t_adm` columns are `schedule − enqueue`; emitted only when both endpoints are non-zero, else left
zero and flagged by `completed=false` / empty instance.

## 3. Instrumentation points

- **`ParentRequest`** (`sim/cluster/parent_request.go`): add `PrefillScheduleTime int64` and
  `DecodeScheduleTime int64` (zero = not yet scheduled), documented alongside the existing phase
  timestamps.
- **`OnAdmit` correlation** (existing `feedAdmission` path, `sim/cluster/cluster.go`): when an
  admitted sub-request ID ends in `_prefill` / `_decode`, set the parent's corresponding schedule
  time to the admit instant. For a non-disaggregated (`local`) request, record its single admit time
  in a lightweight `map[string]int64` owned by the cluster, populated **only** when the outcome trace
  is enabled (no cost on the default path).
- **Emit** (run end, gated on a new config field `RecordPDOutcomes`, set by `--pd-outcome-trace`):
  new writer `sim/trace/pd_outcome_csv.go`, mirroring `sim/trace/edpp_csv.go`. Available on both
  `run` and `replay` (INV-13 parity — `replay` must produce identical rows).
- **No decider changes.** `sim/edpp.go` is read, not modified.

## 4. Analysis — new `campaigns/edpp-study/analyze/estimator_validation.py`

Joins `pd-outcome-trace ⋈ edpp-decision-trace` on `request_id`. Computes, split by `disaggregated`
and `slo_class`, over **completed** requests (with a separate count of truncated/dropped):

- `ttft_p` (predicted, disagg path) vs `realized_ttft` on disaggregated requests; `ttft_d` vs
  `realized_ttft` on local requests — the term-by-term comparison the decider actually uses.
- Predicted admission component vs realized `prefill_t_adm` / `decode_t_adm` / `local_t_adm`
  (the §3.8 target). The shipped estimator does not emit the admission component as its own column;
  the analysis reconstructs it as the queue-wait term (`qp_raw / mu_p_nom` for the prefill side,
  `qd_raw / mu_d_nom` for the decode side) from the existing `--edpp-decision-trace` columns.
- Bias metrics: mean/median signed error, ratio (realized / predicted — the "174×" figure), p50/p90/p99
  of both predicted and realized, bucketed by `input_tokens`.

Output: a structured **JSON bias report** (always) plus the same table to stdout — diffable, testable,
CI-friendly, and pinnable as a regression anchor. An optional predicted-vs-realized **scatter/quantile
PNG** behind a `--plots` flag (presentation only, not a validation gate).

## 5. Testing & invariants

- **Unit:** `OnAdmit` correlation sets `PrefillScheduleTime` / `DecodeScheduleTime` on the correct
  parent for a synthetic disaggregated request; the local path records the single admit time.
- **Invariant (INV-5 extension):** `enqueue ≤ schedule ≤ completion` for both sub-requests;
  `t_adm ≥ 0` wherever both endpoints are set.
- **INV-13 parity:** run with `--trace-output`, then `replay` with `--pd-outcome-trace` → byte-identical
  outcome CSV.
- **INV-6 determinism:** identical CSV across two seeded runs.
- **Regression:** full `go test ./...` green; `golangci-lint run ./...` clean.
- **Sanity anchor:** re-run the exact synth edpp@2P2D rate-2.0 operating point from
  `out/diag/REPRO.md` and confirm the harness reproduces the known large under-prediction (~174×).
  If the harness disagrees with the archived figure, the harness is wrong — not the estimator. A fresh
  heterogeneous spec is deliberately deferred to Stage D so instrumentation and workload are not two
  unknowns at once.

## 6. Deliverables

1. `PrefillScheduleTime` / `DecodeScheduleTime` on `ParentRequest` + local-admit map, populated via
   `OnAdmit`.
2. `--pd-outcome-trace` flag (run + replay), `RecordPDOutcomes` config field, `pd_outcome_csv.go`
   writer.
3. `campaigns/edpp-study/analyze/estimator_validation.py` (JSON report + optional `--plots`).
4. Tests (unit, invariant, INV-13, INV-6) and a documented anchor run reproducing the REPRO.md figure.
