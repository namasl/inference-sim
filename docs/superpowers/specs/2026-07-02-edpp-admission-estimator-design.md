# Stage C — Occupancy-Aware Admission-Delay Estimator + Isolated Fidelity Ablation (Design)

Date: 2026-07-02
Branch: `feat/edpp-estimator-validation` (Stage C continues here after Stage B)
Related: `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (§3.7–3.8 admission delay);
`docs/superpowers/specs/2026-07-01-edpp-estimator-validation-design.md` (Stage A harness, reused);
`docs/superpowers/specs/2026-07-01-edpp-work-model-design.md` (Stage B work model, consumed by the
roll-forward). Layer-2 closed-form analysis is a SEPARATE design doc (parallel track, not this spec).

## 1. Goal & scope

Replace EDPP's occupancy-blind admission-delay estimate with a **pluggable estimator** offering four
fidelity levels, and measure each per-pool on **minimal, low-noise topologies** that isolate the
estimator against a single-engine realized admission delay. Layer-1 (observable, arrival-free) only.

This is the estimator whose ~905× decode-admission under-prediction Stage A measured. It builds on the
Stage-B-corrected work model (the roll-forward simulates the batch draining via the per-iteration
identity `T_iter = α + Σδ`).

Non-goals (deferred): the closed-form Layer-2 analysis (separate doc); the full Q2 baseline comparison
on a heterogeneous workload (Stage D); refitting coefficients; changing the latency model.

## 2. The seam being replaced

Today (`sim/edpp.go:420-421`): `ttft_d = qD/muDec + compute`, `ttft_p = qP/muPf + compute`, where the
`qD/muDec` and `qP/muPf` terms are the admission-delay estimates — **waiting-backlog work ÷
occupancy-aware rate**, blind to the RUNNING batch that holds the slots/KV. When waiting work ≈ 0 but
the batch is full, these give ≈ 0 while reality is the wait for a running request to finish (the ~905×
miss). Stage C replaces **only these two terms** with `estimator.EstimateTAdm(poolState)`, keeping the
`+compute` term (`nChunks·(T_iter+δ)`). The same interface serves the admission term inside both
estimators: the decode-side estimate `ttft_d` uses the **decode-pool** admission delay, and the
prefill-side estimate `ttft_p` uses the **prefill-pool** admission delay. (Note: `ttft_d`/`ttft_p`
here are code symbols in `sim/edpp.go`; the *pools* they consume are the admission-event axis defined
in §5, NOT the local-vs-disagg routing alternatives those symbols denoted in Stage A.)

## 3. Estimators

A single-method Go interface in `sim/` (e.g. `AdmissionDelayEstimator.EstimateTAdm(ctx) float64`),
where `ctx` carries the pool/instance snapshot + this request's KV need + the coeffs. Four **deployable**
levels + two **measurement-only** oracle variants:

Deployable (INV-9-safe; may drive routing):
- **`waiting`** — `qWork/mu` (extracted current formula). The strawman; reproduces today exactly.
- **`little`** — `T_adm ≈ L̄_q / λ_adm` from `QueueDepth` and a measured per-instance admission rate.
  Aggregate, arrival-process-free (holds for any ergodic system); lags on transients.
- **`fluid`** — occupancy-*conditioned* mean-field `T_adm ≈ N_ahead / X̂_dep`, with
  `X̂_dep = B / (R̄ · T_iter)` computed from the current snapshot (`B = BatchSize`, `R̄` = mean
  estimated remaining decode steps, `T_iter` = occupancy-aware per-iteration time). If a slot and KV
  already fit (`BatchSize < MaxBatchSize` and `FreeKVBlocks ≥ need`) → ≈ 0. This is the Layer-1
  counterpart of the Layer-2 birth–death first-passage closed form.
- **`rollforward`** — the **true per-request deterministic** look-ahead: simulate the current batch
  iteration by iteration (each iteration `T_iter`, running requests depart after their estimated
  remaining decode steps, freeing their KV blocks) until a batch slot AND enough free KV exist for this
  request's prompt; accumulate elapsed time. Highest fidelity.

Measurement-only (read true `o_r`; MUST NOT drive routing — see §7 INV-9):
- **`fluid(oracle)`**, **`rollforward(oracle)`** — same math as `fluid`/`rollforward` but using the
  running batch's **true** remaining steps (`o_r − progress`) instead of the `N̂_out` estimate. Used only
  to decompose estimator-form error from output-length-prediction error (§6).

Remaining decode steps for the deployable variants use `N̂_out − decode_progress` (INV-9-safe estimate,
the per-class running mean EDPP already maintains).

## 4. Snapshot enrichment

`RoutingSnapshot` gains, populated in the instance→snapshot path (INV-7 Periodic tier, same as
`BatchSize`), gated so runs without an active/logged occupancy-estimator pay nothing:
- `RemainingDecodeWork float64` (or steps + count) — `Σ over running decode reqs of (N̂_out − progress)`.
  Feeds `fluid`. INV-9-safe (estimate).
- For `rollforward`: the running batch's per-request `(remainingSteps, kvBlocksHeld)` — a compact slice
  (length ≤ `MaxBatchSize`). Affordable because the study topologies are single-instance; gated on the
  roll-forward estimator being active or logged.
- The **oracle** variants additionally need true remaining `o_r − progress` per running request; this is
  a measurement-plane read of `OutputTokens` (permissible — see §7), populated only when oracle logging
  is enabled.

## 5. Isolation experiments (the microbenchmark)

Both topologies **force routing** (no per-request choice), so the estimator is a *measured* quantity,
not a control input. This lets us **log all six variants side-by-side in one run per topology** against
the same executions.

**Terminology (read this — the symbols are not what Stage A meant).** `pool` here is an *admission
event*, not an instance role. A request incurs one admission per pool it passes through: `local` (a
non-disaggregated request admitted on a mixed instance, one event), or `prefill` **then** `decode`
(the two sub-requests of a disaggregated request — prefill on the P instance, decode on the mixed
instance after KV transfer). The **decode-pool admission delay** is a real component of a
disaggregated request's TTFT because the first token is emitted by the decode instance, so its
decode sub-request must win a batch slot there. This is DISTINCT from Stage A's `ttft_d`/`ttft_p`,
which were *whole-path* TTFT estimates for the local vs disaggregated routing alternatives. Below we
say **decode-pool** / **prefill-pool admission-delay estimate**, never `ttft_d`/`ttft_p`.

- **T1 — decode-pool isolation:** a single decode-capable instance, `--pd-decider never` (never
  disaggregate) → every request prefills+decodes on that one engine. Measures the **decode-pool
  admission-delay estimate** vs the **realized local admission delay** on that engine.
- **T2 — both pools, disagg path:** `--prefill-instances 1 --decode-instances 1 --pd-decider always`
  → every request prefills on the single P, transfers, decodes on the single D. A disaggregated
  request has TWO admission events, so this measures BOTH: the **prefill-pool admission-delay
  estimate** vs realized **prefill** sub-req admission, AND the **decode-pool admission-delay
  estimate** vs realized **decode** sub-req admission (the decode sub-request still queues for a
  slot on the D engine — that wait is part of TTFT).

One decode-capable engine / one prefill engine, no balancing → predicted-vs-realized gap is *purely*
estimator quality.

### 5b. Instrumentation: realized local admission delay (closes a Stage A gap)

T1 needs the realized admission delay of **locally-routed** requests, which Stage A left at 0
(`local_t_adm` unimplemented, its documented limitation #1). Stage C captures the local enqueue instant
(reuse the `OnAdmit` hook / `GatewayEnqueueTime`) so `local_t_adm = local_schedule − local_enqueue`.
T2's prefill/decode sub-req `t_adm` already work from Stage A.

### 5c. Per-request estimator-prediction logging

A **new companion trace** `--edpp-admission-trace <path>` (not the Stage A outcome trace — keep that
artifact stable) emits one row per request: `request_id`, `pool` (decode/prefill/local), the realized
admission delay, and the admission-delay **prediction** of every variant at its routing instant —
`t_adm_pred_{waiting, little, fluid, rollforward, fluid_oracle, rollforward_oracle}`. Keyed by
`request_id` for joining. Deterministic, sorted (INV-6). In T1/T2 these are pure measurements (routing
forced). Mirrors the Stage A/B trace plumbing (writer in `sim/trace/`, run+replay wiring, gated on the flag).

## 6. Validation — the fidelity ablation

`repro_stage_c.sh` runs T1 and T2 (one run each; all variants logged) and an analysis script emits an
ablation table with, per pool (decode from T1+T2, prefill from T2):
- predicted-vs-realized admission-delay **median ratio** per variant (the ~905× for `waiting`),
- the **error decomposition**: `realized − rollforward(oracle)` = estimator-form error;
  `rollforward(oracle) − rollforward(N̂)` = `N̂_out`-prediction error,
- TTFT-SLO attainment at the operating point.

**Success:** the median ratio collapses monotonically `waiting → little → fluid → rollforward`, ideally
to ~1× for `rollforward(oracle)` on both pools, with the deployable `rollforward(N̂)` close behind and
the residual attributed by the decomposition. Recorded in FINDINGS "Stage C".

Operating point: a **saturating** load (e.g. synth@2P2D rate 2.0 analog at the T1/T2 topologies) so the
running batch is genuinely full and admission delay is large — otherwise every estimator reads ≈ 0 and
the ablation is flat. (Note `T̂` only influences real routing where `z_ttft > 0`; the microbenchmark
sidesteps this since routing is forced, but the operating point must still saturate to exercise the
estimators.)

## 7. Invariants & guards

- **INV-9 (oracle boundary):** the `--edpp-tadm-estimator` flag that *drives routing* accepts only
  deployable variants (`waiting|little|fluid|rollforward`); selecting an oracle variant as the routing
  driver is a fatal config error. Oracle predictions are computed in the **measurement plane** (logged,
  never fed to a routing/servability decision) and only when oracle logging is enabled — so reading
  `o_r` there is within the "only execution/measurement may read `OutputTokens`" boundary.
- **INV-6 determinism:** estimators are pure functions of snapshot + coeffs; trace rows sorted by
  `request_id`; no RNG.
- **INV-13 run/replay parity:** any new trace columns identical across run and replay for the same config.
- **Default `waiting`** → the decider is byte-identical to pre-Stage-C (regression guard; legitimate
  because the default preserves behavior — the other estimators change decisions by design).
- **INV-7:** new snapshot fields are Periodic-tier (batch-occupancy), consistent with `BatchSize`.

## 8. Testing

- Estimator unit tests on hand-built pool states: `waiting` reproduces the pre-change formula exactly;
  the **full-batch/zero-waiting** case (the exact ~905× bug) yields ≈ 0 for `waiting` but a large,
  correct `T_adm` for `fluid` and `rollforward`; `rollforward` matches a hand-computed multi-step
  departure schedule; `little` matches `L̄_q/λ`.
- Oracle vs deployable: on a batch with known true vs estimated remaining, `rollforward(oracle)` and
  `rollforward(N̂)` differ by exactly the injected `o_r` gap.
- INV-9 guard test: selecting an oracle variant as the routing driver is rejected.
- `local_t_adm` capture unit test (T1 path); INV-6 determinism; INV-13 parity on new columns.
- Full `go test ./...` green; `gofmt`/`go vet` clean (golangci-lint in CI).

## 9. Deliverables

1. `AdmissionDelayEstimator` interface + `waiting`/`little`/`fluid`/`rollforward` implementations
   (`sim/`), with oracle logging variants; `--edpp-tadm-estimator` flag (deployable-only for routing).
2. `RoutingSnapshot` enrichment (`RemainingDecodeWork`; per-request roll-forward slice; oracle remaining),
   gated; instance→snapshot population.
3. Decider seam swap in `sim/edpp.go` (`ttft_d`/`ttft_p` admission terms → estimator), default `waiting`.
4. `local_t_adm` capture (§5b) + per-request estimator-prediction logging (§5c).
5. `repro_stage_c.sh` (T1 + T2) + analysis producing the ablation table with error decomposition;
   FINDINGS "Stage C" section.
6. Tests per §8.

## 10. Known limitations (paper-relevant)

Carries Stage B's trained-physics attention over-count (the roll-forward `T_iter` uses the same coeffs;
a roofline/real-server pass is the fidelity follow-up). The ablation is a single-operating-point
microbenchmark on minimal topologies — deliberately, to isolate estimator quality; broader
operating-point/topology sweeps and the deployed-routing impact are Stage D. The closed-form Layer-2
analysis (birth–death first-passage `T̂`, the fluid `N_ahead/X̂_dep` form, and the estimator-bias →
optimality-gap bound) is a separate design doc, validated numerically against this stage's curves.
