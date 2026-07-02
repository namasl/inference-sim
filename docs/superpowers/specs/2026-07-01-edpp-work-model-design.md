# Stage B — Corrected EDPP Work Model + Per-Request Validation (Design)

Date: 2026-07-01
Branch: `feat/edpp-estimator-validation` (Stage B continues here after Stage A)
Related: `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (§3.6 work model);
`docs/design/2026-07-01-joint-routing-session-handoff.md` (Stage B);
`docs/superpowers/specs/2026-07-01-edpp-estimator-validation-design.md` (Stage A harness, reused pattern).

## 1. Goal & scope

Two coupled deliverables:
- **(a) Correct** the EDPP work-model formulas so they equal the work the **active latency model
  actually charges** per request (`W_p` gains the cache/context cross term; `W_d` becomes the exact
  decode trajectory sum). This is the §3.6 *direction*, but tied to the simulator's basis rather than
  §3.6's physics-pure form — see §2 and the §7 fidelity note.
- **(b) Validate** them empirically against **per-request realized work** — the "perfect before Stage C" bar.

Same frozen coefficients (`C0, C1, C_pf, C_attn, α`) — no refit (design §3.6 is explicit). INV-9 is
untouched: the *decider* keeps using estimates (`N̂_out`, cache-adjusted `a_p`); the *validation*
harness uses realized `a_r, a_p, o_r` only in post-hoc analysis, never in a routing decision.

Non-goals (deferred): the occupancy-aware admission-delay roll-forward `T̂` (Stage C); re-fitting
coefficients; changing `m_pf`/`m_dec`.

## 2. The two corrections

Current shipped forms (verified):
- `Wp(a_p) = C_pf·a_p + (C_attn/2)·a_p²` (`sim/edpp_coeffs.go:62`) — the no-cache limit.
- Decode work in `OnRoute` (`sim/edpp.go:564`): `N̂_out · (C0 + C1·NomDecodeCtx)` — fixed nominal
  context (default 2048), linear in output length, ignoring `a_r` and trajectory growth.

Corrected forms:
- **`Wp(a_p, a_r) = C_pf·a_p + C_attn·a_p·(a_r + a_p/2)`.** New signature takes `a_r`. This is the
  trajectory sum of the **active latency model's actual per-step prefill charge** — the default
  trained-physics model (`sim/latency/trained_physics_model.go:181`) charges attention as
  `ti·(si + ti/2)` with `si = len(InputTokens)` (full prompt), and `C_attn` was calibrated against
  exactly this basis (`scripts/calibration/fit_coeffs.py`). Note the **`+ a_p/2`** (not §3.6's
  `− a_p/2`): this matches the simulator we run and its fitted `C_attn`, so per-request validation (§4)
  hits ~0 error. **Basis caveat:** the roofline backend uses the physically-causal prior-context basis
  (`≈ a_r − a_p/2`); `W_p` here is tied to the **active** model, which is trained-physics by default.
  The trained-physics basis over-counts causal attention (~3× at single-chunk) — a documented
  latency-model fidelity gap (see §7), NOT corrected in Stage B.
- **`Wd(a_r, o) = C0·o + C1·o·(a_r + (o−1)/2)`** — the **exact discrete per-step sum**
  `Σ_{k=0}^{o−1}(C0 + C1·(a_r + k))`. This already matches the active model: trained-physics accrues
  decode context as `sumCtx += ProgressIndex` per decode step, and at decode step `k` a request's
  `ProgressIndex = a_r + k`, so per-step decode work `= C0 + C1·(a_r + k)` sums to the form above —
  agreeing with the realized accumulation (§4) to float precision. `o = N̂_out` (per-class running
  mean) in the decider (INV-9-safe: `a_r` known at routing, `o` estimated). §3.6 states a continuous
  `o·(a_r + o/2)`; this design uses the exact discrete sum. Reconcile §3.6 (both the `W_p` sign and the
  `W_d` discreteness) in a later design-doc pass (§7).

`NomDecodeCtx`: **kept** as a config field (still consumed by the `selectedDecodeState` fallback,
`sim/edpp_coeffs.go:477`). `W_d` no longer routes through it; document that at the `OnRoute` call site.

Call sites: `Decide` (`sim/edpp.go`) and `OnRoute` updated to pass `a_r = len(req.InputTokens)`
alongside the cache-adjusted `a_p`. The drift terms `balance_term_d`/`balance_term_p` consume the
corrected `W_p`. No other decision-rule change.

## 3. Per-request work instrumentation

- A per-request work accumulator keyed by request ID: `map[string]struct{ prefillWork, decodeWork float64 }`,
  owned by the `InstanceSimulator`, allocated and written **only** when a new gate is set (mirrors the
  `BLIS_STEP_CSV` zero-cost-when-off contract; no allocation / no writes on the default path).
- Updated once per engine step inside `executeBatchStep`, in the existing loop over
  `RunningBatch.Requests`, adding each request's exact per-step δ — **mirroring the active
  (trained-physics) latency model's per-step charge verbatim** (`trained_physics_model.go:178-185`),
  so the sum is the realized trajectory work by construction:
  - **Prefill** (this step processes `s_r = NumNewTokens` new tokens; full prompt `a_r = len(InputTokens)`):
    `+= C_pf·s_r + C_attn·s_r·(a_r + s_r/2)`. Note it uses the **full** `a_r` (matching the latency
    model's `si`), NOT the prior context. Summed over chunks reconstructs `W_p`; single-chunk prefill
    reconstructs the closed form exactly, chunked prefill differs by `C_attn·(a_p² − Σs_r²)/2` (the
    validation buckets by chunk count and reports this gap explicitly).
  - **Decode** (step at position `k`, `ProgressIndex = a_r + k`): `+= C0 + C1·(a_r + k)`. Summed over
    the `o` decode steps reconstructs the exact discrete `W_d`.
  Uses the frozen coeffs; identical arithmetic to the active latency model, so the sum is the realized
  trajectory work by construction. If a run uses the roofline backend instead, the prefill basis
  differs (causal, prior-context) — the accumulator reads the active model's basis, and the validation
  notes which model was active.
- Emission: at request completion, one row per request via a new `--edpp-work-trace <path>` CSV
  (available on `run` and `replay`; INV-13 parity, INV-6 determinism — rows sorted by `request_id`),
  gated on a `RecordWorkTrace` config field. Mirrors Stage A's `--pd-outcome-trace` plumbing
  (flag var + registration in `cmd/root.go`, shared write helper, replay mirror).
- Columns: `request_id, slo_class, a_r, a_p_realized, o_r_realized, cache_hit_frac,
  realized_prefill_work, realized_decode_work, wp_closed, wd_closed, wp_closed_nocache_old`.
  `wp_closed`/`wd_closed` are the corrected formulas evaluated with **realized** `a_r`, `a_p`, `o_r`;
  `wp_closed_nocache_old` is the old no-cache form (to quantify the cross-term contribution).
  `a_p_realized` = the sum of new prefill tokens the request actually processed across its prefill
  steps (`Σ s_r`), which inherently excludes any cached prefix — so it comes directly from the
  accumulator, not from separate KV-cache-hit accounting; `cache_hit_frac = 1 − a_p_realized/a_r`.

## 4. Analysis — `campaigns/edpp-study/analyze/work_model_validation.py`

Reads the work-trace and reports, over completed requests, bucketed by prompt length, output length,
and `cache_hit_frac`:
- **Prefill:** relative error `realized_prefill_work` vs `wp_closed` — expected ≈ 0 for
  single-chunk prefill; for chunked prefill the residual equals the documented `C_attn·(a_p²−Σs_r²)/2`
  chunking term (reported, not an error).
- **Decode:** relative error `realized_decode_work` vs `wd_closed` — expected ≈ 0 (discrete sum used
  on both sides, so no `C1·o/2` gap).
- **Correction effect vs the old shipped form:** `wp_closed − wp_closed_nocache_old` decomposes into
  (i) the **basis change** — visible even on synth (no cache): old attention was `(C_attn/2)·a_r²`,
  corrected is `C_attn·a_r·(a_r + a_r/2) = 1.5·C_attn·a_r²`, i.e. ~3× the attention term, matching the
  active latency model; and (ii) the **cache effect** — additional on RAG (shared prefix, `a_p < a_r`).
  Report both so the routing-decision impact is explicit.
JSON report to stdout / `--out`; optional scatter PNG behind `--plots`.
**"Model exact" checkpoint:** max |relative error| of `realized_*_work` vs closed form is within float
tolerance (e.g. < 1e-6) for single-chunk requests on **both** synth (no-cache) and `rag_rate2.0`
(shared-prefix), and equals the documented chunking term otherwise. If it doesn't, the work model — or
the accumulator's mirroring of the active latency model — is wrong.

## 5. Testing & invariants

- **Unit (`sim/edpp_coeffs_test.go`):** `Wp(a_p, a_r)` equals `C_pf·a_p + C_attn·a_p·(a_r + a_p/2)` on
  hand cases; at no cache (`a_p = a_r`) equals `C_pf·a_r + 1.5·C_attn·a_r²` (the corrected basis, NOT
  the old `(C_attn/2)·a_r²`); cross/cache term grows as `a_p` shrinks below `a_r`; `Wd(a_r, o)` equals
  `Σ_{k=0}^{o−1}(C0+C1(a_r+k))` on hand cases incl. `o = 0` → 0, `o = 1` → `C0 + C1·a_r`.
- **Decision-change is expected (NOT a regression):** the prefill attention term legitimately changes
  (~3×), so synth decisions will shift vs pre-change. Do NOT assert byte-identical output. Instead,
  record the pre/post disagg-fraction and decision-trace delta on synth@2P2D as a documented,
  expected change in FINDINGS (the correction working). The real correctness gate is the per-request
  work validation below, not a golden diff.
- **Accumulator unit test (`sim/`):** a synthetic 2-request batch stepped N times sums per-request δ
  to the analytic `W_p`/`W_d`; disabled-gate produces no rows / no allocation.
- **INV-13 / INV-6:** run-vs-replay identical work-trace CSV; two seeded runs byte-identical.
- **Validation runs:** synth@2P2D rate 2.0 (single-chunk work exact; basis change vs old visible) AND
  `rag_rate2.0` (shared-prefix cache effect, work still exact vs realized). Both recorded in
  FINDINGS + a `repro_stage_b.sh`.

## 6. Deliverables

1. Corrected `Wp(a_p, a_r) = C_pf·a_p + C_attn·a_p·(a_r + a_p/2)` and `Wd(a_r, o)` (exact discrete sum)
   in `sim/edpp_coeffs.go`; `OnRoute`/`Decide` pass `a_r` (`sim/edpp.go`); `NomDecodeCtx` retained, no
   longer feeding `W_d`.
2. Per-request work accumulator in `InstanceSimulator` (`sim/`), gated; mirrors the active latency
   model's per-step charge; `--edpp-work-trace` flag + `RecordWorkTrace` config + CSV writer +
   run/replay wiring (mirrors Stage A).
3. `campaigns/edpp-study/analyze/work_model_validation.py` (JSON + optional `--plots`).
4. `campaigns/edpp-study/repro_stage_b.sh` (tracked) + FINDINGS "Stage B" section with the exactness
   checkpoint, the basis-change (~3× attention) and cache-effect results, and the expected synth
   decision shift.
5. Tests: coeff unit (corrected basis), accumulator unit, INV-13/INV-6. (No byte-identical regression
   — decision change is expected per §5.)
6. Documentation of the physics gap (§7).

## 7. Documented latency-model fidelity gap (deferred, not fixed here)

The default trained-physics latency model charges prefill attention on a **full-input-length** basis
(`ti·(a_r + ti/2)`), which over-counts physically-causal attention (`≈ a_r − a_p/2`, the triangular
`N²/2`) by up to ~3× at single-chunk. `C_attn` is calibrated to this basis, and Stage B's `W_p`
matches it so EDPP's congestion estimate is consistent with the simulator it runs in — the correct
choice for routing-decision fidelity. The roofline backend uses the physically-causal basis.

This is a **known, deliberate** limitation, NOT corrected in Stage B (fixing it means changing the
latency model + refitting `C_attn`, invalidating the frozen coeffs and all prior findings — a separate
scoped task). Actions:
- FINDINGS "Stage B" section states this explicitly, with the roofline-vs-trained-physics contrast.
- Add a one-line reconciliation note to `docs/design/2026-06-30-...md` §3.6 (on the design branch, in a
  later pass): the shipped `W_p` uses the `+ a_p/2` trained-physics basis and the discrete `W_d`;
  §3.6's causal `− a_p/2` and continuous `W_d` are the physics-pure forms (roofline-consistent).
- If a future study runs the roofline backend, `W_p` must switch to the causal basis; the accumulator
  already reads whichever model is active, so validation will flag the mismatch.
