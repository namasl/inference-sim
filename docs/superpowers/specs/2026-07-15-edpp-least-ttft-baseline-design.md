# `least-ttft` disaggregation baseline (Design)

Date: 2026-07-15
Branch: `feat/edpp-estimator-validation`
Related:
- `campaigns/edpp-study/FINDINGS.md` (the workload-spectrum sweep — to be recorded) and this session's diagnosis:
  on a fixed 1P2D topology, the optimal disaggregation policy is a *global corner* for decode-bound
  (`always`) and prefill-bound (`never`) workloads, EXCEPT that on prefill-bound overload an *interior*
  dynamic split beats both corners and reduced-EDPP achieves it (goodput ~0.92 vs best static fraction ~0.60,
  vs corners 0.03–0.11, robust across seeds).
- `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (EDPP's drift-plus-penalty rule).

## 1. Purpose

Determine whether EDPP's prefill-bound win is **specific to the drift-plus-penalty machinery** (backlog drift
`Q_i`, SLO virtual queues `z_ttft`/`z_itl`, penalty weight `V`, transfer penalty) or is **generic dynamic
least-TTFT routing** that any load-aware rule would capture. This is the decisive experiment that frames the
paper: if a minimal least-TTFT rule matches EDPP, the machinery is unnecessary (simplification/characterization
result); if EDPP beats it, the machinery earns its keep (method result).

To answer it cleanly we need a baseline that differs from reduced-EDPP in **exactly one thing**: the decision
rule. Everything else — the admission-delay estimator, the work model, the predicted TTFTs, the
scorer-selected decode instance — must be identical, or the comparison is confounded.

## 2. The rule

For the scorer-selected decode instance `d`, EDPP's reduced path already computes two predicted
time-to-first-token values (`sim/edpp.go:602-603`):
- `ttftD` = local placement: admission delay on `d` + local prefill (`tAdmD + nChunks·(tBminus1 + deltaPfChunkLocal)`).
- `ttftP` = disaggregated placement: admission delay on the prefill pool + prefill there + KV transfer
  (`tAdmP + nChunks·(tIterPrefill + deltaPfChunk) + c_xfer`).

The `least-ttft` rule is simply:
```
Disaggregate  ⟺  ttftP < ttftD
```
It **reuses `ttftP`/`ttftD` verbatim** and ignores everything else EDPP computes: the backlog/balance drift
terms (`balanceTermD/P`, i.e. `Q_i`), the SLO virtual queues (`z_ttft`, `z_itl`), the standalone transfer
penalty term, and `V`. Note `ttftP` **already includes** the transfer *latency* (`c_xfer`), so transfer is
still correctly priced as latency — it is simply not double-counted as a separate penalty.

Ties (`ttftP == ttftD`) → local, mirroring reduced-EDPP's strict `Disaggregate = lhs > rhs` tie handling
(ties → local).

## 3. Architecture — a decision-rule MODE on the EDPP decider (not a standalone decider)

- Add `EDPPConfig.Rule string` (default `"dpp"`; new value `"least-ttft"`). Validate to those two values in
  `NewEDPPDecider`/`EDPPConfig.validate` (unknown → panic, R3 style).
- Store `rule` on `EDPPDecider`.
- At the reduced decision site (`sim/edpp.go:623`, `dec := DisaggregationDecision{Disaggregate: lhs > rhs}`),
  branch on `d.rule`:
  - `"least-ttft"` → `Disaggregate: ttftP < ttftD`
  - `"dpp"` (default) → the existing `lhs > rhs`
  Everything above line 623 (all estimation) is unchanged and shared.
- CLI: `--edpp-rule` (string, default `"dpp"`), wired into `EDPPConfig.Rule`; used with `--pd-decider edpp`.

**Rationale for the mode vs a standalone decider.** The baseline's scientific value is being *EDPP minus the
machinery*. Implementing it as a mode guarantees byte-identical `ttftP`/`ttftD` (same rollforward estimator,
same work model, same chunk terms, same scorer-selected `d`), so the sole difference from reduced-EDPP is the
final comparison. A standalone decider would recompute the predicted TTFTs and risk subtle estimation drift
that would confound the very comparison the baseline exists to make.

## 4. Scope & non-goals

- **Reduced path only.** Mirrors how every comparison in this study was run (scorer picks the decode instance;
  the decider picks local-vs-disaggregate). The joint (`--edpp-joint`) path is untouched; `least-ttft` +
  `--edpp-joint` is rejected with a clear fatal at the CLI boundary (unsupported combination) — out of scope.
- **No new estimation.** `least-ttft` adds no new latency math; it only selects a different comparison over
  values EDPP already produces.
- **Config-only + one branch.** No change to the estimator, work model, scorer, or joint path.
- Trace: the existing `--edpp-decision-trace` already logs `ttft_d`/`ttft_p`; no new trace needed. (The trace's
  `lhs/rhs/disaggregate` columns reflect the `dpp` rule; under `least-ttft` the `disaggregate` column must
  reflect the rule actually used — set it from the same branch.)

## 5. Testing

- **Unit — rule selects on predicted TTFT** (`sim/edpp_test.go`): a decider with `Rule="least-ttft"` returns
  `Disaggregate=true` on a state where the local decode instance is congested (drives `ttftD` high) and
  `false` where the prefill pool is congested (drives `ttftP` high). Reuse the existing reduced-path test
  helpers (`defaultTestEDPPConfig`, `decodeState`, prefill-snapshot closure).
- **Unit — machinery is bypassed (the key guard):** with `Rule="least-ttft"`, set the `z_ttft`/`z_itl` virtual
  queues large (via the SLO-feedback path or direct state) and confirm the decision is **unchanged** — proving
  the drift/z terms do not enter. (Under `dpp`, the same large z would change the decision; assert that
  contrast so the test would fail if the branch leaked into the machinery.)
- **Regression — default unchanged:** `Rule="dpp"` (default) is byte-identical to today; full `go test
  ./sim/... ./cmd/...` green; a golden reduced-decision test unchanged.
- **Config validation:** unknown `--edpp-rule` value → clear fatal at the CLI boundary; `least-ttft` +
  `--edpp-joint` → rejected with an explanatory error.

## 6. The experiment this unlocks (recorded separately, not part of the build)

Extend the workload-spectrum harness to add a `least-ttft` arm (`--pd-decider edpp --edpp-rule least-ttft`,
same decode-balancing scorer, same per-archetype SLO) alongside `never`/`always`/`edpp(dpp)`. Decision:
- Prefill-bound (where `edpp` got ~0.92): `least-ttft ≈ edpp` ⟹ the win is generic dynamic least-TTFT routing;
  the drift-plus-penalty apparatus is unnecessary (**simplification/characterization paper**).
- `edpp ≫ least-ttft` ⟹ the machinery earns its keep (**method paper**).
Record the spectrum table (never/always/least-ttft/edpp × archetype × load) in FINDINGS.

## 7. Deliverables

1. `EDPPConfig.Rule` + `EDPPDecider.rule` + validation (`sim/edpp.go`).
2. The one-line decision branch at the reduced decision site (`sim/edpp.go:623`).
3. `--edpp-rule` CLI flag wired to `EDPPConfig.Rule` (`cmd/root.go`).
4. Unit tests (rule-selects-on-TTFT; machinery-bypassed; config validation) + regression.
5. (Follow-on, separate) spectrum experiment + FINDINGS entry.

## 8. Risks

- **Trace `disaggregate` column consistency** — must reflect the active rule, not always `lhs > rhs`. Covered
  in §4.
- **Interpretation risk (not a code risk):** if `least-ttft` ties EDPP on prefill-bound but EDPP wins
  elsewhere (e.g. under tighter SLOs where the z terms matter), the framing is nuanced, not binary — record
  the full spectrum, not a single operating point.
