EDPP coeff-wiring plan execution. Plan: docs/superpowers/plans/2026-06-24-edpp-rule-coeffs-wiring.md
Task 1: complete (commits b99e280..449b8c9, review clean)
  Minor (for final review): validate() branches AlphaP<=0, C1<0, CPf<=0 lack dedicated test cases (spec fixture under-covered).
  Minor: alpha-divergence denominator is AlphaD (spec unspecified; well-defined).
Task 2: complete (commits 449b8c9..feb3404, review clean)
  Minor (for final review): muPrefill clamp not independently exercised (symmetric with muDecode via clampMu).
Task 3: complete (commits feb3404..d29b93f, review clean — accessor reads RunningBatch.Requests, same as BatchSize; both refresh paths)
Task 4: complete (commits d29b93f..090e45c, review clean — 23/23 TestEDPP green)
  Minor (for final review): tauGuard test comment says "< AlphaD" but guard is "<=" (comment precision).
Task 5: complete (commits 090e45c..20722b7, review clean + 1 fix)
  Fix 20722b7: made TestEDPP_AlphaZero probe-free (forward-compat with Task 8 probe deletion).
  Note: probe-only tests TestEDPP_ExtractAlpha_CancelsDelta / TestEDPP_MarginalDelta_RecoversPerCopyWork still reference probe helpers — Task 8 deletes them.
Task 6: complete (commits 20722b7..717123c, incl 1 critical fix, re-reviewed clean)
  Critical FIXED (717123c): disaggregated-completion key mismatch leaked qp/qd/pending; now correlated to parent ID via edppConservationKey. Conservation test added.
  Important FIXED: Forget() cleanup wired to all terminal paths (timeout/drop/empty-signal).
  CARRY-FORWARD for Task 9 (finding D): implementer added DeploymentConfig.EDPPCoeffs (sim.EDPPCoeffs) field + cluster.go sets EDPPConfig.Coeffs=config.EDPPCoeffs. Task 9's --edpp-coeffs MUST load INTO this existing field (single mechanism), NOT add a competing EDPPCoeffsPath path-load that bypasses it. Reconcile.
  Note (for final review): EDPP does not naturally disaggregate in cluster test harness (decode snapshot Qd=0 at decision, ITL never breaches) — orthogonal follow-up; Task 7 swaps Decide to bookkept qpWork/qdWork which may affect this.
  Minor: edppConservationKey fallback parentRequests scan is O(n) (only on non-normal path).
Task 7: complete (commit 21e0e70, review clean — all predictor/test arithmetic reconciled)
  Minor: selectedDecodeITL now uncalled (Task 8 deletes); ITLP/ITLD trace field comments stale (always 0 now).
Task 8: complete (commits 21e0e70..3fcbb11, review clean + 1 coverage fix)
  Fix 3fcbb11: restored chunk-cap coverage (TestEDPP_ChunkCap, capped 2560 vs uncapped 10000).
  Extra deletion: TestEDPP_DeltaPfChunk (called deleted chunkInflation).
Task 9: complete (commits 3fcbb11..4a2be34, review clean — reconciled design: --edpp-coeffs loads into existing DeploymentConfig.EDPPCoeffs; both run+replay sites wired)
  Minor (FIX IN FINAL WAVE): TestResolveEDPPCoeffs_FrozenLlama70b tolerance is 1.0 — tighten to 1e-6 using exact 16613.539607002218 (cmd/root_test.go or wherever the test lives).
Task 10: complete (commits 4a2be34..2368a1c, review clean + 1 strengthening fix)
  Fix 2368a1c: dimensionless anchor now asserts LHS/RHS exact invariance across k + boundary-true case + TTFT term. All 3 §11 anchors pass.
Task 11: complete (commit 3296ea3, review clean — docs accurate, no deleted-symbol refs). NOTE: docs/superpowers/ is gitignored; design+plan artifacts remain on-disk only.
ALL 11 TASKS COMPLETE. Final whole-branch review next.

FINAL WHOLE-BRANCH REVIEW (opus): READY TO MERGE. No Critical/Important blockers.
  - Conservation sound across all terminal paths; INV-9/INV-6/R3/R4 intact; §11 anchors genuine.
  - Minors a,b,c,e fixed in 14d2cba; d (O(n) fallback) left as acceptable.
  IMPORTANT FOLLOW-UP (tracked, non-blocking): latent under-disaggregation bias — (1) deferred μ·Δt drain + P-work held to decode-completion overstates qpWork (damps further disagg); (2) decode-side TTFT measurement understates z_ttft for P-routed reqs. Recommend follow-up: implement μ·Δt drain or release P-work at prefill-complete; reconsider decode-side TTFT.
  Minor: OnRoute ap (raw len(InputTokens)) vs Decide ap (cache-adjusted) asymmetry — intentional/benign, undocumented at call site cluster.go:2030.

=== PLAN 2 (2026-06-25): waiting-only backlog + co-residency predictors ===
Base (PLAN 1 final) = 14d2cba. Plan: docs/superpowers/plans/2026-06-25-edpp-waiting-only-backlog.md
P2-Task 1: complete (commit 3acec64, review clean — TTFT_P uses T_pf(B−1); S_pf=0 tests unaffected)
P2-Task 2: complete (commits 3acec64..d395710, review clean + minors fix d395710 — OnAdmit kernel hook, tick+nil tests, preemption caveat doc)
P2-Task 3: complete (commits d395710..9eb5af7, opus review + 1 important fix)
  Waiting-only migration: OnAdmit drains at admission, OnComplete stops draining; cluster feedAdmission + 2-site wiring; correlation keys MATCH all 3 cases (D / prefill+"_prefill" / decode+"_decode").
  Important FIXED (9eb5af7): added TestEDPP_Cluster_ConservationViaAdmission_NormalCompletion — covers OnAdmit drain for normally-completing (OnComplete-path) requests; revert-verified (leaks qd=3458,pending=1 w/o wiring). Prior conservation tests only exercised Forget-path (forced-disagg artifact).
P2-Task 4: complete (commit 801766b, docs-only — backlog comment now waiting-only; no stale 'over-hold'/'removed at completion' refs)
P2-Task 4: complete (commit 801766b, docs-only).
P2 FINAL WHOLE-BRANCH REVIEW (opus): READY TO MERGE. Conservation sound end-to-end (3 cases); test gap closed (unit + revert-verified integration); predictors symmetric, no live-μ coupling; INV-9/INV-6/R4 intact.
  Minor FIXED (4b4504a): stale cluster.go:1220 OnComplete comment corrected.
PLAN 2 COMPLETE (14d2cba..4b4504a).

=== PLAN 3 (2026-06-25): EDPP empirical study ===
Base = 4b4504a. Plan: docs/superpowers/plans/2026-06-25-edpp-empirical-study.md
Task 1: complete (commit 4b4504a..5079659, review clean — Spec✅ quality Approved)
  Minor (final review): test asserts was_disaggregated key-presence but not prefill/decode_instance_id key-presence (field round-trip already covers them).
  ⚠️ resolved by controller: whole-tree build clean; only SimResult{} site is named-field literal at replay.go:874.
Task 2: complete (commit 5079659..826deec, review clean — Spec✅ quality Approved, no blockers)
  Adaptation: 8 existing test call sites updated to extractSimResults(m, nil).
  CARRY-FORWARD to Task 4: reviewer ⚠️ — verify cs.ParentRequests() is actually POPULATED during real replay (grep results JSON for was_disaggregated in Task 4 first run).
Task 3: complete (commit 826deec..4b602a5, review clean — Spec✅ quality Approved). Generated specs gitignored.

DESIGN DECISION (2026-06-25, user-confirmed) — topology + P/D-ratio axis:
- Code reality (verified): non-disagg/local path routes to DECODE pool; FormBatch prefills any ProgressIndex==0 req regardless of role, so a --decode-instances (decode-ONLY role) instance DOES run prefill locally for local requests. So "D = does both" maps to --decode-instances, NOT --prefill-decode-instances. Confirmed by TestDisaggregation_NonDisaggRoutedToDecodePoolOnly + newTestDisaggDeploymentConfig(4,2,2) (no shared pods).
- never/all-local baseline = --num-instances 4 homogeneous (each does both). NOT forced onto a split — equal-hardware comparison is INTENTIONAL, not a confound.
- Disaggregating arms (edpp/always/prefix-threshold) swept over P/D SPLITS at 4 total: 1P3D, 2P2D, 3P1D (the P/D-ratio study axis).
- Trace baked ONCE per (wl,rate) at num-instances 4 (topology-independent request stream), replayed across all decider×split combos.
- File naming: results_<wl>_rate<R>_<decider>_<tag>.json (tag = agg for never, else 1P3D/2P2D/3P1D); decisions_<wl>_rate<R>_edpp_<tag>.csv.
- Grid: 2 wl × 6 rates × {never@4 + (edpp/always/prefix)×3 splits} = 12 bakes + 120 replays.
- Task 4 (commit d61fd29) sweep.sh must be REWRITTEN for this; Task 6 glob/regex must parse decider+split.
Task 4: complete (REVISED for P/D-split design; new commit). Smoke synth/1.0: 10 results + 3 decisions, topology (1P3D/2P2D/3P1D) ACCEPTED (no validation error), disagg field populated end-to-end (always@2P2D=5000/5000; edpp=0 at low rate expected; never=0). RESOLVES Task2 carry-forward (ParentRequests populated in replay). num_requests=5000.
Task 5: complete (window.py — steady_window converges@204 on synth/1.0; offered_work rag=15255/500 synth=2500/4000 tok/s, correct P/D-boundedness).
Task 6: complete (report.py — tier1 summary, regret join 1141 misses joined to dominant term, 6 plots). Validated on smoke cell.
  Minor (final review): regret_join uses full (un-windowed) request set while summary uses steady-state slice — exploratory regret may include warmup. Acceptable; note in findings.
  Preview finding: synth/1.0 edpp(all-local) ttft_p99 125ms < always 142ms @1P3D — declines to disagg decode-bound load.
Task 7: complete (FINDINGS.md template).
FINAL WHOLE-BRANCH REVIEW (4b4504a..a1ebeda): READY TO MERGE, no Critical/Important. Go changes clean (nil-safe, disagg bool correct, R4/R8/omitempty OK, behavioral tests). 2 Minor Python items FIXED in follow-up commit: build_summary empty-glob guard + regret_join join-key contract comment. regret_join windowing caveat documented in FINDINGS.
ALL 7 TASKS COMPLETE. Full sweep running in background (rag is slow; ~12 bakes + 120 replays). Re-run report.py + fill FINDINGS once sweep done.

SLO REVISION (2026-06-25, user-driven): default τ_ttft=500ms was unmeetable for RAG 60k prefills (degenerate metric + EDPP z_ttft term). Fix:
- RAG clients split into slo_class standard (vector-qa) + batch (doc-read). NOTE: "interactive" is NOT a valid slo_class (valid: critical/standard/sheddable/batch/background) — used "standard".
- Per-workload per-class SLO/τ in sweep.sh: rag {standard: ttft500ms/itl150ms, batch: ttft5s/itl200ms}; synth {batch: ttft2s/itl150ms}. Passed via --slo-ttft/--slo-itl + --edpp-tau-ttft-classes/--edpp-tau-itl-classes.
- report.py: per-(workload,class) SLO_TTFT_US/SLO_ITL_US; added slo_itl_attain column (decode-bound synth → ITL is the real SLO).
- slo_class change requires re-bake; old sweep killed, full sweep relaunched (persistent monitor b4vrnr36z).
PRELIM (old degenerate SLO, RAG only): never@4 best on TTFT p99; edpp best of disagg arms at 1P3D/2P2D (stays local); edpp pathological at 3P1D (piles local load on lone D). disagg_frac rises w/ load & P-count. Reason never wins: equal-hardware split strands decode-only nodes (tiny RAG decode) + halves prefill servers. To be re-evaluated under realistic SLOs + with synth.

=== PLAN 4 (2026-07-01): EDPP estimator-validation harness (Stage A) ===
Branch feat/edpp-estimator-validation, base a8c20ad. Plan: docs/superpowers/plans/2026-07-01-edpp-estimator-validation.md
NOTE: golangci-lint MISSING on this machine — using go vet + gofmt per task; run golangci-lint at CI/final gate.
Task 1: complete (commit 109537e, review clean — SPEC ✅ / QUALITY Approved, no Critical/Important)
  CARRY-FORWARD to Task 4 (Minors): (1) recordAdmissionTime only wired inside `if cs.sloFeedback != nil` OnAdmit closures — fine for Stage A (--pd-decider edpp sets sloFeedback) but if --pd-outcome-trace must work without EDPP, revisit. (2) localAdmitTimes init only inside PD-enabled block; SetRecordPDOutcomes(true) must not be enabled outside a PD deployment (nil-map risk).
Task 2: complete (commit 5189e64, review clean — SPEC ✅ / QUALITY Approved)
  ⚠️ Minor (non-blocking): writer test is single happy-path; multi-record caller-order ("no sort") contract is covered downstream by Task 4's determinism test. record.go/pd_outcome_csv.go doc comments forward-reference Task 3/4 artifacts (intentional).
Task 3: complete (commit c26dd11, review clean — SPEC ✅ / QUALITY Approved). Disaggregated derived in builder (no ParentRequest field); tests assert sorted order, computed disagg, t_adm values + causality law.
  RESOLVED by controller (reviewer ⚠️ on metric keying): projectPDMetrics() (cluster.go:882, run-end) RE-KEYS per-request E2E/TTFT/ITL from sub-req ids to PARENT id (m.Request*[pid]). So realized metrics ARE keyed by the parentRequests map key. No gap.
  CARRY-FORWARD to Task 4 (ordering): BuildPDOutcomeRecords MUST be called AFTER projectPDMetrics has finalized metrics (i.e., at run end on final metrics) — else disagg records get zero metrics.
  Known limitation (documented): LocalTAdm always 0 (local absolute enqueue instant not tracked without flow control); downstream analysis must not treat it as meaningful.
Task 4: complete (commit 221d895, review clean — SPEC ✅ / QUALITY Approved, no findings)
  Shared helper writePDOutcomeTrace (root.go:1082) called from run (root.go:2170) + replay (replay.go:730); both pass cs.AggregatedMetrics() (finalized, projectPDMetrics-mutated in place at cluster.go:882 — ordering constraint satisfied). Flag shared via registerSimConfigFlags (no double-register). Zero-cost when unset (verified). Parity test builder-level only.
  ⚠️ Minor (final review): no automated CLI-level run==replay CSV parity assertion (builder-level determinism test only).
  *** SIGNIFICANT PRE-EXISTING FINDING (NOT introduced here; reproduces WITHOUT the flag; OUT OF SCOPE for this diff — confirmed by reviewer): PD+EDPP REPLAY completes only ~23/50 requests vs run 50/50. So replay-side outcome CSV realized/completed columns are NOT trustworthy for PD+EDPP until fixed. Admission-time columns match run byte-for-byte. AFFECTS bake-and-replay study methodology (edpp-study bakes once + replays across cells). Stage A anchor (Task 6) uses `blis run` directly, so anchor is unaffected. RAISE TO USER + separate investigation.
Task 5: complete (commit 0385132, review clean — SPEC ✅ / QUALITY Approved). Go-bool parse fix (parse_go_bool) applied to completed+disaggregated (draft's `== True` matched zero rows); div-by-zero/inf/empty guards present; e2e-tested on synthetic CSVs (total5/completed4/truncated1, disagg ratio 2.0 via ttft_p, local 1.279 via ttft_d).
  ⚠️ Minor (final review): truncated_or_dropped denominator assumes every outcome row has a decision match; add a warning if len(df)!=len(out) (join-loss unaccounted). Non-blocking; anchor run has total join match.
Task 5 FIX: empty-class grouping (commit 235c9a5) — local rows (empty slo_class→NaN) were silently dropped by groupby; now bucketed class=unknown + join-loss warning. Recovers ttft_d (local-path) validation.
Task 6: complete (commit 2634c98, FINDINGS.md). ANCHOR RAN SUCCESSFULLY (blis run bake + replay 2P2D edpp, synth2.0, 5000 reqs, disagg 4545/local 455). Harness works end-to-end.
  RESULT: ttft_p disagg median 2.08×; ttft_d local median 2.74× but tail p90 324s vs 0.55s (~590×) = archived HOL-blind reproduced; decode-admission (waiting-only) median ~905× under-pred (0.31s vs 285s) = Stage C case. prefill-adm 1.33×.
  DOWNGRADED the Task-4 "replay 23/50" finding: on full 5000-req config replay completes 4545/5000 (2-decode saturation, expected) — NOT a parity/engine bug; 23/50 was small-config horizon artifact. Admission columns match run byte-for-byte.
  FOLLOW-UPS (in FINDINGS): (1) local rows lack slo_class/input_tokens (no ParentRequest) — analysis buckets class=unknown; proper fix populates from original request stream. (2) `completed` proxy (E2E>0) = 5000 vs sim 4545 — looser than sim completion. (3) decode-side TTFT understated (realized decode T_adm 285s vs TTFT 120ms) — orthogonal known issue.
ALL 6 TASKS COMPLETE. Final whole-branch review next.
FINAL WHOLE-BRANCH REVIEW (opus, a8c20ad..2634c98): READY TO MERGE. No Critical/Important. Verified: no sim behavior change (edpp.go/edpp_coeffs.go untouched); zero-cost-when-unset; data-path keying (AggregatedMetrics post-projectPDMetrics, parent-keyed); INV-5/6/13 hold. Graceful degradation for non-EDPP PD deciders (zero columns, not error) — acceptable.
  Acceptable follow-ups (post-merge, non-blocking): (1) end-to-end CLI run==replay CSV-diff test; (2) populate slo_class/input_tokens for local rows from original request stream; (3) reconcile `completed` proxy (E2E>0) with sim completed_requests; (4) CI MUST run golangci-lint (missing locally).
PLAN 4 COMPLETE (a8c20ad..2634c98, branch feat/edpp-estimator-validation, 7 commits). NOT pushed/PR'd.
REPRO RECORDED (commit c3db115): campaigns/edpp-study/repro_stage_a.sh (tracked, runnable) bakes+replays+analyzes → out/stage_a/bias.json. Verified byte-identical to manual anchor (~110s). FINDINGS "Reproduction" block has checkpoint numbers (2.085 / 904.8 / 1.326 / 2.740). Findings + repro fully recorded in tracked files (FINDINGS.md + repro_stage_a.sh); code branch feat/edpp-estimator-validation NOT pushed/PR'd (WIP, no destination yet).

=== PLAN 5 (2026-07-02): EDPP work model correction + per-request validation (Stage B) ===
Branch feat/edpp-estimator-validation, base <see below>. Plan: docs/superpowers/plans/2026-07-02-edpp-work-model.md
DECISION (user-approved): W_p matches ACTIVE latency model (trained-physics basis, +a_p/2), NOT §3.6's causal −a_p/2. Trained-physics over-counts causal attention ~3× (roofline is causal); documented as deferred latency-model fidelity gap (spec §7). NO byte-identical synth regression — decision shift expected. golangci-lint still MISSING locally (go vet+gofmt).
Task 1: complete (commit 3c28373, review clean — SPEC ✅ / QUALITY Approved). Wp=C_pf·a_p+C_attn·a_p·(a_r+a_p/2); Wd discrete sum; both call sites pass a_r; Wp(100,100) test 3100→8100. No refit.
  Minor (final review): (1) stale arithmetic comments in sim/edpp_test.go:888,915 (wd=2148/qd=4148 from old deltaBarDecode path; tests still pass via loose assertions) — update comments. (2) deltaBarDecode now has no production caller (intended; still unit-tested).
Task 2: complete (commit 453f56e, review clean — SPEC ✅ / QUALITY Approved). WorkTraceRecord (12 fields) + WriteWorkTraceCSV (no sort), mirrors pd_outcome_csv.go.
  Minor (final review): writer test only pins first 7 columns; trailing 5 float columns' order unverified by unit test (covered end-to-end by Task 5/6). Pre-existing gofmt flag on untouched sim/trace/trace.go (not ours).
Task 3: complete (commit 159fd86, review clean — SPEC ✅ / QUALITY Approved). Per-request work accumulator in Simulator; prefill uses FULL si (float s/2.0, matching trained-physics MODEL not the recorder's int nt/2 — correct for closed-form validation); decode C0+C1·ProgressIndex; zero-cost when disabled; classification matches recorder. SetWorkTrace/WorkAccumulators wired in Task 4.
Task 4: complete (commit cfa6523, review clean — SPEC ✅ / QUALITY Approved, no Critical/Important). --edpp-work-trace flag on run+replay via shared writeWorkTrace helper (INV-13); BuildWorkTraceRecords correlates prefill/decode sub-reqs to parent via parentRequests + claimed map; closed forms with realized ap/ar/o; sorted by request_id. coeffs=config.EDPPCoeffs; instances via cs.instances/inst.sim/inst.id.
  Run/replay decode divergence (o_r/realized_decode) = pre-existing DES issue (hits pd-outcome identically), OUT OF SCOPE — does NOT break Stage B exactness (each execution internally consistent: realized work + closed form use same realized o_r).
  Minor (final review): (1) local-fallthrough relies on claimed map, not suffix exclusion — correct under "parent registered before sub-req runs" invariant; add a one-line comment. (2) vestigial workByInstance field (test-only holder).
Task 5: complete (commit 2f1051f, review clean — SPEC ✅ / QUALITY Approved). work_model_validation.py: single-chunk/chunked prefill split, all-rows decode, exact-0 gates, div0/inf/empty guards, lazy matplotlib. NOTE: brief's checkpoint "median_wp_over_old ≈1.53" was MY arithmetic error; correct value = 2.022 (median of 2.976, 1.068). Script correct; reviewer independently confirmed. Fix Task 6 checkpoint expectation to 2.022.
  Minor: prefill_chunks==0 falls in neither group (unreachable); lambda-assign nit.
Task 6: complete (commit ec1871b). repro_stage_b.sh + FINDINGS "Stage B". VALIDATION RAN (synth+rag @2P2D rate2.0, 5000 each):
  MODEL EXACT — single-chunk prefill max_abs_rel_err 5.6e-16 (synth) / 5.8e-16 (rag); decode 1.2e-15 (rag, all 5000) / synth median0 p90 2.5e-4 (small preemption tail). Chunked prefill residual = documented Σs² term (synth n=299 max0.28; rag n=3582 max0.22).
  Surprise: synth is NOT no-cache (median cache_hit_frac 0.91); rag 0.18. Doesn't affect exactness (accumulator+closed form both use realized a_p). correction median wp/old ~1.04 (linear term dominates; 3× basis change only at a_p≈a_r).
  §7 fidelity gap documented in FINDINGS. Decision-shift quantification deferred (needs pre-Task-1 rebuild; not the correctness gate).
ALL 6 STAGE B TASKS COMPLETE (c3db115..ec1871b). Final whole-branch review (Stage B delta) next.
STAGE B FINAL REVIEW (opus, c3db115..ec1871b): READY TO MERGE. No Critical/Important. Verified: accumulator == model StepTime classification (float-exact); INV-9 (decider input-side only); correlation total/no-leak; INV-6 sort; INV-13 shared write path; zero-cost off. Minors all acceptable follow-ups: stale edpp_test.go comments; dead deltaBarDecode; writer test pins 7 cols; vestigial workByInstance; py nits. CI must run golangci-lint.
PLAN 5 (Stage B) COMPLETE (c3db115..ec1871b, 6 commits). Branch feat/edpp-estimator-validation now = Stage A (7) + Stage B (6). NOT pushed/PR'd (WIP).

=== PLAN 6 (2026-07-02): EDPP occupancy-aware admission estimator + fidelity ablation (Stage C) ===
Branch feat/edpp-estimator-validation, base ad3acc6. Plan: docs/superpowers/plans/2026-07-02-edpp-admission-estimator.md
Spec: docs/superpowers/specs/2026-07-02-edpp-admission-estimator-design.md. 8 tasks.
Key: pluggable estimator (waiting/little/fluid/rollforward + oracle logging-only variants), replaces qD/muDec & qP/muPf admission terms; default waiting=byte-identical; INV-9 guard rejects oracle as routing driver; T1(single decode,never)/T2(1P1D,always) forced-routing microbenchmarks log all 6 variants; error decomposition realized−oracle(form) vs oracle−N̂(prediction). Layer-2 closed-form = SEPARATE next track.
NOTE: specs/plans/ledger now TRACKED on branch (force-added). golangci-lint still missing locally.
Task 1: complete (commit 1d0b7f8, review clean — SPEC ✅ / QUALITY Approved). AdmissionContext+RunningReqState+AdmissionDelayEstimator iface+waiting impl; seam swaps qD/muDec & qP/muPf only; default byte-identical (clampMu≥1e-3 so guard never fires); factory errors on unknown; constructor panics per R3.
  Minor: Name() method added beyond single-method iface (useful, harmless); waiting Mu<=0 guard unreachable on current clamped path (intentional).
Task 2: complete (commit 0dc7ada, review clean — SPEC ✅ / QUALITY Approved). little estimator = QueueDepth/AdmissionRate, guard rate<=0→0; case "little" in factory. Mirrors waiting pattern.
Task 3: complete (commits b769254..44065d0, review clean — SPEC ✅ / QUALITY Approved). RoutingSnapshot enriched (RunningDecode/RemainingDecodeWork/AdmissionRate); Decide fills both context literals; oracle-gated TrueRemaining (INV-9); zero-cost when detail off. FIX 44065d0: DispatchRate was structurally 0 in routing path (#1382) → re-wired snap.DispatchRate=inst.LatencyStats().DispatchRate into buildRouterState GATED on admission-detail flag; AdmissionRate = explicit-field > DispatchRate/1e6 > 0. little now functional.
  Minor (final review / downstream): (1) KVBlocks=⌈ProgressIndex/blockSize⌉ = total-seq blocks not decode-only (fine for rollforward, name may mislead). (2) DispatchRate 0 until first completion → little inactive during warmup. (3) ReqKVNeed uses len(InputTokens) (prefill need, applied to decode ctx too). (4) no PD integration run yet — Task 8 microbenchmark is the empirical check.
Task 4: complete (commit 9be5533, review clean — SPEC ✅ / QUALITY Approved). fluid estimator: free-slot→0; else N_ahead(=1)/X̂_dep, X̂_dep=BatchSize/(RemainingStepsEst·TIter). Test incl. full-batch/zero-waiting→~5000µs where waiting=0 (the bug). Minor: fluid models only slot-freeing (nAhead=1 hardcoded), not KV-driven admission — comment overpromises; documented simplification.
Task 5: complete (commit ac54d6e, review clean — SPEC ✅ / QUALITY Approved). rollforward: per-req departStep=TrueRemaining≥0 else max(RemainingStepsEst,1); sort.SliceStable asc; walk freed slots+KV; return firstDepart·TIter, cap at last; tests 3000µs (oracle) + 4000µs (est). Minor: departStep·TIter step-vs-elapsed (per brief); int64() floors est; cap/multi-departure branches untested (Task 8 exercises).
Task 6: complete (commit 7262fae, review clean — SPEC ✅ / QUALITY Approved). fluid_oracle/rollforward_oracle registered (reuse impls, prefer TrueRemaining); IsDeployableEstimator truth table; INV-9 guard in NewEDPPDecider panics on non-deployable routing driver (oracle still constructible directly for logging).
  Minor (final review): (1) guard panic path in NewEDPPDecider not exercised by a recover-test (INV-9 load-bearing — consider adding). (2) fluid_oracle untested e2e. (3) panic-vs-logrus.Fatalf: correct for library code.
Task 7: complete (commits 45a8ff3 + fix 0fe035f, review SPEC ✅ / QUALITY approved-after-gofmt-fix). --edpp-admission-trace (AdmissionRecord + WriteAdmissionCSV, 9 cols, no sort); local_t_adm now real (local enqueue captured at InjectRequestOnline); Decide attaches AdmissionCtxDecode/Prefill to decision (gated SetCaptureAdmissionContext, decision unaffected); BuildAdmissionRecords runs all 6 estimators per request, sorted; run+replay shared writeAdmissionTrace (INV-13). Renamed legacy AdmissionRecord→AdmissionDecisionRecord (complete). FIX 0fe035f: gofmt cmd/root.go + sim/trace/trace.go (from the rename).
  Minor (final review): (1) disagg row path unit-tested only (all-local smoke) — Task 8 T2 exercises live. (2) BuildAdmissionRecords p() maps estimator-construction error→0 (hardcoded-valid names, won't fire). (3) guard panic path still lacks recover-test (Task 6 minor).
Task 8: complete (commit 7725726). repro_stage_c.sh (T1 force-local via c-xfer 100s / T2 force-disagg c-xfer 0s, edpp 1P1D) + admission_ablation.py + honest FINDINGS "Stage C".
  RESULT: HEADLINE VALIDATED — T1 local (ttft_d, saturated p50 560s): waiting 57× under-pred → rollforward 1.29× near-exact (N̂ error ~0). The 905×-analog fixed by rollforward on the local pool.
  OPEN ISSUES (documented, NOT clean monotonic ablation): (1) fluid anomalous (~1e6× under-pred — RemainingStepsEst/free-slot collapse, needs debug). (2) little inert (predicts 0 — AdmissionRate unavailable at decision despite Task-3 wiring; DispatchRate 0 until first completion). (3) prefill-pool estimators broken (only decode RunningDecode enriched; prefill context has no occupancy → ttft_p unvalidated). (4) rollforward OVER-predicts on disagg decode path (T2 0.41×) — 905×-analog lives on local path (T1), which it fixes.
ALL 8 STAGE C TASKS COMPLETE (code 1d0b7f8..7725726). Infra merge-ready; fluid/little/prefill = follow-ups before paper ablation figure. Final whole-branch review next.
STAGE C FINAL REVIEW (opus, ad3acc6..7725726): READY TO MERGE (as infrastructure). No Critical/Important, no blockers. Verified: default byte-identical (TAdmEstimator unset→waiting; clampMu makes guard dead-defensive), estimator replaces only 2 admission terms, INV-9 oracle guard, INV-6 sort, INV-13 shared write path, INV-7 Periodic tier, zero-cost off, AdmissionRecord rename consistent, local_t_adm INV-5-guarded.
  Deferred follow-ups (land 1-3 before paper fidelity figure): (1) fluid ~1e6× under-pred bug; (2) little predicts 0 (needs admission-rate signal); (3) prefill-pool enrichment (only decode enriched → ttft_p inert); (4) rollforward over-predicts disagg decode; (5) NO CLI flag to select TAdmEstimator (programmatic-only — add before Stage D deploys rollforward as driver, also exercises guard runtime path); (6) minors: guard recover-test, rollforward cap/multi-departure tests, KVBlocks approx.
PLAN 6 (Stage C) COMPLETE (infra). Branch feat/edpp-estimator-validation = Stage A(7)+B(6)+C(spec/plan+8 tasks+fixes). Headline: rollforward fixes 57×→1.3× admission under-pred on saturated local pool. NOT pushed (user: push after Stage C).
