# Per-instance θ_i in the EDPP joint decider (Design)

Date: 2026-07-14
Branch: `feat/edpp-estimator-validation`
Related:
- `campaigns/edpp-study/FINDINGS.md` "Hardware-θ opportunity test" + "Saturating-regime follow-up"
  (this is T-B in `campaigns/edpp-study/TODO.md` "ROADMAP" — the change those results motivate).
- `docs/design/2026-06-30-pd-joint-routing-problem-formulation.md` (§5.3 objective J; heterogeneous
  per-instance θ_i is part of the formulated model — this realizes it in the decider).
- The `hw_config_by_gpu` bundle wiring (commits `60dcdaa..b07a79f`) — the prerequisite that lets
  node-pool-placed instances run on different `HWConfig`. This design is its decider-side counterpart.
- `scripts/calibration/README.md` + `fit_coeffs.py` + `repro_llama70b.sh` (the calibration pipeline reused
  for offline θ_i extraction).

## 1. Motivation & goal

The EDPP joint decider scores every candidate (decode d, prefill p) with **one global** work-model
coefficient set `d.coeffs EDPPCoeffs` (`sim/edpp.go:293`; α_d, α_p, C0, C1, C_pf, C_attn). It has no
per-instance physics. The simulator, however, now executes node-pool-placed instances on **per-instance
`HWConfig`** (fast H100 vs slow A100 via `hw_config_by_gpu`), so two same-role decode instances genuinely
run at different speeds.

The T-A opportunity test (FINDINGS) showed the consequence in two regimes:
- **Under-capacity** (fast node has spare room): joint already wins reactively (goodput ~0.97 vs blind
  load-balance 0.77) — its per-instance congestion terms (`Q_i`, occupancy-aware `T̂`) suffice to *avoid*
  the slow node. θ_i not required here.
- **Saturating** (fast node saturated; the goodput-optimal split is an **interior ~86%-fast** allocation,
  0.96): reactive joint == blind load-balance == reduced (all ~0.82, ~77% fast) — they **equalize queues**,
  but the optimum **over-weights** the faster node. Reactive signals provably cannot supply that; only
  proactive per-instance speed knowledge can.

**Goal.** Give the joint decider a **per-instance `EDPPCoeffs` (θ_i)**, keyed by the candidate's GPU type,
so `jointCandidateCost` evaluates each candidate's work/rate terms with its own hardware physics. The
argmin then proactively weights the fast node toward the optimal split. Success is quantified against the
T-A saturating result (§7): θ_i-joint should move the realized fast-share from ~77% toward ~86% and goodput
from ~0.82 toward ~0.96, beating reduced-EDPP and blind load-balance, while leaving the under-capacity
regime and all homogeneous behavior unchanged.

## 2. Scope

**In scope**
1. A per-GPU θ store on the decider (`coeffsByGPU map[string]EDPPCoeffs`) + a `coeffsFor(gpuType)` selector
   that falls back to the single global `d.coeffs`.
2. θ selection in `jointCandidateCost` (decode-side terms keyed to the decode candidate's GPU; prefill-side
   to the prefill candidate's GPU), and in the reduced path for its single fixed d.
3. Offline extraction that produces one coeffs file per GPU type, **consistent with the simulator's own
   trained-physics execution** at that `HWConfig` (reusing the existing calibration pipeline).
4. Config plumbing: a `coeffs_by_gpu` bundle field mirroring `hw_config_by_gpu`.
5. The snapshot prerequisite: populate `RoutingSnapshot.GPUType` on the EDPP-facing snapshots.
6. Determinism preservation when the hoisted candidate-invariant precomputes move into the candidate loop.

**Non-goals (explicitly deferred)**
- **Per-instance `Z^I_i`** (ITL-deficit queue). The ITL term stays per-class `z_itl`
  (`sim/edpp.go:731-735`, `752-754`, `853-854`). The saturating-split gap is a decode work-rate / µ problem
  the θ_i work model addresses; `Z^I_i` is a separate ITL-fairness axis and is not needed to close the
  split. Deferred to a later sub-project.
- No change to the reduced *rule structure* or the formulation. θ_i only swaps which coefficient set feeds
  existing terms.
- No autoscaler/replay support beyond what node-pools already allow (node-pools are replay-fatal per
  INV-13 — consistent with T-A; heterogeneous θ runs use `blis run`).
- No roofline-derived or analytically-scaled θ (an alternative that was considered and rejected in favor of
  extraction-from-sim-physics for consistency).

**Hard backward-compatibility invariant.** When `coeffs_by_gpu` is absent (or all instances share one GPU
type), the run is **byte-identical** to today (INV-6). `coeffsFor` returns `d.coeffs` for every candidate,
and the moved precomputes recompute the same values.

## 3. Offline θ_i extraction (produce the coeffs files)

Each non-H100 coeffs file is produced by re-running the **existing** calibration procedure with the
instance served on the target `HWConfig`, so the fitted θ_i is exactly the trained-physics work model the
simulator uses to execute that hardware (the no-confound property).

Pipeline (`scripts/calibration/`):
- `BLIS_STEP_CSV=<f>` taps one CSV row per executed engine step (`t_iter_us, b_dec, kv, s_pf, pf_ctx,
  batch_size`) — off by default, no effect on determinism.
- `fit_coeffs.py D*.csv P*.csv -o coeffs.json` fits the six constants with R²/collinearity diagnostics.
- `repro_llama70b.sh` is the H100 instantiation and regenerates the committed
  `coeffs-llama70b-h100-tp4.json` bit-exactly (the trust check).

New: `scripts/calibration/repro_theta_by_gpu.sh` generalizes `repro_llama70b.sh` to run the same decode
(D1–D4) and prefill (P1–P3) calibration `blis run`s with the instance pinned to a target `HWConfig`, via a
**single-pool `hw_config_by_gpu` bundle** (`--num-instances 1 --policy-config <onepool-bundle>`,
`gpu_type: <GPU>`, `hw_config_by_gpu.<GPU>: {tflops_peak, bw_peak_tbs, ...}`), then fits with
`fit_coeffs.py`. Output: `coeffs-llama70b-<gpu>-tp4.json`. For the T-A synthetic slow device this yields
`coeffs-llama70b-a100crippled-tp4.json` (HWConfig 400 TFLOPS / 0.7 TB/s). H100 stays the committed file.

**Feasibility to resolve in the plan:** confirm `BLIS_STEP_CSV` taps steps under a single-instance
node-pool bundle (the README's calibration uses plain single-instance `blis run` with `--hardware`). If it
does not, the fallback is a tiny harness that samples `latency.NewLatencyModel(LatencyCoeffs, HWConfig)`
directly over the same (b_dec, kv, s_pf, pf_ctx) calibration grid and writes the identical CSV shape — same
fit, no full simulation. Either route produces a file consistent with the sim's execution physics.

The fitted files are committed under `scripts/calibration/` and their R²/`cond_*` diagnostics recorded (the
fit must be well-conditioned, matching the H100 file's quality bar).

## 4. Config surface & keying

**Keying by `gpu_type`** — natural (few hardware types), mirrors `HWConfigByGPU`, and a bundle already
declares `gpu_type` per pool. (InstanceID keying was considered and rejected: it needs a construction-time
ID→θ map and buys nothing for a study where hardware is homogeneous within a GPU type.)

**Bundle field.** Add to `PolicyBundle` (`sim/bundle.go`), mirroring the `hw_config_by_gpu` pattern:
```yaml
coeffs_by_gpu:
  H100: scripts/calibration/coeffs-llama70b-h100-tp4.json
  A100: scripts/calibration/coeffs-llama70b-a100crippled-tp4.json
```
- yaml-tagged field `CoeffsByGPU map[string]string \`yaml:"coeffs_by_gpu"\`` (GPU type → coeffs file path).
  nil map ⇒ no override.
- **Validated at load (fail-fast).** The paths are resolved and each file is loaded via the existing
  `sim.LoadEDPPCoeffs` (which already runs `EDPPCoeffs.validate()` — positive α, finite constants) in the
  CLI resolve step (`cmd/root.go`, parallel to the `hwConfigByGPUFromBundle` conversion and next to the
  existing `bundle.Validate()` call), producing `map[string]EDPPCoeffs` passed to the decider constructor.
  A missing/unreadable/invalid file errors before the run starts. `bundle.Validate()` may additionally
  reject empty path strings structurally, but the filesystem read stays in the cmd load path (bundle
  parsing does no I/O). Only consumed when `--pd-decider edpp`.

**Snapshot prerequisite.** `buildPoolFilteredSnapshots` (`sim/cluster/cluster.go:1248-1261`) does not set
`snap.GPUType` today (only `buildRouterState` at `cluster_event.go:83,116` does). Add `snap.GPUType =
inst.GPU()` there so decode/prefill candidates reaching `decideJoint` carry their hardware identity. This is
inert when `coeffs_by_gpu` is absent (`coeffsFor` ignores the field and returns `d.coeffs`), preserving
byte-identity.

## 5. Decider changes (`sim/edpp.go`)

**Store + selector.**
- Add `coeffsByGPU map[string]EDPPCoeffs` to the decider struct (nil ⇒ homogeneous).
- `func (d *edppDecider) coeffsFor(gpuType string) EDPPCoeffs` returns `d.coeffsByGPU[gpuType]` if present,
  else `d.coeffs`. An empty `gpuType` (homogeneous / snapshot not carrying it) returns `d.coeffs`.
- `NewEDPPDecider` accepts and stores the map; validates each entry.

**θ selection in the cost math.**
- `jointCandidateCost` (`edpp.go:826-874`): `thetaD := d.coeffsFor(ds.GPUType)` for decode-side terms
  (`tIterDecode`, `muDecode`, `Wd`, `deltaBarDecode`); `thetaP := d.coeffsFor(ps.GPUType)` for prefill-side
  (`tIterPrefill`, `muPrefill`, `Wp`). Replace the `d.coeffs.*` calls accordingly.
- **Move the hoisted candidate-invariant precomputes into the candidate loop.** `wd` (`edpp.go:746`), `mDec`
  (`edpp.go:752`), and `muPNom` (`edpp.go:389`) are precomputed once from the single θ today; under
  per-instance θ they vary by candidate and must be computed per candidate using the selected θ. The
  `edpp.go:750-751` comment already anticipates this ("becomes discriminating under per-instance θ_i").
- **Reduced path** (`Decide`, single fixed d, `edpp.go:485-590`): apply `coeffsFor` to that d's GPU type for
  its terms. The optimization *win* is in joint (choosing d); reduced correctness just avoids scoring a
  heterogeneous d with the wrong θ.

**Determinism (INV-6) — the sensitive part.**
- The deterministic tie-break in the argmin (lowest-index / existing rule) is unchanged; θ selection only
  changes the J *values*, not the enumeration/tie-break order.
- When `coeffsByGPU` is nil (or every candidate's `gpuType` maps to `d.coeffs`), the moved precomputes
  recompute exactly the prior values, so output is byte-identical. This is asserted by a golden test (§7).

## 6. Data flow (end to end)

```
offline:  blis run (single-pool HWConfig bundle, BLIS_STEP_CSV) -> fit_coeffs.py
              -> coeffs-<model>-<gpu>-tp4.json   (consistent with sim physics)

config:   --policy-config bundle.yaml
              node_pools + hw_config_by_gpu  (execution: per-instance HWConfig)
              coeffs_by_gpu                  (decider: per-gpu θ_i file paths)

load:     LoadPolicyBundle -> Validate() -> cmd/root.go loads coeffs_by_gpu
              -> map[string]EDPPCoeffs -> NewEDPPDecider(coeffsByGPU=...)

runtime:  buildPoolFilteredSnapshots sets snap.GPUType = inst.GPU()
              -> decideJoint enumerates (d,p); jointCandidateCost uses
                 thetaD=coeffsFor(ds.GPUType), thetaP=coeffsFor(ps.GPUType)
              -> argmin J picks the speed-aware (d,p)
```

## 7. Testing & validation

**Unit — determinism / fallback (INV-6).**
- With no `coeffs_by_gpu`: `go test ./sim/...` and a golden joint-decision test are byte-identical to
  pre-change (the moved precomputes reproduce prior values).
- `coeffsFor` returns `d.coeffs` for an unmapped/empty gpuType; returns the mapped θ for a mapped one.

**Unit — θ_i is actually consumed (selection guard).**
- A decider built with two distinct GPU→θ entries: for identical candidate snapshot state, `jointCandidateCost`
  (or a full `decideJoint`) yields a **different** J / different chosen d under H100 vs A100 θ — so removing
  the `coeffsFor` selection (reverting to `d.coeffs`) turns the test red. (Analogous to the T-A wiring guard.)

**Integration / acceptance — the T-A saturating bar (pass/fail).**
- Re-run `campaigns/edpp-study/repro_hetero_hw.sh SAT=1` with a `coeffs_by_gpu` bundle (H100 + extracted
  slow-device θ). **PASS = θ_i-joint shifts the realized fast-share from ~77% toward ~86% and goodput from
  ~0.82 toward ~0.96, beating reduced-EDPP and blind load-balance across the 3 seeds.** Record the split
  and goodput per seed in FINDINGS.
- **Under-capacity regime unchanged:** the non-SAT run stays ≥ its current ~0.97 (θ_i must not regress the
  regime joint already wins).

**Regression.**
- Homogeneous single-GPU run byte-identical; `go build ./... && go test ./sim/... ./cmd/...` green; gofmt/lint clean.

## 8. Deliverables

1. `scripts/calibration/repro_theta_by_gpu.sh` + the committed per-GPU coeffs file(s) with diagnostics.
2. `PolicyBundle.CoeffsByGPU` (`coeffs_by_gpu`) + `Validate()` (`sim/bundle.go`).
3. cmd wiring: load `coeffs_by_gpu` → `map[string]EDPPCoeffs` → `NewEDPPDecider` (`cmd/root.go`).
4. `RoutingSnapshot.GPUType` populated in `buildPoolFilteredSnapshots` (`sim/cluster/cluster.go`).
5. Decider: `coeffsByGPU` + `coeffsFor` + per-candidate θ selection + moved precomputes (`sim/edpp.go`).
6. Unit tests (determinism, fallback, selection guard) + the SAT acceptance run recorded in FINDINGS.

## 9. Risks

- **Determinism regression** from moving the hoisted precomputes — mitigated by the golden byte-identity test
  and the nil-map fallback path.
- **Extraction feasibility** (`BLIS_STEP_CSV` under a node-pool bundle) — mitigated by the direct
  latency-model sampling fallback (§3).
- **θ_i insufficient to reach the optimum.** If θ_i-joint improves the split but still undershoots ~86%, that
  is itself a publishable finding (bounds what per-instance work-model knowledge can achieve vs. the residual
  that needs congestion/ITL coupling) — not a failure of the change. The acceptance bar is "beats reduced and
  blind and moves toward the optimum," not "reaches 1.0."
