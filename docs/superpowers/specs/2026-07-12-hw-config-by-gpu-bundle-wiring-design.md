# Wire `hw_config_by_gpu` into the policy bundle (Design)

Date: 2026-07-12
Branch: `feat/edpp-estimator-validation`
Related: `campaigns/edpp-study/FINDINGS.md` "T-A" (this is T-A step 1 — the minimal enabling change for a
heterogeneous decode pool, and a prerequisite for sub-project 2 / T-B in `campaigns/edpp-study/TODO.md`
"ROADMAP"); `docs/reference/configuration.md` (documents the `hw_config_by_gpu` YAML shape).

## 1. Goal & scope

Let a `--policy-config` YAML supply per-GPU-type hardware calibration (`TFlopsPeak`/`BwPeakTBs`/MFU/memory)
so that node-pool-placed instances of different `gpu_type` run at genuinely different speeds. Today
`DeploymentConfig.HWConfigByGPU` (`sim/cluster/deployment.go:166`) is consumed
(`sim/cluster/cluster.go:342`, `direct_actuator.go:83`, `infra_lifecycle_event.go:51`) but **never populated
by any non-test code**, and `PolicyBundle` (`sim/bundle.go`, strict parse via `KnownFields(true)`) has no
`hw_config_by_gpu` field — so the documented config cannot actually be loaded. This wires it.

**Config-plumbing only.** No change to the latency model, routing, EDPP, or the decision rule. Empty/omitted
`hw_config_by_gpu` stays backward-compatible (all consumers already guard `if hc, ok := …; ok`).

## 2. Change (mirrors the existing `node_pools` bundle pattern)

1. **`sim/bundle.go` — bundle-local mirror struct + field.** `sim.HardwareCalib`
   (`sim/model_hardware_config.go:69`) has `json:` tags only (CamelCase), so it cannot parse the documented
   snake_case YAML. Add a yaml-tagged mirror, exactly as `NodePoolBundleConfig` mirrors
   `cluster.NodePoolConfig`:
   ```go
   type HardwareCalibBundleConfig struct {
       TFlopsPeak float64 `yaml:"tflops_peak"`
       TFlopsFP8  float64 `yaml:"tflops_fp8"`   // optional; 0 = no native FP8
       BwPeakTBs  float64 `yaml:"bw_peak_tbs"`
       MfuPrefill float64 `yaml:"mfu_prefill"`
       MfuDecode  float64 `yaml:"mfu_decode"`
       MemoryGiB  float64 `yaml:"memory_gib"`   // optional
   }
   ```
   Add to `PolicyBundle`: `HWConfigByGPU map[string]HardwareCalibBundleConfig \`yaml:"hw_config_by_gpu"\``
   (nil = no override, matching the `node_pools` "nil = none" convention).
2. **`sim/bundle.go` `Validate()` — validate at load, not at run.** For each entry: require
   `tflops_peak > 0` and `bw_peak_tbs > 0` (the exact invariant the consumers panic on at run time —
   `cluster.go:344` etc.), and MFU/memory (if set) finite ≥ 0. Return a clear `hw_config_by_gpu[%q]: …`
   error, mirroring the `node_pools[%d] %q: …` validation already in `Validate()`.
3. **`cmd/root.go` — convert + wire.** Build `map[string]sim.HardwareCalib` from `bundle.HWConfigByGPU`
   (field-by-field copy) and set it on the `DeploymentConfig` literal next to `NodePools: bundleNodePools`
   (~line 1955), mirroring the `bundle.NodePools` conversion loop at ~line 1721. Wire on BOTH `run` and
   `replay` if both build a `DeploymentConfig` from the bundle (confirm in situ; match wherever `NodePools`
   is wired). Note: node-pools are already replay-fatal per INV-13, so heterogeneous runs use `blis run`.

## 3. Testing

- **Unit (`sim/bundle_test.go`):** a bundle YAML with `hw_config_by_gpu` (H100: `tflops_peak 1979`,
  `bw_peak_tbs 3.35`; A100: `1248` / `2.0` — the `docs/reference/configuration.md` values) round-trips into
  `PolicyBundle.HWConfigByGPU` with the right float values; `Validate()` REJECTS an entry with
  `tflops_peak <= 0` or `bw_peak_tbs <= 0` with the documented error; strict parse still accepts the field
  (no `KnownFields` error).
- **Integration (`cmd/` or `sim/cluster/`):** a `blis run` (or `NewClusterSimulator`) with a 2-pool bundle
  (H100 + A100, capacities forcing distinct placement) + `hw_config_by_gpu` yields placed instances whose
  per-instance `HWConfig.TFlopsPeak` differs by pool — i.e. two decode-role instances end up on different
  hardware. Assert the two `TFlopsPeak` values are the pool-specified ones (1979 vs 1248). This is the
  concrete proof that heterogeneous serving is now configurable (T-A step 2).
- **Regression:** `go build ./... && go test ./sim/... ./cmd/...` green; a bundle WITHOUT `hw_config_by_gpu`
  is byte-identical to before (empty map → consumers' `ok` guard → no override).

## 4. Deliverables

1. `HardwareCalibBundleConfig` + `PolicyBundle.HWConfigByGPU` + `Validate()` (`sim/bundle.go`).
2. Bundle→`DeploymentConfig.HWConfigByGPU` wiring in `cmd/root.go` (+ `cmd/replay.go` if it wires NodePools).
3. Unit + integration tests per §3.

## 5. Non-goals / next

Per-instance `θ_i` in the EDPP *decider* (still a single global coeff set — the rest of T-B), the
fixed-plan brute-force opportunity test, and the heterogeneous node-pool bundle authoring itself are
SEPARATE follow-ups (T-A steps 3–4 / T-B). This change only makes heterogeneous *serving* configurable.
