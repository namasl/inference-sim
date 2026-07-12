# Wire `hw_config_by_gpu` into the Policy Bundle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `--policy-config` YAML supply per-GPU-type hardware (`hw_config_by_gpu`) so node-pool-placed instances of different `gpu_type` run at different speeds — populating the consumed-but-never-set `DeploymentConfig.HWConfigByGPU`.

**Architecture:** Mirror the existing `node_pools` bundle pattern: a yaml-tagged `HardwareCalibBundleConfig` in `sim/bundle.go` (because `sim.HardwareCalib` has `json:` tags only), validated in `PolicyBundle.Validate()`, then converted to `map[string]sim.HardwareCalib` and set on the `DeploymentConfig` literal in `cmd/root.go`.

**Tech Stack:** Go 1.22, `gopkg.in/yaml.v3` (strict `KnownFields(true)`), cobra CLI.

## Global Constraints

- Branch: `feat/edpp-estimator-validation`. Spec: `docs/superpowers/specs/2026-07-12-hw-config-by-gpu-bundle-wiring-design.md`.
- **Config-plumbing only** — no change to the latency model, routing, EDPP, or the decision rule.
- **Backward-compatible:** omitted/empty `hw_config_by_gpu` behaves exactly as today (consumers already guard `if hc, ok := …; ok`). A bundle without the field must parse and run byte-identically.
- YAML shape (verbatim, snake_case — from `docs/reference/configuration.md`): `tflops_peak`, `tflops_fp8` (optional), `bw_peak_tbs`, `mfu_prefill`, `mfu_decode`, `memory_gib` (optional). Map key = GPU-type string matching a `node_pools` entry's `gpu_type`.
- Validation invariant (matches the run-time panic at `sim/cluster/cluster.go:344`): each entry needs `tflops_peak > 0` and `bw_peak_tbs > 0`.
- `sim.HardwareCalib` fields (`sim/model_hardware_config.go:69`): `TFlopsPeak, TFlopsFP8, BwPeakTBs, MfuPrefill, MfuDecode, MemoryGiB` (all float64).
- Strict parsing already on: `sim/bundle.go:120` `decoder.KnownFields(true)`.
- Go: `go build ./... && go test ./sim/... ./cmd/...`; gofmt before commit.

---

### Task 1: Bundle struct + field + validation (`sim/bundle.go`)

**Files:**
- Modify: `sim/bundle.go` (add `HardwareCalibBundleConfig`, `PolicyBundle.HWConfigByGPU`, validation in `Validate()`)
- Test: `sim/bundle_test.go`

**Interfaces:**
- Produces: `sim.HardwareCalibBundleConfig` (fields `TFlopsPeak, TFlopsFP8, BwPeakTBs, MfuPrefill, MfuDecode, MemoryGiB float64` with yaml tags); `PolicyBundle.HWConfigByGPU map[string]HardwareCalibBundleConfig`. Task 2 (cmd) converts these to `sim.HardwareCalib`.

- [ ] **Step 1: Write the failing tests**

Add to `sim/bundle_test.go`:
```go
func TestPolicyBundle_HWConfigByGPU_RoundTrip(t *testing.T) {
	yamlSrc := `
scheduler: fcfs
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 4, gpu_memory_gib: 80, initial_nodes: 1, min_nodes: 1, max_nodes: 1}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 1248.0, bw_peak_tbs: 2.0, mfu_prefill: 0.5, mfu_decode: 0.5}
`
	b, err := LoadPolicyBundleFromBytes([]byte(yamlSrc)) // if only LoadPolicyBundle(path) exists, write a temp file (see note)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	h := b.HWConfigByGPU["H100"]
	if h.TFlopsPeak != 1979.0 || h.BwPeakTBs != 3.35 || h.MfuPrefill != 0.5 {
		t.Fatalf("H100 parsed wrong: %+v", h)
	}
	if b.HWConfigByGPU["A100"].TFlopsPeak != 1248.0 {
		t.Fatalf("A100 tflops parsed wrong: %+v", b.HWConfigByGPU["A100"])
	}
}

func TestPolicyBundle_HWConfigByGPU_RejectsNonPositive(t *testing.T) {
	for _, bad := range []string{
		"hw_config_by_gpu:\n  H100: {tflops_peak: 0, bw_peak_tbs: 3.35}\n",
		"hw_config_by_gpu:\n  H100: {tflops_peak: 1979.0, bw_peak_tbs: 0}\n",
	} {
		if _, err := LoadPolicyBundleFromBytes([]byte("scheduler: fcfs\n" + bad)); err == nil {
			t.Fatalf("expected validation error for %q", bad)
		}
	}
}
```
**Note on the loader:** use whatever the existing tests use to load a bundle. If there is a `LoadPolicyBundleFromBytes`/`ParsePolicyBundle` helper, use it; otherwise `LoadPolicyBundle` takes a path — write the YAML to `t.TempDir()` and load that. Confirm in situ (grep `func LoadPolicyBundle` / existing `bundle_test.go` patterns) and match it.

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run 'TestPolicyBundle_HWConfigByGPU' -v`
Expected: FAIL — `HWConfigByGPU` field undefined (compile error), or strict-parse error on the unknown `hw_config_by_gpu` key.

- [ ] **Step 3: Add the struct + field**

In `sim/bundle.go`, near `NodePoolBundleConfig`:
```go
// HardwareCalibBundleConfig mirrors sim.HardwareCalib for YAML loading in PolicyBundle.
// sim.HardwareCalib carries json tags only (CamelCase); this yaml-tagged mirror lets a
// policy-config supply per-GPU hardware in snake_case. Converted to sim.HardwareCalib in cmd/.
type HardwareCalibBundleConfig struct {
	TFlopsPeak float64 `yaml:"tflops_peak"`
	TFlopsFP8  float64 `yaml:"tflops_fp8"` // optional; 0 = no native FP8
	BwPeakTBs  float64 `yaml:"bw_peak_tbs"`
	MfuPrefill float64 `yaml:"mfu_prefill"`
	MfuDecode  float64 `yaml:"mfu_decode"`
	MemoryGiB  float64 `yaml:"memory_gib"` // optional
}
```
Add to the `PolicyBundle` struct (next to `NodePools`):
```go
	HWConfigByGPU map[string]HardwareCalibBundleConfig `yaml:"hw_config_by_gpu"` // nil = no per-GPU override
```

- [ ] **Step 4: Add validation in `Validate()`**

In `PolicyBundle.Validate()`, alongside the existing `node_pools` validation loop:
```go
	for gpu, hc := range b.HWConfigByGPU {
		if !(hc.TFlopsPeak > 0) {
			return fmt.Errorf("hw_config_by_gpu[%q]: tflops_peak must be > 0, got %v", gpu, hc.TFlopsPeak)
		}
		if !(hc.BwPeakTBs > 0) {
			return fmt.Errorf("hw_config_by_gpu[%q]: bw_peak_tbs must be > 0, got %v", gpu, hc.BwPeakTBs)
		}
	}
```
(Use `!(x > 0)` so `NaN` is also rejected. If `Validate` is not the method name / not called by the loader, match the existing validation entry point — confirm in situ.)

- [ ] **Step 5: Run to verify tests pass**

Run: `go test ./sim/ -run 'TestPolicyBundle_HWConfigByGPU' -v && go test ./sim/...`
Expected: PASS; no other `sim` test regresses. gofmt `sim/bundle.go` `sim/bundle_test.go`.

- [ ] **Step 6: Commit**

```bash
git add sim/bundle.go sim/bundle_test.go
git commit -m "feat(config): PolicyBundle.HWConfigByGPU (hw_config_by_gpu) + validation"
```

---

### Task 2: Wire bundle → `DeploymentConfig.HWConfigByGPU` (`cmd/`)

**Files:**
- Modify: `cmd/root.go` (convert `bundle.HWConfigByGPU` → `map[string]sim.HardwareCalib`, set on the `DeploymentConfig` literal); `cmd/replay.go` **iff** it also builds a `DeploymentConfig` with `NodePools` from the bundle.
- Test: `cmd/` integration test (new or extend an existing bundle/node-pool test).

**Interfaces:**
- Consumes: `PolicyBundle.HWConfigByGPU` (Task 1); `sim.HardwareCalib` (`sim/model_hardware_config.go`); `DeploymentConfig.HWConfigByGPU map[string]sim.HardwareCalib` (`sim/cluster/deployment.go:166`).

- [ ] **Step 1: Write the failing integration test**

Add a `cmd/` test (mirror the nearest existing bundle/NodePools test) that builds a cluster from a bundle with two pools (H100 + A100) sized so a 3-instance 1P2D deployment places instances on different pools, plus `hw_config_by_gpu`, and asserts the placed instances carry the pool-specific `TFlopsPeak`. Skeleton:
```go
func TestHWConfigByGPU_Wired_PerInstanceHardwareDiffers(t *testing.T) {
	// bundle: 2 pools (H100 cap for prefill+1 decode, A100 cap for the other decode) + hw_config_by_gpu.
	// Build the DeploymentConfig via the same cmd path that wires NodePools, run/construct the cluster,
	// and collect each instance's HWConfig.TFlopsPeak.
	// EXPECT: the set of per-instance TFlopsPeak values includes BOTH 1979.0 (H100) and 1248.0 (A100)
	//         — i.e. HWConfigByGPU took effect (before this change, all instances shared the CLI --gpu calib).
	...
}
```
Confirm the exact construction entry point in situ (how existing `cmd`/`sim/cluster` tests build a cluster from a bundle + NodePools; reuse it). The assertion that matters: **two instances end up with different `HWConfig.TFlopsPeak` matching the pool specs** — impossible before this wiring.

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./cmd/ -run TestHWConfigByGPU_Wired -v`
Expected: FAIL — instances all share one `TFlopsPeak` (HWConfigByGPU never populated), so the assertion for two distinct pool-specific values fails.

- [ ] **Step 3: Wire the conversion in `cmd/root.go`**

Near the `bundle.NodePools` conversion loop (~`cmd/root.go:1720`), add:
```go
			var bundleHWConfigByGPU map[string]sim.HardwareCalib
			if len(bundle.HWConfigByGPU) > 0 {
				bundleHWConfigByGPU = make(map[string]sim.HardwareCalib, len(bundle.HWConfigByGPU))
				for gpu, hc := range bundle.HWConfigByGPU {
					bundleHWConfigByGPU[gpu] = sim.HardwareCalib{
						TFlopsPeak: hc.TFlopsPeak,
						TFlopsFP8:  hc.TFlopsFP8,
						BwPeakTBs:  hc.BwPeakTBs,
						MfuPrefill: hc.MfuPrefill,
						MfuDecode:  hc.MfuDecode,
						MemoryGiB:  hc.MemoryGiB,
					}
				}
			}
```
Then set it on the `DeploymentConfig` literal next to `NodePools: bundleNodePools` (~line 1955):
```go
			HWConfigByGPU: bundleHWConfigByGPU,
```
Confirm the exact `sim` import alias and the `DeploymentConfig` literal location in situ.

- [ ] **Step 4: Mirror on `replay` if applicable**

Grep `cmd/replay.go` for how it builds its `DeploymentConfig` / whether it wires `bundle.NodePools`. If replay builds a `DeploymentConfig` from the bundle with `NodePools`, apply the same `HWConfigByGPU` conversion+assignment there (INV-13 parity). If replay does NOT support node-pools (they are replay-fatal), note that and skip — do not add a partial path.

- [ ] **Step 5: Run tests + build**

Run: `go test ./cmd/ -run TestHWConfigByGPU_Wired -v && go build -o blis main.go && go test ./sim/... ./cmd/...`
Expected: the integration test PASSES (two distinct pool-specific `TFlopsPeak`); build green; no regression. gofmt changed files.

- [ ] **Step 6: Commit**

```bash
git add cmd/root.go cmd/replay.go cmd/<test file>
git commit -m "feat(config): wire bundle hw_config_by_gpu -> DeploymentConfig.HWConfigByGPU"
```

---

## Notes for the implementer (confirm-in-situ)

- **Bundle loader for the unit test (Task 1 Step 1):** match however `sim/bundle_test.go` currently loads a bundle (`LoadPolicyBundle(path)` vs a bytes/parse helper). If only a path loader exists, write to `t.TempDir()`.
- **`Validate()` entry point (Task 1 Step 4):** confirm the method name and that the loader calls it (so a bad `hw_config_by_gpu` errors at load). Match the existing `node_pools` validation's placement/style.
- **`DeploymentConfig` literal + `sim` alias (Task 2):** confirm the exact line and import alias; the assignment must sit in the same literal that already sets `NodePools: bundleNodePools`.
- **Cluster-construction entry point for the integration test (Task 2 Step 1):** reuse whatever existing `cmd`/`sim/cluster` node-pool tests use to build a cluster from a bundle; the only new assertion is per-instance `HWConfig.TFlopsPeak` divergence.
