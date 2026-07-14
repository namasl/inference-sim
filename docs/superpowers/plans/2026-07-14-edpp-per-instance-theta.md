# Per-instance θ_i in the EDPP joint decider — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the EDPP joint decider score each candidate decode/prefill instance with per-GPU-type coefficients (θ_i), so the drift-plus-penalty argmin proactively over-weights faster hardware toward the goodput-optimal split.

**Architecture:** Add a per-GPU θ store (`map[gpu_type]EDPPCoeffs`) on the decider, selected per candidate inside `jointCandidateCost` via a `coeffsFor(gpuType)` helper (fallback to the single global `d.coeffs`). The class normalizer `W*` stays global (a common reference scale across candidates); only the per-candidate **work/rate** terms use θ_i. Config arrives via a `coeffs_by_gpu` bundle field mirroring the already-merged `hw_config_by_gpu`. The θ files are produced offline by re-running the existing calibration pipeline with the instance served on each target `HWConfig`.

**Tech Stack:** Go 1.22, `gopkg.in/yaml.v3` (strict `KnownFields(true)`), cobra CLI; Python `fit_coeffs.py` for offline coeff fitting; the `campaigns/edpp-study/` bash harness for validation.

## Global Constraints

- Branch: `feat/edpp-estimator-validation`. Spec: `docs/superpowers/specs/2026-07-14-edpp-per-instance-theta-design.md`.
- **Backward-compatible / INV-6 byte-identity:** with no `coeffs_by_gpu` (or all instances one GPU type), every run is byte-identical to today. `coeffsFor` returns `d.coeffs` for every candidate; the precomputes moved into the candidate loop recompute the same float values.
- **Normalizer stays global.** `normFor`/`W*`/`muPNom`/`muDNom` are the class reference scale and are NOT per-instance. θ_i enters only per-candidate work/rate terms (`Wd`, `Wp`, `tIterDecode`, `tIterPrefill`, `muDecode`, `muPrefill`, `deltaBarDecode`, `CPf·chunk`).
- **Determinism (INV-6):** candidate enumeration order and the strict `c.J < best.J-1e-12` tie-break are unchanged; θ selection changes only J values.
- **Keying by `gpu_type`** (not InstanceID); `coeffs_by_gpu` value is a coeffs-file path; nil ⇒ fall back to `--edpp-coeffs`.
- **Deferred (non-goals):** per-instance `Z^I_i` (ITL stays per-class `z_itl`); no roofline-derived θ; no reduced-rule structural change.
- Strict YAML parse (`sim/bundle.go:133` `KnownFields(true)`) must still accept the new field.
- Go: `go build ./... && go test ./sim/... ./cmd/...`; gofmt before every commit. Commit ONLY the files each task names (≈12 untracked scratch files/dirs must stay untracked — never `git add -A`).

**Existing patterns to mirror (from the merged `hw_config_by_gpu` wiring):**
- `sim/bundle.go:27` `HWConfigByGPU map[string]HardwareCalibBundleConfig` field; `Validate()` loop at `sim/bundle.go:386-391`.
- `cmd/root.go:1216` `hwConfigByGPUFromBundle`; wired at `cmd/root.go:1772`; `bundle.Validate()` called at `cmd/root.go:773`.
- `DeploymentConfig.HWConfigByGPU` set on the deployment literal; consumed at `sim/cluster/cluster.go:342`.
- `EDPPConfig.Coeffs` (`sim/edpp.go:80`) fed from `DeploymentConfig.EDPPCoeffs` at `sim/cluster/cluster.go:457`.

---

### Task 1: `coeffs_by_gpu` bundle field + validation (`sim/bundle.go`)

**Files:**
- Modify: `sim/bundle.go` (add `PolicyBundle.CoeffsByGPU`; add a structural-validation loop in `Validate()`)
- Test: `sim/bundle_test.go`

**Interfaces:**
- Produces: `PolicyBundle.CoeffsByGPU map[string]string` (`yaml:"coeffs_by_gpu"`; GPU type → coeffs-file path). Task 6 (cmd) loads each path into `sim.EDPPCoeffs`.

- [ ] **Step 1: Write the failing tests**

Add to `sim/bundle_test.go` (mirror the existing `TestPolicyBundle_HWConfigByGPU_*` tests — use the same loader helper those tests use, e.g. `LoadPolicyBundleFromBytes` or a `t.TempDir()` file; confirm in situ which exists and match it):
```go
func TestPolicyBundle_CoeffsByGPU_RoundTrip(t *testing.T) {
	yamlSrc := `
scheduler: fcfs
coeffs_by_gpu:
  H100: scripts/calibration/coeffs-llama70b-h100-tp4.json
  A100: scripts/calibration/coeffs-llama70b-a100crippled-tp4.json
`
	b, err := loadBundleForTest(t, yamlSrc) // same helper the HWConfigByGPU tests use
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := b.CoeffsByGPU["H100"]; got != "scripts/calibration/coeffs-llama70b-h100-tp4.json" {
		t.Fatalf("H100 path parsed wrong: %q", got)
	}
	if got := b.CoeffsByGPU["A100"]; got != "scripts/calibration/coeffs-llama70b-a100crippled-tp4.json" {
		t.Fatalf("A100 path parsed wrong: %q", got)
	}
}

func TestPolicyBundle_CoeffsByGPU_RejectsEmptyPath(t *testing.T) {
	b, err := loadBundleForTest(t, "scheduler: fcfs\ncoeffs_by_gpu:\n  H100: \"\"\n")
	if err != nil {
		t.Fatalf("load (parse) should succeed: %v", err)
	}
	if err := b.Validate(); err == nil {
		t.Fatalf("expected Validate() error for empty coeffs_by_gpu path")
	}
}

func TestPolicyBundle_CoeffsByGPU_AbsentIsNil(t *testing.T) {
	b, err := loadBundleForTest(t, "scheduler: fcfs\n")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if b.CoeffsByGPU != nil {
		t.Fatalf("expected nil CoeffsByGPU when omitted, got %v", b.CoeffsByGPU)
	}
}
```
If the existing tests use a different loader name, replace `loadBundleForTest` with that exact call in all three tests (do not add a new helper if one already exists).

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run 'TestPolicyBundle_CoeffsByGPU' -v`
Expected: FAIL — `CoeffsByGPU` undefined (compile error), or strict-parse error on unknown `coeffs_by_gpu`.

- [ ] **Step 3: Add the field**

In `sim/bundle.go`, next to `HWConfigByGPU` (line 27):
```go
	CoeffsByGPU       map[string]string                    `yaml:"coeffs_by_gpu"`      // nil = no per-GPU θ override; value = EDPP coeffs file path
```

- [ ] **Step 4: Add validation in `Validate()`**

In `PolicyBundle.Validate()`, immediately after the `HWConfigByGPU` loop (after `sim/bundle.go:391`). Structural only — no filesystem I/O here (bundle parsing does no I/O; the file is loaded+validated in cmd, Task 6):
```go
	for gpu, path := range b.CoeffsByGPU {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("coeffs_by_gpu[%q]: path must not be empty", gpu)
		}
	}
```
Confirm `strings` is imported (add to the import block if not).

- [ ] **Step 5: Run to verify tests pass**

Run: `go test ./sim/ -run 'TestPolicyBundle_CoeffsByGPU' -v && go test ./sim/...`
Expected: PASS; no other `sim` test regresses. gofmt `sim/bundle.go` `sim/bundle_test.go`.

- [ ] **Step 6: Commit**

```bash
git add sim/bundle.go sim/bundle_test.go
git commit -m "feat(config): PolicyBundle.CoeffsByGPU (coeffs_by_gpu) + structural validation"
```

---

### Task 2: Populate `RoutingSnapshot.GPUType` on the EDPP-facing snapshots (`sim/cluster/cluster.go`)

**Files:**
- Modify: `sim/cluster/cluster.go` (`buildPoolFilteredSnapshots`)
- Test: `sim/cluster/cluster_test.go` (or the nearest existing snapshot/pool test file — confirm in situ)

**Interfaces:**
- Produces: decode/prefill `RoutingSnapshot`s reaching the EDPP decider now carry `snap.GPUType = inst.GPU()` (previously empty on this path; only `buildRouterState` set it — `sim/cluster/cluster_event.go:83`). `RoutingSnapshot.GPUType` field exists (`sim/routing.go:24`).

- [ ] **Step 1: Write the failing test**

Add a test that builds a small cluster whose instances have known GPU types and asserts the pool-filtered snapshots carry them. Confirm in situ how existing cluster tests construct a `ClusterSimulator` and reach `buildPoolFilteredSnapshots` (it is unexported — the test lives in package `cluster`). Skeleton:
```go
func TestBuildPoolFilteredSnapshots_CarriesGPUType(t *testing.T) {
	// Build a cluster (reuse the nearest existing helper) with >=1 decode instance
	// whose InstanceSimulator.GPU() returns a known value (e.g. "H100").
	c := newTestClusterWithGPUs(t /* ... */)
	snaps := c.buildPoolFilteredSnapshots(PoolRoleDecode)
	if len(snaps) == 0 {
		t.Fatal("expected >=1 decode snapshot")
	}
	for _, s := range snaps {
		if s.GPUType == "" {
			t.Fatalf("snapshot %s missing GPUType", s.ID)
		}
	}
}
```
Match the real construction helper; the only assertion that matters is `GPUType != ""` on the pool-filtered snapshots.

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/cluster/ -run TestBuildPoolFilteredSnapshots_CarriesGPUType -v`
Expected: FAIL — `GPUType` empty (never set on this path).

- [ ] **Step 3: Set GPUType in the builder**

In `buildPoolFilteredSnapshots`, inside the `for _, inst := range c.instances` loop, right after `snap := c.snapshotProvider.Snapshot(inst.ID(), c.clock)`:
```go
		snap.GPUType = inst.GPU()
```
(Mirror `sim/cluster/cluster_event.go:83`. Inert when `coeffs_by_gpu` is absent — `coeffsFor` ignores it.)

- [ ] **Step 4: Run tests + build**

Run: `go test ./sim/cluster/ -run TestBuildPoolFilteredSnapshots_CarriesGPUType -v && go test ./sim/cluster/...`
Expected: PASS; no regression. gofmt the changed files.

- [ ] **Step 5: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/cluster_test.go
git commit -m "feat(edpp): populate RoutingSnapshot.GPUType on pool-filtered snapshots"
```

---

### Task 3: Decider θ store + `coeffsFor` helper + `EDPPConfig.CoeffsByGPU` (`sim/edpp.go`)

Adds the per-GPU θ store and selector WITHOUT using it in the cost math yet (so behavior stays byte-identical). Tasks 4/5 consume it.

**Files:**
- Modify: `sim/edpp.go` (`EDPPConfig` struct; `EDPPDecider` struct; `NewEDPPDecider`; new `coeffsFor` method)
- Test: `sim/edpp_test.go`

**Interfaces:**
- Consumes: `EDPPCoeffs` (`sim/edpp_coeffs.go:12`).
- Produces: `EDPPConfig.CoeffsByGPU map[string]EDPPCoeffs`; `EDPPDecider.coeffsByGPU map[string]EDPPCoeffs`; method `func (d *EDPPDecider) coeffsFor(gpuType string) EDPPCoeffs`. Task 4/5 call `coeffsFor`; Task 6 (cmd) populates `EDPPConfig.CoeffsByGPU`.

- [ ] **Step 1: Write the failing test**

Add to `sim/edpp_test.go`:
```go
func TestEDPPDecider_CoeffsFor(t *testing.T) {
	base := EDPPCoeffs{AlphaD: 100, AlphaP: 100, C0: 1, C1: 1, CPf: 1, CAttn: 0}
	a100 := EDPPCoeffs{AlphaD: 400, AlphaP: 400, C0: 4, C1: 4, CPf: 4, CAttn: 0}
	d := newTestEDPPDecider(t, EDPPConfig{ // reuse the standard test-config helper; add CoeffsByGPU
		Coeffs:      base,
		CoeffsByGPU: map[string]EDPPCoeffs{"A100": a100},
		// ...the minimal valid fields the existing edpp tests set (Tau*, V, NomPrefillTokens, etc.)
	})
	if got := d.coeffsFor("A100"); got != a100 {
		t.Fatalf("coeffsFor(A100) = %+v, want %+v", got, a100)
	}
	if got := d.coeffsFor("H100"); got != base { // unmapped ⇒ fallback
		t.Fatalf("coeffsFor(unmapped) = %+v, want base %+v", got, base)
	}
	if got := d.coeffsFor(""); got != base { // empty ⇒ fallback
		t.Fatalf("coeffsFor(\"\") = %+v, want base %+v", got, base)
	}
}
```
Confirm in situ the existing helper the edpp tests use to build a decider with a valid config (grep `NewEDPPDecider(` in `sim/edpp_test.go`); reuse it, adding the `CoeffsByGPU` field.

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run TestEDPPDecider_CoeffsFor -v`
Expected: FAIL — `CoeffsByGPU` field and `coeffsFor` undefined (compile error).

- [ ] **Step 3: Add the config field, store, and helper**

In `EDPPConfig` (`sim/edpp.go:66`), after `Coeffs EDPPCoeffs`:
```go
	CoeffsByGPU       map[string]EDPPCoeffs // per-GPU-type θ_i overrides; nil ⇒ use Coeffs for every candidate (homogeneous)
```
In the `EDPPDecider` struct (near `coeffs EDPPCoeffs`, `sim/edpp.go:293`):
```go
	// Per-GPU-type θ_i (design 2026-07-14). nil ⇒ homogeneous: coeffsFor returns coeffs.
	coeffsByGPU map[string]EDPPCoeffs
```
In `NewEDPPDecider`, in the struct literal (alongside `coeffs: cfg.Coeffs`):
```go
		coeffsByGPU: cfg.CoeffsByGPU,
```
And validate each entry defensively at construction (mirrors the `cfg.validate()` panic-on-invalid style, R3) — add right after the existing `cfg.validate()` call at the top of `NewEDPPDecider`:
```go
	for gpu, c := range cfg.CoeffsByGPU {
		if err := c.validate(); err != nil {
			panic(fmt.Sprintf("NewEDPPDecider: coeffs_by_gpu[%q]: %v", gpu, err))
		}
	}
```
(`EDPPCoeffs.validate()` is unexported and in-package — `sim/edpp_coeffs.go:118`. `fmt` is already imported.)
Add the helper (near the other small `EDPPDecider` methods):
```go
// coeffsFor returns the per-GPU-type θ_i for gpuType, or the global coeffs when
// no override exists (nil map, unmapped type, or empty gpuType). This is the single
// selection point for per-instance heterogeneous coefficients; with no coeffs_by_gpu
// it returns d.coeffs for every candidate, preserving byte-identity (INV-6).
func (d *EDPPDecider) coeffsFor(gpuType string) EDPPCoeffs {
	if gpuType != "" {
		if c, ok := d.coeffsByGPU[gpuType]; ok {
			return c
		}
	}
	return d.coeffs
}
```

- [ ] **Step 4: Run to verify tests pass**

Run: `go test ./sim/ -run TestEDPPDecider_CoeffsFor -v && go test ./sim/...`
Expected: PASS; all existing edpp tests still green (the store is unused so far → byte-identical). gofmt `sim/edpp.go` `sim/edpp_test.go`.

- [ ] **Step 5: Commit**

```bash
git add sim/edpp.go sim/edpp_test.go
git commit -m "feat(edpp): per-GPU θ_i store + coeffsFor selector (unused; byte-identical)"
```

---

### Task 4: Per-candidate θ in the joint cost path (`sim/edpp.go`)

The core change. `jointCandidateCost` selects `thetaD` for decode-side terms and `thetaP` for prefill-side terms; the candidate-invariant `wd`/`mDec` (currently hoisted into `jointEvalCtx`) move into the per-candidate cost using `thetaD`; `chunkTerms`' `deltaPf = CPf·chunk` takes θ. Normalizer `W*` stays global.

**Files:**
- Modify: `sim/edpp.go` (`decideJoint` precompute block ~`edpp.go:744-756`; `jointEvalCtx` struct ~`edpp.go:790`; `jointCandidateCost` ~`edpp.go:824-880`; `chunkTerms` ~`edpp.go:695-703`)
- Test: `sim/edpp_test.go`

**Interfaces:**
- Consumes: `coeffsFor` (Task 3); `RoutingSnapshot.GPUType` (Task 2).
- Produces: joint argmin that ranks candidates by their per-GPU physics.

- [ ] **Step 1: Write the failing tests**

Add to `sim/edpp_test.go`. Test A — **byte-identity when homogeneous** (guards INV-6): build a joint decider with NO `CoeffsByGPU`, feed two decode snapshots, assert the chosen decision equals a golden captured from the pre-change code (or, simpler and robust: assert that setting `CoeffsByGPU` to `{H100: coeffs, A100: coeffs}` — i.e. both equal to the global coeffs — yields the identical decision as no `CoeffsByGPU`). Test B — **θ_i is consumed** (guards the feature): two decode snapshots with `GPUType` "H100" and "A100"; with `CoeffsByGPU{"A100": muchSlowerθ}` the argmin must pick the H100 decode node, and removing the `coeffsFor` selection (reverting to `d.coeffs`) must flip/deny that.
```go
func TestDecideJoint_HomogeneousCoeffsByGPU_ByteIdentical(t *testing.T) {
	base := stdTestCoeffs()
	cfg := stdJointConfig(base) // existing helper: joint=true, valid Tau*/V/Nom*, etc.
	dPlain := NewEDPPDecider(cfg, stubModel(), nil, nil)
	cfgDup := cfg
	cfgDup.CoeffsByGPU = map[string]EDPPCoeffs{"H100": base, "A100": base}
	dDup := NewEDPPDecider(cfgDup, stubModel(), nil, nil)

	req := stdReq("batch", 256) // input tokens
	state := stdRouterStateTwoDecode(t, "H100", "A100") // two decode snaps w/ GPUType set
	if got, want := dDup.Decide(req, state), dPlain.Decide(req, state); got != want {
		t.Fatalf("duplicate-θ decision %+v != plain %+v (byte-identity broken)", got, want)
	}
}

func TestDecideJoint_PerInstanceTheta_PrefersFastGPU(t *testing.T) {
	base := stdTestCoeffs()
	slow := base
	slow.AlphaD *= 4; slow.C0 *= 4; slow.C1 *= 4; slow.CPf *= 4 // A100 much slower
	cfg := stdJointConfig(base)
	cfg.CoeffsByGPU = map[string]EDPPCoeffs{"H100": base, "A100": slow}
	d := NewEDPPDecider(cfg, stubModel(), nil, nil)

	req := stdReq("batch", 256)
	// Two decode candidates with IDENTICAL live state but different GPUType, so only θ_i
	// distinguishes them. Put the fast one at the HIGHER instance ID so a tie-break would
	// pick the slow one — proving θ_i (not ordering) drives the choice.
	state := stdRouterStateTwoDecodeCustom(t,
		snap("instance_1", "A100"), snap("instance_2", "H100"))
	dec := d.Decide(req, state)
	if dec.DecodePodOverride != "instance_2" {
		t.Fatalf("expected joint to pick fast H100 instance_2, got %q", dec.DecodePodOverride)
	}
}
```
Confirm/reuse the real test helpers in `sim/edpp_test.go` (grep for how existing joint tests build `RouterState` with decode snapshots — e.g. `TestJoint_ReducesToScorerSliceMatchesReduced`). Set `GPUType` on the snapshots those helpers build.

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./sim/ -run 'TestDecideJoint_(Homogeneous|PerInstanceTheta)' -v`
Expected: Test A may already PASS (θ unused yet); Test B FAILS — argmin ignores θ, picks by tie-break (`instance_1`), not `instance_2`.

- [ ] **Step 3: Thread θ into `chunkTerms`**

`chunkTerms` currently returns `(nChunks, deltaPfChunk = d.coeffs.CPf*chunk)` (`sim/edpp.go:695-703`). `nChunks` is θ-independent; only `deltaPf` depends on `CPf`. Change its signature to take the θ whose `CPf` applies:
```go
// chunkTerms returns (n_chunks, δ_pf-chunk) for a_p uncached tokens, where δ_pf-chunk =
// θ.CPf·chunk is charged on the pool the prefill runs on (local ⇒ decode θ, disagg ⇒ prefill θ).
func (d *EDPPDecider) chunkTerms(theta EDPPCoeffs, ap int) (float64, float64) {
	chunk := ap
	if d.cfg.ChunkTokens > 0 && d.cfg.ChunkTokens < chunk {
		chunk = d.cfg.ChunkTokens
	}
	return math.Ceil(float64(ap) / float64(chunk)), theta.CPf * float64(chunk)
}
```
(If `chunkTerms` is also called from the reduced path, update that call site in Task 5; within Task 4 update only the joint callers below. Grep `d.chunkTerms(` — the joint calls are in `jointCandidateCost`.)

- [ ] **Step 4: Move the candidate-invariant precomputes and add nHatOut to `jointEvalCtx`**

In `jointEvalCtx` (`sim/edpp.go:790`): remove `wd float64` and `jDecodeITL float64`; add `nHatOut float64`. Result:
```go
type jointEvalCtx struct {
	req       *Request
	n         edppNorm
	zTTFT     float64
	zITL      float64
	reqKVNeed int64
	nHatOut   float64 // per-class realized output-length estimate; wd/mDec are now per-candidate (θ_i)
}
```
In `decideJoint` (`sim/edpp.go:744-756`): delete the `wd :=` and `mDec :=`/`jDecodeITL :=` precomputes and the `wd`/`jDecodeITL` fields from the `ec` literal; keep `nHatOut := d.nHatFor(class).mean()` and pass `nHatOut: nHatOut` in `ec`. The comment block explaining m_dec candidate-invariance is now obsolete — replace with a one-line note that wd/mDec are computed per candidate in `jointCandidateCost` under θ_i.

- [ ] **Step 5: Select θ per candidate in `jointCandidateCost`**

In `jointCandidateCost` (`sim/edpp.go:824`), at the top select the decode θ and compute the (now per-candidate) decode work + ITL marginal:
```go
func (d *EDPPDecider) jointCandidateCost(ec *jointEvalCtx, ds RoutingSnapshot, ps *RoutingSnapshot) float64 {
	n := ec.n
	thetaD := d.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)
	wd := thetaD.Wd(len(ec.req.InputTokens), ec.nHatOut)
	mDec := thetaD.deltaBarDecode(float64(len(ec.req.InputTokens)) + ec.nHatOut/2)
	jDecodeITL := ec.zITL * (mDec / n.tauITL)
	_, qdRaw := d.instWorkRaw(ds.ID)
	qd := qdRaw / n.wStarD
	decodeCtx := AdmissionContext{
		QWork: qdRaw, Mu: thetaD.muDecode(bDec, kv, sPfD),
		BatchSize: ds.BatchSize, MaxBatchSize: int(ds.MaxBatchSize),
		FreeKVBlocks: ds.FreeKVBlocks, ReqKVNeed: ec.reqKVNeed,
		TIter: tIterD, QueueDepth: ds.QueueDepth,
		AdmissionRate: admissionRateFromSnapshot(ds), RemainingStepsEst: d.decodeRemStepsEst(ds, ec.req.SLOClass),
		Running: censorOracleRemaining(ds.RunningDecode),
	}
	tAdmD := d.tadmEstimator.EstimateTAdm(decodeCtx)
	jDecodeBacklog := qd * (wd / n.wStarD)
```
Local branch — prefill co-resident on d ⇒ use `thetaD` for `Wp` and `chunkTerms`, and `jDecodeITL` computed above:
```go
	if ps == nil {
		apLoc := d.apForInstance(ec.req, ds.ID)
		nChunksLoc, deltaPfLoc := d.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), len(ec.req.InputTokens))
		tHatLocal := tAdmD + nChunksLoc*(tIterD+deltaPfLoc)
		return jDecodeBacklog +
			qd*(wpLoc/n.wStarD) +
			ec.zTTFT*(tHatLocal/n.tauTTFT) +
			jDecodeITL +
			ec.zITL*(deltaPfLoc/n.tauITL)
	}
```
Disagg branch — prefill on `*ps` ⇒ use `thetaP := d.coeffsFor(ps.GPUType)` for `Wp`, `tIterPrefill`, `muPrefill`, `chunkTerms`:
```go
	thetaP := d.coeffsFor(ps.GPUType)
	apP := d.apForInstance(ec.req, ps.ID)
	nChunksP, deltaPfP := d.chunkTerms(thetaP, apP)
	wpP := thetaP.Wp(maxInt(apP, 0), len(ec.req.InputTokens))
	qpRaw, _ := d.instWorkRaw(ps.ID)
	qp := qpRaw / n.wStarP
	sPfP := ps.ResidentPrefillTokens
	tIterP := thetaP.tIterPrefill(sPfP)
	prefillCtx := AdmissionContext{
		QWork: qpRaw, Mu: thetaP.muPrefill(sPfP),
		BatchSize: ps.BatchSize, MaxBatchSize: int(ps.MaxBatchSize),
		FreeKVBlocks: ps.FreeKVBlocks, ReqKVNeed: ec.reqKVNeed,
		TIter: tIterP, QueueDepth: ps.QueueDepth,
		AdmissionRate: admissionRateFromSnapshot(*ps), RemainingStepsEst: d.prefillRemStepsEst(*ps),
		Running: ps.RunningPrefill,
	}
	tAdmP := d.tadmEstimator.EstimateTAdm(prefillCtx)
	tHatDisagg := tAdmP + nChunksP*(tIterP+deltaPfP) + float64(d.cfg.CXferUs)
	return jDecodeBacklog +
		qp*(wpP/n.wStarP) +
		ec.zTTFT*(tHatDisagg/n.tauTTFT) +
		jDecodeITL +
		d.transferPenalty(n)
```
(`jDecodeBacklog` now uses the per-candidate `wd`, so it genuinely distinguishes decode nodes by speed — the fast node's smaller `Wd` lowers its J.)

- [ ] **Step 6: Run to verify tests pass**

Run: `go test ./sim/ -run 'TestDecideJoint_(Homogeneous|PerInstanceTheta)' -v && go test ./sim/...`
Expected: both new tests PASS; **all existing joint tests still green** (byte-identity under homogeneous θ — the moved precomputes recompute identical values). If any existing joint golden test changed, STOP: the homogeneous path is not byte-identical and must be fixed before proceeding. gofmt.

- [ ] **Step 7: Commit**

```bash
git add sim/edpp.go sim/edpp_test.go
git commit -m "feat(edpp): per-candidate θ_i in jointCandidateCost (thetaD/thetaP); normalizer stays global"
```

---

### Task 5: Reduced path — decode-side θ for the selected d (`sim/edpp.go`)

The reduced rule scores one fixed decode instance `d = state.SelectedInstance`. Apply that instance's θ to the **decode-side** terms only; the prefill side is pool-aggregate (no single prefill instance chosen at Decide time), so it stays on `d.coeffs`. Byte-identical when homogeneous.

**Files:**
- Modify: `sim/edpp.go` (`Decide`, reduced block `edpp.go:485-590`)
- Test: `sim/edpp_test.go`

**Interfaces:**
- Consumes: `coeffsFor` (Task 3); `RoutingSnapshot.GPUType` (Task 2); `d.selectedDecodeSnapshot(state)`.

- [ ] **Step 1: Write the failing test**

```go
func TestDecideReduced_HomogeneousByteIdentical(t *testing.T) {
	base := stdTestCoeffs()
	cfg := stdReducedConfig(base) // joint=false
	dPlain := NewEDPPDecider(cfg, stubModel(), nil, prefillSnapsStub())
	cfgDup := cfg
	cfgDup.CoeffsByGPU = map[string]EDPPCoeffs{"H100": base}
	dDup := NewEDPPDecider(cfgDup, stubModel(), nil, prefillSnapsStub())
	req := stdReq("batch", 256)
	state := stdRouterStateSelected(t, "instance_1", "H100")
	if dDup.Decide(req, state) != dPlain.Decide(req, state) {
		t.Fatal("reduced decision changed under duplicate-θ CoeffsByGPU (byte-identity broken)")
	}
}
```
(A decode-side-θ selection test for reduced is optional — the value is in joint; this task's bar is correctness + byte-identity. Confirm the reduced test helpers in situ.)

- [ ] **Step 2: Run to verify it fails / passes**

Run: `go test ./sim/ -run TestDecideReduced_HomogeneousByteIdentical -v`
Expected: PASS before the change (θ unused in reduced) — this test is the byte-identity guard for the change you are about to make. Keep it; it must stay green after Step 3.

- [ ] **Step 3: Select decode-side θ in the reduced block**

In `Decide` reduced block, after `decSnap, _ := d.selectedDecodeSnapshot(state)` is available (note: `decSnap` is computed at `edpp.go:518` region; if the θ is needed earlier than `decSnap`, resolve the selected snapshot up front). Introduce:
```go
	thetaD := d.coeffsFor(decSnap.GPUType)
```
Replace the **decode-side** `d.coeffs.*` calls in the reduced block with `thetaD.*`:
- `wp := d.coeffs.Wp(ap, len(req.InputTokens))` → `wp := thetaD.Wp(ap, len(req.InputTokens))` (the request's prefill demand, charged against the decode node's balance)
- `muDec := d.coeffs.muDecode(bDec, kv, sPf)` → `thetaD.muDecode(...)`
- `tBminus1 := d.coeffs.tIterDecode(bDec, kv, sPf)` → `thetaD.tIterDecode(...)`
- `deltaPfChunk := d.coeffs.CPf * float64(chunk)` → `thetaD.CPf * float64(chunk)`
- the `decodeCtx.TIter` already uses `tBminus1`; leave `decodeCtx` as is once `tBminus1` uses `thetaD`.

Leave the **prefill-side** on `d.coeffs` (pool-aggregate): `muPf := d.coeffs.muPrefill(sPfPrefill)`, `prefillCtx.TIter: d.coeffs.tIterPrefill(sPfPrefill)`, and `ttftP`'s `d.coeffs.tIterPrefill(sPfPrefill)`. Add a one-line comment: `// prefill side is pool-aggregate (no single p chosen in the reduced rule) → global coeffs`.

Confirm the selected-decode snapshot is resolved before first decode-θ use; if `bDec, kv, sPf := d.selectedDecodeState(state)` is used before `decSnap`, resolve `decSnap`/its GPUType alongside it (both derive from the selected instance).

- [ ] **Step 4: Run tests**

Run: `go test ./sim/ -run TestDecideReduced_HomogeneousByteIdentical -v && go test ./sim/...`
Expected: PASS; all existing reduced tests green (homogeneous ⇒ `thetaD == d.coeffs`). gofmt.

- [ ] **Step 5: Commit**

```bash
git add sim/edpp.go sim/edpp_test.go
git commit -m "feat(edpp): reduced path uses selected decode instance's θ (decode-side; prefill pool-aggregate)"
```

---

### Task 6: Wire `coeffs_by_gpu` bundle → decider (`cmd/root.go`, `sim/cluster`)

Load the bundle's `coeffs_by_gpu` paths into `map[string]sim.EDPPCoeffs` (fail-fast on missing/invalid file) and thread it to `EDPPConfig.CoeffsByGPU`, mirroring the `HWConfigByGPU` flow.

**Files:**
- Modify: `cmd/root.go` (new `edppCoeffsByGPUFromBundle`; set on the deployment literal); `sim/cluster/deployment.go` (add `DeploymentConfig.EDPPCoeffsByGPU`); `sim/cluster/cluster.go:457` (pass into `EDPPConfig`).
- Test: `cmd/` integration test (mirror the `hw_config_by_gpu` wiring test, e.g. `cmd/hwconfig_bundle_wiring_test.go`).

**Interfaces:**
- Consumes: `PolicyBundle.CoeffsByGPU` (Task 1); `sim.LoadEDPPCoeffs` (`sim/edpp_coeffs.go:37`); `EDPPConfig.CoeffsByGPU` (Task 3).
- Produces: a running decider whose `coeffsByGPU` is populated from the bundle.

- [ ] **Step 1: Write the failing integration test**

Add a `cmd/` test that builds a deployment from a bundle carrying `coeffs_by_gpu` (two entries pointing at real coeffs files — reuse the committed H100 file for both keys in this test so it does not depend on Task 7's A100 file) and asserts the resulting `DeploymentConfig.EDPPCoeffsByGPU` carries the loaded `EDPPCoeffs` for each GPU type. Mirror the structure of the `hw_config_by_gpu` wiring test; the assertion that matters: `dc.EDPPCoeffsByGPU["H100"].AlphaD > 0` and both keys present. Removing the wiring line must fail it.

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./cmd/ -run TestCoeffsByGPU_Wired -v`
Expected: FAIL — `EDPPCoeffsByGPU` empty / field undefined.

- [ ] **Step 3: Add the loader + deployment field + wiring**

In `cmd/root.go`, near `hwConfigByGPUFromBundle` (line 1216):
```go
// edppCoeffsByGPUFromBundle loads each coeffs_by_gpu path into EDPPCoeffs (fail-fast:
// a missing/unreadable/invalid file is fatal, matching resolveEDPPCoeffs). Returns nil
// when the bundle omits coeffs_by_gpu (homogeneous).
func edppCoeffsByGPUFromBundle(bundle *sim.PolicyBundle) map[string]sim.EDPPCoeffs {
	if bundle == nil || len(bundle.CoeffsByGPU) == 0 {
		return nil
	}
	out := make(map[string]sim.EDPPCoeffs, len(bundle.CoeffsByGPU))
	for gpu, path := range bundle.CoeffsByGPU {
		c, err := sim.LoadEDPPCoeffs(path)
		if err != nil {
			logrus.Fatalf("coeffs_by_gpu[%q]: %v", gpu, err)
		}
		out[gpu] = c
	}
	return out
}
```
In `sim/cluster/deployment.go`, next to `HWConfigByGPU` (deployment.go:166 region):
```go
	EDPPCoeffsByGPU map[string]sim.EDPPCoeffs // per-GPU-type θ_i for the EDPP decider; nil = homogeneous
```
In `cmd/root.go`, where `bundleHWConfigByGPU` is built (line 1772) and set on the deployment literal, add the parallel:
```go
			bundleEDPPCoeffsByGPU = edppCoeffsByGPUFromBundle(bundle)
```
and on the `DeploymentConfig` literal (next to `HWConfigByGPU: bundleHWConfigByGPU`):
```go
			EDPPCoeffsByGPU: bundleEDPPCoeffsByGPU,
```
In `sim/cluster/cluster.go`, in the `NewEDPPDecider(sim.EDPPConfig{...})` literal at line 457 (next to `Coeffs: config.EDPPCoeffs`):
```go
				CoeffsByGPU:       config.EDPPCoeffsByGPU,
```
(`bundle.Validate()` is already called at `cmd/root.go:773`, so an empty-path `coeffs_by_gpu` is rejected before this loader runs.)

- [ ] **Step 4: Run tests + build**

Run: `go test ./cmd/ -run TestCoeffsByGPU_Wired -v && go build -o blis main.go && go test ./sim/... ./cmd/...`
Expected: integration test PASSES; build green; no regression. gofmt changed files.

- [ ] **Step 5: Commit**

```bash
git add cmd/root.go sim/cluster/deployment.go sim/cluster/cluster.go cmd/<test file>
git commit -m "feat(config): wire bundle coeffs_by_gpu -> EDPPConfig.CoeffsByGPU"
```

---

### Task 7: Offline θ_i extraction — `repro_theta_by_gpu.sh` + slow-device coeffs file (`scripts/calibration/`)

Produce a coeffs file for the slow device, consistent with the simulator's trained-physics execution at that `HWConfig` (design §3).

**Files:**
- Create: `scripts/calibration/repro_theta_by_gpu.sh`
- Create (committed output): `scripts/calibration/coeffs-llama70b-a100crippled-tp4.json`
- Reference: `scripts/calibration/README.md`, `scripts/calibration/repro_llama70b.sh`, `scripts/calibration/fit_coeffs.py`

**Interfaces:**
- Produces: a coeffs JSON in the shape `edppCoeffsJSON` expects (`sim/edpp_coeffs.go:21` — nested `decode`/`prefill` blocks with `alpha_us`, `c0_us_per_req`, `c1_us_per_token`, `alpha_p_us`, `c_pf_us_per_token`, `c_attn_us_per_unit`), loadable by `sim.LoadEDPPCoeffs`.

- [ ] **Step 1: Confirm the extraction route works**

First verify `BLIS_STEP_CSV` taps steps under a single-instance node-pool bundle pinning the target `HWConfig`. Run one short decode calibration `blis run` (from the README) but with `--num-instances 1 --policy-config <onepool A100 bundle>` and `BLIS_STEP_CSV=/tmp/probe.csv`, and check `/tmp/probe.csv` has rows with the expected columns (`step_idx,t_iter_us,b_dec,kv,s_pf,pf_ctx,batch_size`). If it does NOT tap under a node-pool bundle, use the fallback in Step 2b.

- [ ] **Step 2a: Write `repro_theta_by_gpu.sh` (CSV-tap route)**

Generalize `repro_llama70b.sh`: take a GPU label + `HWConfig` (tflops/bw), emit a single-pool `hw_config_by_gpu` bundle for that device, run the D1–D4 + P1–P3 calibration `blis run`s (same flags as the README/`repro_llama70b.sh`) with `--num-instances 1 --policy-config <bundle>` and `BLIS_STEP_CSV`, then `python scripts/calibration/fit_coeffs.py D*.csv P*.csv -o scripts/calibration/coeffs-llama70b-<gpu>-tp4.json`. Print the fit's R² and `cond_*` (must match the H100 file's quality bar — R² ≈ 1, `cond_*` well under 30).

- [ ] **Step 2b: Fallback (only if Step 1 shows no CSV taps under a bundle)**

Write a tiny Go harness (`scripts/calibration/sample_latency_model.go`, `//go:build ignore`) that builds `latency.NewLatencyModel(LatencyCoeffs, HWConfig_target)` and emits the same `step_idx,t_iter_us,b_dec,kv,s_pf,pf_ctx,batch_size` CSV over the README's (b_dec, kv, s_pf) calibration grid, then feed those CSVs to `fit_coeffs.py`. Document in the script header which route was used and why.

- [ ] **Step 3: Produce + sanity-check the file**

Run the script for the T-A synthetic slow device (`tflops_peak: 400, bw_peak_tbs: 0.7`) → `scripts/calibration/coeffs-llama70b-a100crippled-tp4.json`. Sanity check with a tiny Go/CLI load:
```bash
go run ./... # or a one-off: sim.LoadEDPPCoeffs("scripts/calibration/coeffs-llama70b-a100crippled-tp4.json") must not error
```
Assert the slow device's decode/prefill coefficients are LARGER than the H100 file's (slower hardware ⇒ higher µs/token) — a quick `python -c` comparing the two JSONs. Record R²/cond in the script output.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibration/repro_theta_by_gpu.sh scripts/calibration/coeffs-llama70b-a100crippled-tp4.json
# (+ scripts/calibration/sample_latency_model.go if the fallback route was used)
git commit -m "calib(edpp): repro_theta_by_gpu.sh + slow-device θ_i extracted from sim physics"
```

---

### Task 8: Acceptance — θ_i-joint on the saturating harness + record FINDINGS (`campaigns/edpp-study/`)

Re-run the T-A saturating comparison with `coeffs_by_gpu` and record whether θ_i-joint moves toward the optimum.

**Files:**
- Modify: `campaigns/edpp-study/repro_hetero_hw.sh` (add a `THETA=1` variant that passes a `coeffs_by_gpu` bundle to the joint arm)
- Modify: `campaigns/edpp-study/FINDINGS.md` (record the result)

**Interfaces:**
- Consumes: the wired `coeffs_by_gpu` (Task 6), the slow-device coeffs file (Task 7), the joint θ path (Task 4).

- [ ] **Step 1: Extend the harness**

Add a bundle variant to `repro_hetero_hw.sh` that includes `coeffs_by_gpu: {H100: <h100 file>, A100: <a100crippled file>}` alongside the existing `node_pools` + `hw_config_by_gpu`, and a `THETA=1` mode that runs the joint arm (`--pd-decider edpp --edpp-joint`) with that bundle, at the SAT rate/cap, across the same seeds — printing goodput and the realized fast/slow decode split (reuse the existing `split()` stdout parser).

- [ ] **Step 2: Run the acceptance comparison**

Run: `SAT=1 THETA=1 SEEDS="42 7 123" bash campaigns/edpp-study/repro_hetero_hw.sh`
Record, per seed: θ_i-joint goodput + fast-share, alongside the existing reduced / blind-load-balance / homogeneous-joint / fixed-plan-optimum columns.

- [ ] **Step 3: Evaluate against the acceptance bar (design §7)**

PASS = θ_i-joint shifts the realized fast-share from ~77% toward ~86% and goodput from ~0.82 toward ~0.96, beating reduced-EDPP and blind load-balance across the 3 seeds. Also run the under-capacity (non-SAT) arm with the same bundle and confirm it stays ≥ ~0.97 (no regression of the regime joint already wins). If θ_i improves but undershoots ~86%, record that as the finding (design §9: a bound on what per-instance work-model knowledge achieves), not a failure.

- [ ] **Step 4: Record FINDINGS + commit**

Append a "Per-instance θ_i (T-B) result" section to `campaigns/edpp-study/FINDINGS.md` with the per-seed table and the verdict against the bar. Update `campaigns/edpp-study/TODO.md` T-B status.
```bash
git add campaigns/edpp-study/repro_hetero_hw.sh campaigns/edpp-study/FINDINGS.md campaigns/edpp-study/TODO.md
git commit -m "campaign(edpp): T-B per-instance θ_i acceptance result on saturating harness"
```

---

## Notes for the implementer (confirm-in-situ)

- **Bundle loader helper (Task 1):** match however `sim/bundle_test.go`'s existing `TestPolicyBundle_HWConfigByGPU_*` tests load a bundle; reuse that exact call, don't add a new helper.
- **edpp test helpers (Tasks 3–5):** grep `sim/edpp_test.go` for how existing joint/reduced tests build `EDPPConfig`, `RouterState`, decode/prefill snapshots, and call `NewEDPPDecider`; reuse them and set `RoutingSnapshot.GPUType` on the snapshots. The pseudo-helpers named in the tests (`stdTestCoeffs`, `stdJointConfig`, `stdRouterStateTwoDecode`, etc.) are placeholders for the real ones — use whatever exists.
- **`chunkTerms` callers (Task 4/5):** grep `d.chunkTerms(` before changing its signature; update every caller (joint in Task 4; reduced computes `deltaPfChunk` inline in Task 5 rather than via `chunkTerms` — check and keep consistent).
- **`decideJoint` obsolete comment (Task 4):** the `edpp.go:748-751` comment about m_dec being "candidate-invariant under homogeneous θ … becomes discriminating under per-instance θ_i (sub-project 2)" describes exactly this change — update it to past tense / point at the new per-candidate computation.
- **Deployment literal / `sim` alias (Task 6):** confirm the exact `DeploymentConfig` literal location and that `EDPPCoeffsByGPU` is set in the same literal that sets `HWConfigByGPU`.
