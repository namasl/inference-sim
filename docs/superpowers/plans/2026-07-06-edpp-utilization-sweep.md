# EDPP Decode-Pool Utilization Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the single-point Stage C admission-estimator result into a curve — measure decode-pool predicted-vs-realized admission-delay bias across bounded, stationary operating points below capacity.

**Architecture:** Measurement-only, no Go changes. A new analyzer `utilization_sweep.py` (two modes: `scan-verdict` reads a run's saturation level; `aggregate` rolls per-point `admission_ablation.py` outputs + measured ρ̂ + admission-delay drift into one table/figure) and a new `repro_utilization_sweep.sh` (locate λ* by coarse scan → run the ρ grid on the Stage C T1 isolation topology → aggregate). Reuses the shipped `--edpp-admission-trace`, `--post-hoc-detector composite`, and `admission_ablation.py`.

**Tech Stack:** Go 1.22 (build only, no source change), Python 3 + pandas/numpy (matches existing `analyze/` scripts), bash.

## Global Constraints

- Branch: `feat/edpp-estimator-validation` (continues after the Stage C fix-cluster).
- Model / coeffs (verbatim): `meta-llama/llama-3.3-70b-instruct` / `scripts/calibration/coeffs-llama70b-h100-tp4.json`.
- Topology T1 (decode-pool isolation): `--num-instances 2 --prefill-instances 1 --decode-instances 1 --pd-decider edpp --edpp-coeffs <coeffs> --edpp-tau-ttft 2s --edpp-tau-itl 150ms --slo-ttft "batch=2s" --slo-itl "batch=150ms" --edpp-c-xfer 100s` (huge transfer penalty ⇒ all-local on the single decode engine; only EDPP's `Decide` assembles the estimator contexts).
- Base workload spec: `campaigns/edpp-study/specs/synth_rate1.0.yaml` (rewrite only `aggregate_rate`); 5000 requests/point.
- ρ target grid: `0.5, 0.7, 0.85, 0.9, 0.95, 0.98` (denser near saturation). Offered rate = ρ·λ*.
- λ* method: coarse scan over `0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0` using `--post-hoc-detector composite`; λ* = lowest rate NOT classified STABLE. No bisection. λ* reported, never hard-coded.
- Determinism (INV-6): fixed seed (spec `seed: 42`), sorted traces ⇒ byte-identical across runs.
- INV-9: oracle estimator variants are logging-only (guard already in place); no new control-plane read of `OutputTokens`.
- Outputs under `campaigns/edpp-study/out/utilization_sweep/` (out/ gitignored). Tracked artifacts: the two scripts + FINDINGS + README pointer.
- The `docs/superpowers/` dir is gitignored by project convention; the spec/plan are force-added (`git add -f`) to keep them on the branch.

---

### Task 1: `utilization_sweep.py` — aggregation, drift, verdict (with self-check)

**Files:**
- Create: `campaigns/edpp-study/analyze/utilization_sweep.py`
- Test: `campaigns/edpp-study/analyze/test_utilization_sweep.py`

**Interfaces:**
- Consumes (from the shell in Task 2, per point): a run's stdout metrics JSON (fields `responses_per_sec`, `saturation.level`; may be a single object or a per-instance JSON array), an `admission_ablation.py` report JSON (`pools.<pool>.estimators.<variant>.{median_ratio_real_over_pred, median_signed_error_us, realized_p50_us}`), and the raw `--edpp-admission-trace` CSV (`request_id, pool, realized_t_adm, t_adm_pred_*`).
- Produces:
  - `scan-verdict --metrics <json>`: prints the saturation level to stdout; exit code 0 if STABLE, 1 otherwise.
  - `aggregate --sweep-dir <dir> --lambda-star <float> --pool <name> [--out <json>] [--plots <png>] [--warmup <int>] [--ratio-floor-us <float>]`: writes/print the sweep table JSON.
  - File-naming contract the shell must follow: per point `pt_rate<rate>.metrics.json`, `pt_rate<rate>.ablation.json`, `pt_rate<rate>.admission.csv` in `--sweep-dir`.

- [ ] **Step 1: Write the failing self-check test**

```python
# campaigns/edpp-study/analyze/test_utilization_sweep.py
"""Self-check for utilization_sweep.py aggregation, drift, and verdict math.
Run: python3 campaigns/edpp-study/analyze/test_utilization_sweep.py
Exits nonzero on failure. No external test framework (matches analyze/ convention)."""
import json, os, subprocess, sys, tempfile, csv

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "utilization_sweep.py")


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def test_scan_verdict_stable_exit0():
    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "m.json")
        _write(m, json.dumps({"responses_per_sec": 1.0, "saturation": {"level": "STABLE"}}))
        r = subprocess.run([sys.executable, SCRIPT, "scan-verdict", "--metrics", m],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "STABLE" in r.stdout


def test_scan_verdict_overloaded_exit1():
    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "m.json")
        _write(m, json.dumps({"responses_per_sec": 1.0, "saturation": {"level": "OVERLOADED"}}))
        r = subprocess.run([sys.executable, SCRIPT, "scan-verdict", "--metrics", m],
                           capture_output=True, text=True)
        assert r.returncode == 1


def _point(d, rate, respsec, level, realized, pred_waiting):
    # metrics.json
    _write(os.path.join(d, f"pt_rate{rate}.metrics.json"),
           json.dumps({"responses_per_sec": respsec, "saturation": {"level": level}}))
    # ablation.json (only the fields aggregate reads)
    _write(os.path.join(d, f"pt_rate{rate}.ablation.json"), json.dumps({
        "pools": {"local": {"estimators": {
            "waiting": {"median_ratio_real_over_pred": realized / pred_waiting,
                        "median_signed_error_us": realized - pred_waiting,
                        "realized_p50_us": realized},
            "rollforward": {"median_ratio_real_over_pred": 1.0,
                            "median_signed_error_us": 0.0, "realized_p50_us": realized},
        }}}}))
    # admission.csv: 8 rows, second half 2x the first half -> drift 2.0
    with open(os.path.join(d, f"pt_rate{rate}.admission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "pool", "realized_t_adm", "t_adm_pred_waiting"])
        vals = [10, 10, 10, 10, 20, 20, 20, 20]
        for i, v in enumerate(vals):
            w.writerow([f"request_{i}", "local", v, pred_waiting])


def test_aggregate_rho_drift_and_passthrough():
    with tempfile.TemporaryDirectory() as d:
        _point(d, "0.5", respsec=0.5, level="STABLE", realized=100.0, pred_waiting=50.0)
        out = os.path.join(d, "sweep.json")
        r = subprocess.run([sys.executable, SCRIPT, "aggregate", "--sweep-dir", d,
                            "--lambda-star", "1.0", "--pool", "local", "--out", out,
                            "--warmup", "0"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rep = json.load(open(out))
        pt = rep["points"][0]
        assert abs(pt["rho_hat"] - 0.5) < 1e-9              # 0.5 / 1.0
        assert abs(pt["admission_drift"] - 2.0) < 1e-9      # median(20)/median(10)
        assert pt["stationary_verdict"] == "STABLE"
        assert abs(pt["estimators"]["waiting"]["median_ratio_real_over_pred"] - 2.0) < 1e-9
        assert rep["lambda_star"] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all passed")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 campaigns/edpp-study/analyze/test_utilization_sweep.py`
Expected: FAIL — `FileNotFoundError`/nonzero exit because `utilization_sweep.py` does not exist yet.

- [ ] **Step 3: Write `utilization_sweep.py`**

```python
#!/usr/bin/env python3
"""Utilization sweep: decode-pool admission-estimator fidelity vs load.

Two modes:
  scan-verdict --metrics M.json
      Print the post-hoc saturation level from a run's stdout metrics JSON;
      exit 0 if STABLE, 1 otherwise. Used by the λ* capacity scan.
  aggregate --sweep-dir D --lambda-star L --pool local [--out J] [--plots P] [--warmup N] [--ratio-floor-us F]
      Roll per-point admission_ablation.py reports + measured ρ̂ + admission-delay
      drift + saturation verdict into one table. ρ̂ = responses_per_sec / λ*.
      Drift = median(second half)/median(first half) of realized admission delay,
      ordered by numeric request_id (arrival order), after discarding the first N.

File-naming contract in --sweep-dir (written by repro_utilization_sweep.sh):
  pt_rate<rate>.metrics.json  pt_rate<rate>.ablation.json  pt_rate<rate>.admission.csv
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd


def _load(path):
    with open(path) as f:
        return json.load(f)


def _responses_per_sec(metrics):
    # stdout may be a single object or a per-instance array; sum completion rate.
    if isinstance(metrics, list):
        return float(sum(m.get("responses_per_sec", 0.0) for m in metrics))
    return float(metrics.get("responses_per_sec", 0.0))


def _saturation_level(metrics):
    m = metrics[0] if isinstance(metrics, list) else metrics
    sat = m.get("saturation") or {}
    return str(sat.get("level", "UNKNOWN")).upper()


def scan_verdict(args):
    metrics = _load(args.metrics)
    level = _saturation_level(metrics)
    print(level)
    return 0 if "STABLE" in level else 1


def _numeric_id(rid):
    m = re.search(r"(\d+)", str(rid))
    return int(m.group(1)) if m else -1


def _admission_drift(csv_path, pool, warmup):
    df = pd.read_csv(csv_path)
    df = df[df["pool"] == pool].copy()
    if df.empty:
        return None
    df["_ord"] = df["request_id"].map(_numeric_id)
    df = df.sort_values("_ord").iloc[warmup:]
    r = df["realized_t_adm"].to_numpy(dtype=float)
    if len(r) < 4:
        return None
    half = len(r) // 2
    first, second = np.median(r[:half]), np.median(r[half:])
    if first <= 0:
        return None
    return float(second / first)


def aggregate(args):
    points = []
    for ab_path in sorted(glob.glob(os.path.join(args.sweep_dir, "pt_rate*.ablation.json"))):
        rate = re.search(r"pt_rate(.+)\.ablation\.json$", os.path.basename(ab_path)).group(1)
        base = os.path.join(args.sweep_dir, f"pt_rate{rate}")
        metrics = _load(base + ".metrics.json")
        ablation = _load(ab_path)
        pool = ablation.get("pools", {}).get(args.pool, {})
        estimators = pool.get("estimators", {})
        rho_hat = _responses_per_sec(metrics) / args.lambda_star if args.lambda_star > 0 else None
        drift = _admission_drift(base + ".admission.csv", args.pool, args.warmup)
        realized_p50 = None
        for v in estimators.values():
            realized_p50 = v.get("realized_p50_us")
            if realized_p50 is not None:
                break
        # Ratio is unstable when realized admission is near zero; flag it.
        ratio_meaningful = realized_p50 is not None and realized_p50 >= args.ratio_floor_us
        points.append({
            "offered_rate": float(rate),
            "rho_hat": rho_hat,
            "stationary_verdict": _saturation_level(metrics),
            "admission_drift": drift,
            "realized_p50_us": realized_p50,
            "ratio_meaningful": bool(ratio_meaningful),
            "estimators": estimators,
        })
    points.sort(key=lambda p: (p["rho_hat"] is None, p["rho_hat"]))
    report = {"lambda_star": args.lambda_star, "pool": args.pool,
              "ratio_floor_us": args.ratio_floor_us, "warmup": args.warmup, "points": points}
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    if args.plots:
        _plot(report, args.plots)
    return 0


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = ["waiting", "little", "fluid", "rollforward"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        xs, ys = [], []
        for p in report["points"]:
            e = p["estimators"].get(v, {})
            if p["rho_hat"] is not None and e.get("median_ratio_real_over_pred") is not None and p["ratio_meaningful"]:
                xs.append(p["rho_hat"]); ys.append(e["median_ratio_real_over_pred"])
        if xs:
            ax.plot(xs, ys, marker="o", label=v)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("measured utilization ρ̂"); ax.set_ylabel("median realized/predicted")
    ax.set_yscale("log"); ax.legend(); ax.set_title("Decode-pool admission-estimator bias vs load")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sv = sub.add_parser("scan-verdict"); sv.add_argument("--metrics", required=True)
    sv.set_defaults(func=scan_verdict)
    ag = sub.add_parser("aggregate")
    ag.add_argument("--sweep-dir", required=True)
    ag.add_argument("--lambda-star", type=float, required=True)
    ag.add_argument("--pool", default="local")
    ag.add_argument("--out", default="")
    ag.add_argument("--plots", default="")
    ag.add_argument("--warmup", type=int, default=500)
    ag.add_argument("--ratio-floor-us", type=float, default=2000.0)
    ag.set_defaults(func=aggregate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 campaigns/edpp-study/analyze/test_utilization_sweep.py`
Expected: PASS — prints `ok  test_aggregate_rho_drift_and_passthrough`, `ok  test_scan_verdict_overloaded_exit1`, `ok  test_scan_verdict_stable_exit0`, `all passed`.

- [ ] **Step 5: Commit**

```bash
git add campaigns/edpp-study/analyze/utilization_sweep.py campaigns/edpp-study/analyze/test_utilization_sweep.py
git commit -m "feat(edpp): utilization-sweep analyzer (aggregate + drift + scan-verdict)"
```

---

### Task 2: `repro_utilization_sweep.sh` — λ* scan + ρ-grid runs + aggregate

**Files:**
- Create: `campaigns/edpp-study/repro_utilization_sweep.sh`

**Interfaces:**
- Consumes: `utilization_sweep.py` (`scan-verdict`, `aggregate`) and `admission_ablation.py` from Task 1 / Stage C; the T1 flag set from Global Constraints.
- Produces: `out/utilization_sweep/lambda_star.txt` (λ*), per-point `pt_rate<rate>.{metrics.json,ablation.json,admission.csv}`, and `sweep.json` + `sweep.png`.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Reproduce the decode-pool utilization sweep (see FINDINGS.md "Utilization sweep").
#
# Hardens the single-point Stage C result into a curve: measure decode-pool
# admission-estimator bias across bounded, stationary operating points below
# capacity. Topology = Stage C T1 (1P1D, EDPP, huge transfer penalty ⇒ all-local
# on the single decode engine). Steps:
#   1. Locate λ* — coarse-scan aggregate_rate until the composite saturation
#      detector leaves STABLE. λ* = lowest non-STABLE rate.
#   2. Sweep ρ = {0.5,0.7,0.85,0.9,0.95,0.98} → offered rate ρ·λ*; per point emit
#      --edpp-admission-trace + --post-hoc-detector composite; run admission_ablation.py.
#   3. Aggregate → sweep.json (+ sweep.png): bias vs measured ρ̂, drift, verdict.
#
# Usage (from inference-sim/ repo root):  bash campaigns/edpp-study/repro_utilization_sweep.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
BASE_SPEC="campaigns/edpp-study/specs/synth_rate1.0.yaml"
OUT="campaigns/edpp-study/out/utilization_sweep"; mkdir -p "$OUT"
ANALYZE="campaigns/edpp-study/analyze"
SCAN_RATES=(0.1 0.25 0.5 0.75 1.0 1.5 2.0)
RHO_TARGETS=(0.5 0.7 0.85 0.9 0.95 0.98)

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi

T1_COMMON=(--model "$MODEL"
  --num-instances 2 --prefill-instances 1 --decode-instances 1
  --pd-decider edpp --edpp-coeffs "$COEFFS"
  --edpp-tau-ttft 2s --edpp-tau-itl 150ms --slo-ttft "batch=2s" --slo-itl "batch=150ms"
  --edpp-c-xfer 100s --post-hoc-detector composite)

# spec_for_rate <rate> -> path to a temp spec with aggregate_rate rewritten
spec_for_rate() {
  local rate="$1"; local dst="$OUT/spec_rate${rate}.yaml"
  sed -E "s/^aggregate_rate:.*/aggregate_rate: ${rate}/" "$BASE_SPEC" > "$dst"
  echo "$dst"
}

echo "[1/3] locating λ* by coarse scan..." >&2
LAMBDA_STAR=""
for rate in "${SCAN_RATES[@]}"; do
  spec="$(spec_for_rate "$rate")"
  m="$OUT/scan_rate${rate}.metrics.json"
  ./blis run "${T1_COMMON[@]}" --workload-spec "$spec" >"$m" 2>/dev/null
  if python3 "$ANALYZE/utilization_sweep.py" scan-verdict --metrics "$m" >/dev/null; then
    echo "   rate $rate: STABLE" >&2
  else
    echo "   rate $rate: NON-STABLE -> λ* = $rate" >&2
    LAMBDA_STAR="$rate"; break
  fi
done
if [[ -z "$LAMBDA_STAR" ]]; then
  echo "ERROR: all scan rates STABLE; widen SCAN_RATES upward and re-run." >&2; exit 1
fi
echo "$LAMBDA_STAR" > "$OUT/lambda_star.txt"

echo "[2/3] sweeping ρ grid at λ* = $LAMBDA_STAR ..." >&2
for rho in "${RHO_TARGETS[@]}"; do
  rate="$(python3 -c "print(round(${rho}*${LAMBDA_STAR}, 6))")"
  spec="$(spec_for_rate "$rate")"
  echo "   ρ=$rho -> rate=$rate" >&2
  ./blis run "${T1_COMMON[@]}" --workload-spec "$spec" \
    --edpp-admission-trace "$OUT/pt_rate${rate}.admission.csv" \
    >"$OUT/pt_rate${rate}.metrics.json" 2>/dev/null
  python3 "$ANALYZE/admission_ablation.py" --admission "$OUT/pt_rate${rate}.admission.csv" \
    --out "$OUT/pt_rate${rate}.ablation.json" >/dev/null
done

echo "[3/3] aggregating..." >&2
python3 "$ANALYZE/utilization_sweep.py" aggregate --sweep-dir "$OUT" \
  --lambda-star "$LAMBDA_STAR" --pool local \
  --out "$OUT/sweep.json" --plots "$OUT/sweep.png" >/dev/null
echo "done. λ*=$LAMBDA_STAR ; report: $OUT/sweep.json ; figure: $OUT/sweep.png" >&2
```

- [ ] **Step 2: Make executable**

Run: `chmod +x campaigns/edpp-study/repro_utilization_sweep.sh`

- [ ] **Step 3: Run the sweep end-to-end**

Run: `bash campaigns/edpp-study/repro_utilization_sweep.sh`
Expected: prints the λ* it found and per-ρ rates; finishes with `done. λ*=... ; report: .../sweep.json`. Runtime is several `blis run`s (scan + 6 points); each synth run is ~1–2 min, so budget ~15–25 min. If it errors "all scan rates STABLE," widen `SCAN_RATES` upward and re-run (documented in the script).

- [ ] **Step 4: Sanity-check the output**

Run: `python3 -c "import json; r=json.load(open('campaigns/edpp-study/out/utilization_sweep/sweep.json')); [print(round(p['rho_hat'],3), p['stationary_verdict'], round(p['admission_drift'],2) if p['admission_drift'] else None, {k:round(v['median_ratio_real_over_pred'],2) for k,v in p['estimators'].items() if v.get('median_ratio_real_over_pred')}) for p in r['points']]"`
Expected: one line per retained ρ point (ρ̂ ascending). Verify: (a) low-ρ points are STABLE with drift ≈ 1; (b) `waiting`'s ratio rises as ρ̂ → 1 while `fluid`/`rollforward` stay near 1×. **Do not fudge — if the story differs, that is a finding to record in Task 3, not a bug to hide.**

- [ ] **Step 5: Commit**

```bash
git add campaigns/edpp-study/repro_utilization_sweep.sh
git commit -m "feat(edpp): repro_utilization_sweep.sh (λ* scan + ρ-grid decode-pool sweep)"
```

---

### Task 3: FINDINGS section + README pointer + checkpoint

**Files:**
- Modify: `campaigns/edpp-study/FINDINGS.md` (append a "Utilization sweep" section)
- Modify: `campaigns/edpp-study/README.md` (§7 pointer to the sweep)

**Interfaces:**
- Consumes: the `sweep.json` + `sweep.png` produced by Task 2 and the observed λ*.
- Produces: durable documentation with a reproduction checkpoint (no code).

- [ ] **Step 1: Write the FINDINGS "Utilization sweep" section**

Append a section to `campaigns/edpp-study/FINDINGS.md` containing, filled in from the actual `sweep.json`:
- One-paragraph purpose: hardens the Stage C single-point result to bounded/stationary operating points; the saturating point was a non-stationary stress test.
- The measured **λ*** and how it was found (composite detector, lowest non-STABLE scan rate).
- A table: per retained ρ̂ point — `stationary_verdict`, `admission_drift`, `realized_p50_us`, and `median_ratio_real_over_pred` for `waiting / little / fluid / rollforward` (and the signed-error column where the ratio is not meaningful, i.e. `ratio_meaningful=false`).
- The headline sentence (the actual monotonic story the data shows — e.g. "`waiting` degrades from X× at ρ̂≈0.5 to Y× at ρ̂≈0.98 while `fluid`/`rollforward` hold ≈1× at every stationary point").
- A **Reproduction** block: `bash campaigns/edpp-study/repro_utilization_sweep.sh` → `out/utilization_sweep/sweep.json` (+`sweep.png`); note out/ is gitignored; list the checkpoint numbers a correct re-run must reproduce (λ*, and the `waiting` ratio at the top and bottom ρ̂ points) so a future run reveals harness-vs-estimator regression. Note the low-ρ ratio-instability caveat (ratio meaningful only where `realized_p50_us ≥ ratio_floor_us`; read absolute signed error at low ρ).

- [ ] **Step 2: Add the README §7 pointer**

In `campaigns/edpp-study/README.md` §7 (Stage C), add a short subsection or line: the utilization sweep extends the single-point ablation across load; point to `repro_utilization_sweep.sh`, `analyze/utilization_sweep.py`, and the FINDINGS "Utilization sweep" section; restate ρ̂ = achieved throughput / λ* and the two-layer stationarity check (detector verdict + admission-delay drift).

- [ ] **Step 3: Commit**

```bash
git add campaigns/edpp-study/FINDINGS.md campaigns/edpp-study/README.md
git commit -m "docs(edpp): record utilization-sweep result + reproduction checkpoint"
```

---

## Notes for the implementer (confirm-in-situ)

- **Metrics stdout shape:** Task 1's parser handles both a single metrics object and a per-instance array (`_responses_per_sec` sums; `_saturation_level` reads the first). On the 1P1D T1 topology, confirm which shape `blis run` prints to stdout and that `.saturation.level` is populated when `--post-hoc-detector composite` is set (CLAUDE.md: the stdout `saturation` field carries the composite result; the `--saturation-report` file, unused here, carries a BacklogDriftReport on `run`). If stdout mixes log lines with the JSON, redirect logs: the script already uses `2>/dev/null` so stdout is pure JSON — verify.
- **`local` pool rows exist under T1:** Stage C confirmed `--edpp-c-xfer 100s` yields `local` pool rows on the single decode engine. If a future rebuild changes that, the aggregate `--pool local` will show `n=0`; switch `--pool` accordingly and note it.
- **matplotlib availability:** `--plots` imports matplotlib lazily; if unavailable in the environment, run `aggregate` without `--plots` (the JSON is the artifact; the PNG is presentation only).
