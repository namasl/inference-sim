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
