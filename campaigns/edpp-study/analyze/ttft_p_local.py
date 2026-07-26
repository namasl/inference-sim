#!/usr/bin/env python3
"""ttft_p estimator study (1P1D disaggregated) — two figures.

Reads the per-(workload, ρ) traces from repro_ttft_p_local.sh and, for disaggregated
requests, derives per estimator (fluid, rollforward):

  ttft        : realized = realized_ttft (outcome) ; est = ttft_p (decision trace)
  prefill_adm : realized = prefill_t_adm (outcome) ; est = t_adm_pred_<est>, pool=prefill
                                                            (admission trace)

Medians per point over disaggregated requests. Two PNGs, one panel per workload,
realized (black) vs fluid vs rollforward over offered load ρ = arrival rate / λ*.
fig_ttft_p is the bottom-line total; fig_prefill_adm is the load-sensitive prefill-pool
admission component (parallel to the ttft_d admission figure).
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

US_PER_MS = 1000.0
ESTIMATORS = ["fluid", "rollforward"]
ECOLOR = {"fluid": "tab:green", "rollforward": "tab:blue"}
METRICS = ["ttft", "prefill_adm"]
MTITLE = {"ttft": "TTFT (disaggregated)", "prefill_adm": "prefill-pool admission delay"}


def _go_bool_true(s):
    return s.astype(str).str.strip().str.lower() == "true"


def _point(dir_, rho):
    """{'rho':x, est: {metric: (realized_ms, est_ms)}} for one disagg load point."""
    tag = os.path.join(dir_, f"rho{rho}")
    try:
        out = pd.read_csv(f"{tag}.outcome.csv")
        adm = pd.read_csv(f"{tag}.admission.csv")
    except FileNotFoundError:
        return None
    out = out[_go_bool_true(out["disaggregated"])][["request_id", "realized_ttft", "prefill_t_adm"]]
    adm = adm[adm["pool"] == "prefill"]   # prefill-pool admission predictions
    base = out.merge(adm, on="request_id", how="inner")
    if base.empty:
        return None

    res = {"rho": float(rho)}
    for est in ESTIMATORS:
        dec_path = f"{tag}.decision.{est}.csv"
        if not os.path.exists(dec_path):
            continue
        dec = pd.read_csv(dec_path)[["request_id", "ttft_p"]]
        d = base.merge(dec, on="request_id", how="inner")
        d = d[d["realized_ttft"] > 0]
        if d.empty:
            continue
        med = lambda s: float(np.median(s)) / US_PER_MS
        res[est] = {
            "ttft":        (med(d["realized_ttft"]), med(d["ttft_p"])),
            "prefill_adm": (med(d["prefill_t_adm"]), med(d[f"t_adm_pred_{est}"])),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--rho", required=True, help="space-separated ρ targets (file tags)")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    rhos = args.rho.split()

    workloads = []
    for d in sorted(glob.glob(os.path.join(args.sweep_root, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "lambda_star.txt")):
            pts = [p for p in (_point(d, r) for r in rhos) if p]
            if pts:
                workloads.append((os.path.basename(d), pts))
    if not workloads:
        raise SystemExit(f"no workload dirs with points under {args.sweep_root}")

    for metric in METRICS:
        fig, axes = plt.subplots(1, len(workloads), figsize=(4.4 * len(workloads), 4.4), squeeze=False)
        for ax, (wl, pts) in zip(axes[0], workloads):
            pts = sorted(pts, key=lambda p: p["rho"])
            xs = [p["rho"] for p in pts]
            realized = [next((p[e][metric][0] for e in ESTIMATORS if e in p), np.nan) for p in pts]
            ax.plot(xs, realized, marker="s", color="k", lw=2, label="realized")
            for est in ESTIMATORS:
                ys = [p[est][metric][1] if est in p else np.nan for p in pts]
                ax.plot(xs, ys, marker="o", color=ECOLOR[est], label=est)
            ax.axvline(1.0, color="grey", lw=0.8, ls=":")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("offered load  ρ = arrival rate / λ*")
            ax.set_title(wl)
            ax.grid(True, which="both", ls=":", alpha=0.4)
        axes[0][0].set_ylabel(f"median {MTITLE[metric]} (ms)")
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"Local {MTITLE[metric]}: realized vs estimated (1P1D disaggregated)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = f"{args.out_prefix}_{'ttft_p' if metric == 'ttft' else 'prefill_adm'}.png"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
