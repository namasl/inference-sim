#!/usr/bin/env python3
"""ttft_d local estimator study — three-figure decomposition (admission / prefill / ttft).

Reads the per-(workload, ρ) traces written by repro_ttft_d_local.sh and, for each load
point, joins the decision / admission / outcome traces per request_id (local requests
only) to derive, per estimator (fluid, rollforward):

  admission : realized = local_t_adm            ; est = t_adm_pred_<est>   (admission trace)
  ttft      : realized = realized_ttft           ; est = ttft_d            (decision trace)
  prefill   : realized = realized_ttft − local_t_adm ; est = ttft_d − t_adm_pred_<est>

Medians are taken per request then per point (not median-of-ratios). Emits three PNGs,
each with one panel per workload: realized (black) vs fluid vs rollforward, over measured
utilization ρ̂. The three figures add up: ttft ≈ admission + prefill.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

US_PER_MS = 1000.0
ESTIMATORS = ["fluid", "rollforward"]
ECOLOR = {"fluid": "tab:green", "rollforward": "tab:blue"}
# metric -> (realized column expr, estimate expr) handled inline in _point.
METRICS = ["admission", "prefill", "ttft"]
MTITLE = {"admission": "admission delay", "prefill": "prefill time", "ttft": "TTFT"}


def _go_bool_false(s):
    return s.astype(str).str.strip().str.lower() == "false"


def _point(dir_, rho, lam):
    """Return {'rho':x, est: {metric: (realized_ms, est_ms)}} for one load point.

    x is OFFERED load intensity ρ = arrival_rate / λ* (the file tag), NOT measured
    throughput / λ*: throughput plateaus at capacity in overload, so throughput/λ*
    asymptotes to 1 and cannot represent the ρ>1 overload points this sweep includes.
    """
    tag = os.path.join(dir_, f"rho{rho}")
    try:
        out = pd.read_csv(f"{tag}.outcome.csv")
        adm = pd.read_csv(f"{tag}.admission.csv")
    except FileNotFoundError:
        return None
    rho_offered = float(rho)

    # Realized admission comes from the ADMISSION trace (realized_t_adm), NOT the outcome
    # trace: the --pd-outcome-trace path hardcodes local_t_adm=0 for local requests
    # (cluster.go BuildPDOutcomeRecords), whereas the admission trace computes it properly
    # from localEnqueueTimes/localAdmitTimes.
    out = out[_go_bool_false(out["disaggregated"])][["request_id", "realized_ttft"]]
    adm = adm[adm["pool"] == "local"]
    base = out.merge(adm, on="request_id", how="inner")
    if base.empty:
        return None

    res = {"rho": rho_offered}
    for est in ESTIMATORS:
        dec_path = f"{tag}.decision.{est}.csv"
        if not os.path.exists(dec_path):
            continue
        dec = pd.read_csv(dec_path)[["request_id", "ttft_d"]]
        d = base.merge(dec, on="request_id", how="inner").copy()
        d = d[d["realized_ttft"] > 0]
        if d.empty:
            continue
        real_tadm = d["realized_t_adm"]
        est_tadm = d[f"t_adm_pred_{est}"]
        real_pref = d["realized_ttft"] - real_tadm
        est_pref = d["ttft_d"] - est_tadm
        med = lambda s: float(np.median(s)) / US_PER_MS
        res[est] = {
            "admission": (med(real_tadm), med(est_tadm)),
            "prefill":   (med(real_pref), med(est_pref)),
            "ttft":      (med(d["realized_ttft"]), med(d["ttft_d"])),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--rho", required=True, help="space-separated ρ targets used as file tags")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    rhos = args.rho.split()

    workloads = []
    for d in sorted(glob.glob(os.path.join(args.sweep_root, "*"))):
        lf = os.path.join(d, "lambda_star.txt")
        if os.path.isdir(d) and os.path.exists(lf):
            lam = float(open(lf).read().strip())
            pts = [p for p in (_point(d, r, lam) for r in rhos) if p]
            if pts:
                workloads.append((os.path.basename(d), pts))
    if not workloads:
        raise SystemExit(f"no workload dirs with points under {args.sweep_root}")

    for metric in METRICS:
        fig, axes = plt.subplots(1, len(workloads), figsize=(4.4 * len(workloads), 4.4), squeeze=False)
        for ax, (wl, pts) in zip(axes[0], workloads):
            pts = sorted(pts, key=lambda p: p["rho"])
            xs = [p["rho"] for p in pts]
            # realized is estimator-independent; take it from whichever estimator is present.
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
        fig.suptitle(f"Local {MTITLE[metric]}: realized vs estimated (single collocated instance)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = f"{args.out_prefix}_{metric}.png"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
