#!/usr/bin/env python3
"""Stage A: validate shipped EDPP forward TTFT estimators against realized outcomes.

Joins a --pd-outcome-trace CSV with an --edpp-decision-trace CSV on request_id and
reports estimator bias (predicted vs realized TTFT, and predicted admission component
vs realized T_adm), split by disaggregated x slo_class, over completed requests.

The shipped estimator does not log its admission component directly; we reconstruct it
from the decision trace as the queue-wait term: qp_raw/mu_p_nom (prefill side) and
qd_raw/mu_d_nom (decode side).
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def parse_go_bool(s: pd.Series) -> pd.Series:
    """Parse a Go-serialized bool column ("true"/"false" strings) into Python bools.

    The outcome CSV serializes Go bools as the strings "true"/"false", which pandas
    reads as object/string dtype. A naive `== True` comparison never matches those
    strings, so we normalize explicitly (case-insensitively, trimmed).
    """
    return s.astype(str).str.strip().str.lower() == "true"


def bias_stats(pred: pd.Series, real: pd.Series) -> dict:
    d = pd.DataFrame({"pred": pred, "real": real}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return {"n": 0}
    err = d["real"] - d["pred"]
    ratio = d["real"] / d["pred"].where(d["pred"] != 0, np.nan)
    q = lambda s, p: float(np.percentile(s, p)) if len(s) else None
    return {
        "n": int(len(d)),
        "mean_signed_error": float(err.mean()),
        "median_signed_error": float(err.median()),
        "median_ratio_real_over_pred": float(ratio.median(skipna=True)),
        "pred_p50": q(d["pred"], 50), "pred_p90": q(d["pred"], 90), "pred_p99": q(d["pred"], 99),
        "real_p50": q(d["real"], 50), "real_p90": q(d["real"], 90), "real_p99": q(d["real"], 99),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", required=True, help="--pd-outcome-trace CSV")
    ap.add_argument("--decision", required=True, help="--edpp-decision-trace CSV")
    ap.add_argument("--out", default="", help="write JSON report here (default stdout)")
    ap.add_argument("--plots", default="", help="write predicted-vs-realized scatter PNG here")
    args = ap.parse_args()

    out = pd.read_csv(args.outcome)
    dec = pd.read_csv(args.decision)

    # Go bools serialize as the strings "true"/"false" in the CSV; normalize to real
    # booleans so the completed-filter and disaggregated groupby work correctly.
    out["completed"] = parse_go_bool(out["completed"])
    out["disaggregated"] = parse_go_bool(out["disaggregated"])

    df = out.merge(dec, on="request_id", how="inner", suffixes=("", "_dec"))

    if len(df) != len(out):
        print(f"warning: inner join dropped {len(out) - len(df)} outcome rows "
              f"({len(out)} outcome vs {len(df)} joined); truncated_or_dropped assumes a total join match",
              file=sys.stderr)

    total = len(out)
    completed = df[df["completed"]].copy()
    truncated = total - int(out["completed"].sum())

    # Local (non-disaggregated) requests have an empty slo_class, which pandas reads
    # as NaN. groupby silently drops NaN keys, so fill them into their own group.
    completed["slo_class"] = completed["slo_class"].fillna("unknown").replace("", "unknown")

    # Reconstruct admission components from the decision trace (µs).
    completed["pred_prefill_adm"] = completed["qp_raw"] / completed["mu_p_nom"].where(completed["mu_p_nom"] != 0, np.nan)
    completed["pred_decode_adm"] = completed["qd_raw"] / completed["mu_d_nom"].where(completed["mu_d_nom"] != 0, np.nan)

    report = {"total_requests": int(total), "completed": int(len(completed)), "truncated_or_dropped": int(truncated), "groups": {}}
    for (disagg, cls), g in completed.groupby(["disaggregated", "slo_class"]):
        disagg = bool(disagg)
        key = f"disagg={disagg},class={cls}"
        # ttft_p validates disagg-path TTFT; ttft_d validates the local alternative.
        ttft_pred = g["ttft_p"] if disagg else g["ttft_d"]
        report["groups"][key] = {
            "ttft_pred_vs_realized": bias_stats(ttft_pred, g["realized_ttft"]),
            "prefill_adm_pred_vs_realized": bias_stats(g["pred_prefill_adm"], g["prefill_t_adm"]) if disagg else {"n": 0},
            "decode_adm_pred_vs_realized": bias_stats(g["pred_decode_adm"], g["decode_t_adm"]) if disagg else {"n": 0},
        }

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        pred = np.where(completed["disaggregated"], completed["ttft_p"], completed["ttft_d"])
        ax.scatter(pred, completed["realized_ttft"], s=6, alpha=0.4)
        lim = max(float(np.nanmax(pred)), float(completed["realized_ttft"].max()))
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
        ax.set_xlabel("predicted TTFT (µs)"); ax.set_ylabel("realized TTFT (µs)")
        ax.set_title("EDPP predicted vs realized TTFT"); ax.legend()
        fig.tight_layout(); fig.savefig(args.plots, dpi=120)
        print(f"wrote plot to {args.plots}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
