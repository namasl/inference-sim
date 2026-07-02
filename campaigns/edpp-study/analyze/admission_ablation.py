#!/usr/bin/env python3
"""Stage C: admission-delay estimator fidelity ablation.

Reads an --edpp-admission-trace CSV (one row per request×pool with the realized
admission delay and the six estimator predictions) and reports, per pool, the
predicted-vs-realized bias for each estimator, plus the estimator-form vs
N̂_out-prediction error decomposition.

Columns: request_id, pool, realized_t_adm,
  t_adm_pred_{waiting, little, fluid, rollforward, fluid_oracle, rollforward_oracle}
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

VARIANTS = ["waiting", "little", "fluid", "rollforward", "fluid_oracle", "rollforward_oracle"]


def bias(realized: pd.Series, pred: pd.Series) -> dict:
    d = pd.DataFrame({"r": realized, "p": pred}).replace([np.inf, -np.inf], np.nan).dropna()
    # Only rows where the realized admission is meaningfully nonzero (saturated waits);
    # near-zero realized admission has no ratio signal.
    d = d[d["r"] > 1.0]
    if d.empty:
        return {"n": 0}
    ratio = d["r"] / d["p"].where(d["p"] > 0, np.nan)
    err = d["r"] - d["p"]
    q = lambda s, x: float(np.percentile(s, x)) if len(s) else None
    return {
        "n": int(len(d)),
        "median_ratio_real_over_pred": float(ratio.median(skipna=True)),
        "mean_signed_error_us": float(err.mean()),
        "median_signed_error_us": float(err.median()),
        "realized_p50_us": q(d["r"], 50), "realized_p90_us": q(d["r"], 90),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admission", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.admission)
    report = {"total_rows": int(len(df)), "pools": {}}

    for pool, g in df.groupby("pool"):
        pr = {"n_rows": int(len(g)), "estimators": {}}
        for v in VARIANTS:
            col = f"t_adm_pred_{v}"
            if col in g.columns:
                pr["estimators"][v] = bias(g["realized_t_adm"], g[col])
        # Error decomposition (rows with realized>1 and both oracle+deployable positive).
        d = g[g["realized_t_adm"] > 1.0].copy()
        if not d.empty and "t_adm_pred_rollforward" in d.columns:
            form_err = (d["realized_t_adm"] - d["t_adm_pred_rollforward_oracle"]).abs()
            pred_err = (d["t_adm_pred_rollforward_oracle"] - d["t_adm_pred_rollforward"]).abs()
            pr["decomposition"] = {
                "median_form_error_us": float(form_err.median()),
                "median_nhat_pred_error_us": float(pred_err.median()),
            }
        report["pools"][pool] = pr

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
