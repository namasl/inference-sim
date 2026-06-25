#!/usr/bin/env python3
"""Fit the E3 latency-law coefficients from per-step recorder CSVs.

E3 (design doc 2026-06-18-edpp-dpp-routing-design.md):

    T_iter = alpha + c0*B_dec + c1*KV + c_pf*S_pf  (+ c_attn*pf_ctx)

We fit in pure regimes so each coefficient is independently identifiable:

  * decode rows  (s_pf == 0, b_dec > 0):  t_iter ~ 1 + b_dec + kv
        -> alpha (intercept), c0 (b_dec), c1 (kv)
  * prefill rows (b_dec == 0, s_pf > 0):  t_iter ~ 1 + s_pf + pf_ctx
        -> alpha_p (intercept), c_pf (s_pf), c_attn (pf_ctx)

CSV schema (sim/step_recorder.go):
    step_idx, t_iter_us, b_dec, kv, s_pf, pf_ctx, batch_size

Usage:
    fit_coeffs.py decode.csv prefill.csv -o coeffs.json
    fit_coeffs.py decode.csv prefill.csv -o coeffs.json --validate mixed.csv

Dependencies: numpy only.
"""

import argparse
import csv
import json
import sys

import numpy as np

COLS = ["step_idx", "t_iter_us", "b_dec", "kv", "s_pf", "pf_ctx", "batch_size"]


def load(paths):
    """Load and concatenate recorder CSVs into a structured dict of arrays."""
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            r = csv.DictReader(fh)
            missing = set(COLS) - set(r.fieldnames or [])
            if missing:
                sys.exit(f"{p}: missing columns {sorted(missing)}")
            for row in r:
                rows.append([float(row[c]) for c in COLS])
    if not rows:
        sys.exit("no rows loaded")
    a = np.array(rows, dtype=float)
    return {c: a[:, i] for i, c in enumerate(COLS)}


def ols(y, X):
    """Ordinary least squares. Returns (beta, r2). X already includes intercept."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def condition_number(X_no_intercept):
    """Condition number of the (centered, scaled) regressor matrix.

    A large value means the regressors are near-collinear and the individual
    coefficients are not separately identifiable — the design's line-98 risk
    for (b_dec, kv) when KV ~ B*Lbar.
    """
    M = X_no_intercept - X_no_intercept.mean(axis=0)
    scale = np.linalg.norm(M, axis=0)
    scale[scale == 0] = 1.0
    M = M / scale
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def fit_decode(d):
    m = (d["s_pf"] == 0) & (d["b_dec"] > 0)
    n = int(m.sum())
    if n < 4:
        return {"error": f"only {n} pure-decode rows (need >= 4)"}
    bdec, kv, y = d["b_dec"][m], d["kv"][m], d["t_iter_us"][m]
    X = np.column_stack([np.ones(n), bdec, kv])
    beta, r2 = ols(y, X)
    cond = condition_number(np.column_stack([bdec, kv]))
    out = {
        "n_rows": n,
        "alpha_us": beta[0],
        "c0_us_per_req": beta[1],
        "c1_us_per_token": beta[2],
        "r2": r2,
        "cond_b_dec_kv": cond,
    }
    if cond > 30:
        out["WARNING"] = (
            f"b_dec and kv are near-collinear (cond={cond:.0f} > 30): the c0/c1 "
            "split is unreliable. Vary context length independently of batch "
            "size across runs to break KV ~ B*Lbar."
        )
    return out


def fit_prefill(d):
    m = (d["b_dec"] == 0) & (d["s_pf"] > 0)
    n = int(m.sum())
    if n < 4:
        return {"error": f"only {n} pure-prefill rows (need >= 4)"}
    spf, pfctx, y = d["s_pf"][m], d["pf_ctx"][m], d["t_iter_us"][m]
    X = np.column_stack([np.ones(n), spf, pfctx])
    beta, r2 = ols(y, X)
    cond = condition_number(np.column_stack([spf, pfctx]))
    out = {
        "n_rows": n,
        "alpha_p_us": beta[0],
        "c_pf_us_per_token": beta[1],
        "c_attn_us_per_unit": beta[2],
        "r2": r2,
        "cond_s_pf_pf_ctx": cond,
    }
    if cond > 30:
        out["WARNING"] = (
            f"s_pf and pf_ctx are near-collinear (cond={cond:.0f} > 30): vary "
            "prompt length across prefill runs to identify c_pf vs c_attn."
        )
    return out


def validate(d, coeffs):
    """Predict t_iter on mixed rows (b_dec>0 & s_pf>0) and report error,
    stratified by regime. This is the additivity test: coefficients fit in
    isolation, applied where all terms are simultaneously active."""
    dec, pf = coeffs.get("decode", {}), coeffs.get("prefill", {})
    for need, src in ((("alpha_us", "c0_us_per_req", "c1_us_per_token"), dec),
                      (("c_pf_us_per_token", "c_attn_us_per_unit"), pf)):
        if any(k not in src for k in need):
            return {"error": "coeffs incomplete; cannot validate"}
    m = (d["b_dec"] > 0) & (d["s_pf"] > 0)
    n = int(m.sum())
    if n == 0:
        return {"error": "no mixed rows (b_dec>0 & s_pf>0) to validate against"}
    y = d["t_iter_us"][m]
    pred = (
        dec["alpha_us"]
        + dec["c0_us_per_req"] * d["b_dec"][m]
        + dec["c1_us_per_token"] * d["kv"][m]
        + pf["c_pf_us_per_token"] * d["s_pf"][m]
        + pf["c_attn_us_per_unit"] * d["pf_ctx"][m]
    )
    err = pred - y

    def stats(mask):
        k = int(mask.sum())
        if k == 0:
            return {"n": 0}
        e, yy = err[mask], y[mask]
        return {
            "n": k,
            "mape_pct": float(100 * np.mean(np.abs(e / yy))),
            "rmse_us": float(np.sqrt(np.mean(e ** 2))),
        }

    # Regime split by MARGINAL prefill share — c_pf·S_pf relative to the total
    # marginal (non-α) work. We exclude α deliberately: α is additive and shared
    # by both stages, so it tells us nothing about prefill-vs-decode balance and,
    # being large, would otherwise pin every step near decode_bound. The
    # marginal balance is the axis on which E3's additive form diverges from the
    # model's max(compute,memory) truth.
    pf_marg = pf["c_pf_us_per_token"] * d["s_pf"][m]
    dec_marg = dec["c0_us_per_req"] * d["b_dec"][m] + dec["c1_us_per_token"] * d["kv"][m]
    pf_share = pf_marg / np.maximum(pf_marg + dec_marg, 1e-9)
    return {
        "n_mixed": n,
        "classifier": "marginal prefill share = c_pf*S_pf / (c_pf*S_pf + c0*B_dec + c1*KV)",
        "overall": stats(np.ones(n, dtype=bool)),
        "decode_bound": stats(pf_share < 0.25),
        "near_ridge": stats((pf_share >= 0.25) & (pf_share <= 0.75)),
        "prefill_bound": stats(pf_share > 0.75),
        "note": "MAPE expected to rise in near_ridge: E3 is a local "
                "linearization of max(compute,memory) (design line 121).",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="recorder CSV(s) for fitting")
    ap.add_argument("-o", "--out", help="write coefficients JSON to this path")
    ap.add_argument("--validate", metavar="MIXED_CSV", nargs="+",
                    help="one or more mixed-batch CSVs; predict t_iter on their "
                         "mixed rows (pooled) and report error by regime")
    args = ap.parse_args()

    d = load(args.csv)
    coeffs = {
        "units": {"time": "microseconds", "tokens": "tokens", "requests": "count"},
        "source_csvs": args.csv,
        "total_rows": int(len(d["t_iter_us"])),
        "decode": fit_decode(d),
        "prefill": fit_prefill(d),
    }
    if args.validate:
        coeffs["validation"] = validate(load(args.validate), coeffs)

    text = json.dumps(coeffs, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)

    for half in ("decode", "prefill"):
        w = coeffs[half].get("WARNING")
        if w:
            print(f"\n[{half}] {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
