#!/usr/bin/env python3
"""Stage B: validate the corrected EDPP work model against per-request realized work.

Reads an --edpp-work-trace CSV and reports relative error of realized vs closed-form
prefill/decode work, and the correction effect vs the old shipped form (basis change
+ cache effect). Expected: ~0 error for single-chunk requests; chunked-prefill residual
equals the documented C_attn·(a_p²−Σs_r²)/2 term.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def rel_err(realized: pd.Series, closed: pd.Series) -> dict:
    d = pd.DataFrame({"r": realized, "c": closed}).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["c"] != 0]
    if d.empty:
        return {"n": 0}
    e = (d["r"] - d["c"]) / d["c"]
    q = lambda p: float(np.percentile(e.abs(), p))
    return {
        "n": int(len(d)),
        "mean_rel_err": float(e.mean()),
        "median_rel_err": float(e.median()),
        "max_abs_rel_err": float(e.abs().max()),
        "abs_rel_err_p90": q(90), "abs_rel_err_p99": q(99),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="--edpp-work-trace CSV")
    ap.add_argument("--out", default="")
    ap.add_argument("--plots", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.work)
    single = df[df["prefill_chunks"] == 1]
    chunked = df[df["prefill_chunks"] > 1]

    report = {
        "total_requests": int(len(df)),
        "single_chunk_prefill": int(len(single)),
        "chunked_prefill": int(len(chunked)),
        "prefill_work_single_chunk": rel_err(single["realized_prefill_work"], single["wp_closed"]),
        "prefill_work_chunked": rel_err(chunked["realized_prefill_work"], chunked["wp_closed"]),
        "decode_work": rel_err(df["realized_decode_work"], df["wd_closed"]),
        "correction_effect": {
            # basis change visible even at cache_hit_frac≈0; cache effect grows with cache_hit_frac.
            "median_wp_over_old": float(
                (df["wp_closed"] / df["wp_closed_nocache_old"].where(df["wp_closed_nocache_old"] != 0, np.nan))
                .median(skipna=True)
            ),
            "median_cache_hit_frac": float(df["cache_hit_frac"].median()),
        },
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
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].scatter(df["wp_closed"], df["realized_prefill_work"], s=6, alpha=0.4)
        ax[0].set_title("prefill: realized vs closed"); ax[0].set_xlabel("wp_closed"); ax[0].set_ylabel("realized")
        ax[1].scatter(df["wd_closed"], df["realized_decode_work"], s=6, alpha=0.4)
        ax[1].set_title("decode: realized vs closed"); ax[1].set_xlabel("wd_closed"); ax[1].set_ylabel("realized")
        for a in ax:
            lim = max(a.get_xlim()[1], a.get_ylim()[1]); a.plot([0, lim], [0, lim], "k--", lw=1)
        fig.tight_layout(); fig.savefig(args.plots, dpi=120)
        print(f"wrote plot to {args.plots}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
