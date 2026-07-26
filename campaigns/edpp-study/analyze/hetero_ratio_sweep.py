#!/usr/bin/env python3
"""Plot the heterogeneity ratio sweep (structural ablation, experiment 1).

Reads ratio_sweep.csv (columns: ratio,arm,seed,goodput) and produces a two-panel
figure:
  (A) goodput vs decode-theta ratio N, per policy, mean over seeds (band = min..max);
  (B) regret vs the per-N best-static-split oracle.

Color: Okabe-Ito colorblind-safe categorical palette, assigned to policies in a
FIXED order (never cycled). One y-axis per panel. Recessive grid. The two key
contrast series (least-TTFT, drift+VaR) are direct-labeled.

Usage:
  python3 campaigns/edpp-study/analyze/hetero_ratio_sweep.py \
    --csv campaigns/edpp-study/out/hetero_ratio/ratio_sweep.csv \
    --out campaigns/edpp-study/out/hetero_ratio/hetero_ratio_sweep.png
"""
import argparse
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito CVD-safe palette, assigned in fixed order to policies.
PALETTE = {
    "always":     "#999999",  # neutral grey — a fixed corner
    "prefix":     "#999999",  # coincides with always (see paper)
    "kairos":     "#E69F00",  # orange
    "least-ttft": "#D55E00",  # vermilion
    "lt-joint":   "#56B4E9",  # sky blue (hardware-aware least-TTFT)
    "dpp":        "#0072B2",  # blue
    "dpVaR":      "#009E73",  # bluish green (our rule)
}
LABEL = {
    "always": "always", "prefix": "prefix-thr.", "kairos": "Kairos",
    "least-ttft": "least-TTFT", "lt-joint": "least-TTFT-joint",
    "dpp": "drift+penalty", "dpVaR": "drift+VaR",
}
# draw order (behind -> front); our rule + its foil on top.
ORDER = ["always", "prefix", "kairos", "least-ttft", "lt-joint", "dpp", "dpVaR"]


def load(path):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        by[float(r["ratio"])][r["arm"]].append(float(r["goodput"]))
    return by


def agg(vals):
    return sum(vals) / len(vals), min(vals), max(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by = load(args.csv)
    Ns = sorted(by)

    fig, (axg, axr) = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---- Panel A: goodput vs ratio ----
    for arm in ORDER:
        if arm == "prefix":  # coincides with always; skip to avoid double line
            continue
        mean = [agg(by[N][arm])[0] for N in Ns]
        lo = [agg(by[N][arm])[1] for N in Ns]
        hi = [agg(by[N][arm])[2] for N in Ns]
        c = PALETTE[arm]
        lw = 2.4 if arm in ("dpVaR", "dpp") else 1.6
        axg.plot(Ns, mean, marker="o", ms=5, lw=lw, color=c, label=LABEL[arm], zorder=3)
        axg.fill_between(Ns, lo, hi, color=c, alpha=0.12, zorder=1)
    # best-static-split oracle reference
    optm = [agg(by[N]["optimum"])[0] for N in Ns]
    axg.plot(Ns, optm, ls="--", lw=1.4, color="#444444", label="best static split", zorder=2)
    axg.set_xlabel(r"decode-$\theta$ ratio  $N=\theta_{\rm slow}/\theta_{\rm fast}$")
    axg.set_ylabel("goodput")
    axg.set_ylim(0, 1.03)
    axg.set_title("(A) goodput vs heterogeneity ratio")
    axg.grid(True, alpha=0.25, lw=0.6)
    axg.legend(fontsize=8, loc="lower left", framealpha=0.9)

    # ---- Panel B: regret vs best-static-split oracle ----
    for arm in ORDER:
        if arm == "prefix":
            continue
        reg = [agg(by[N]["optimum"])[0] - agg(by[N][arm])[0] for N in Ns]
        c = PALETTE[arm]
        lw = 2.4 if arm in ("dpVaR", "dpp") else 1.6
        axr.plot(Ns, reg, marker="o", ms=5, lw=lw, color=c, zorder=3)
        # direct-label the key contrast series at the right edge
        if arm in ("least-ttft", "lt-joint", "dpVaR"):
            axr.annotate(LABEL[arm], xy=(Ns[-1], reg[-1]), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8,
                         color=c, fontweight="bold")
    axr.axhline(0, color="#888888", lw=0.8, ls=":")
    axr.set_xlabel(r"decode-$\theta$ ratio  $N=\theta_{\rm slow}/\theta_{\rm fast}$")
    axr.set_ylabel("regret vs best static split")
    axr.set_title("(B) worst-case exposure vs ratio")
    axr.grid(True, alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    # print the headline numbers
    print("\nworst-case regret across the ratio range:")
    for arm in ORDER:
        if arm == "prefix":
            continue
        reg = [agg(by[N]["optimum"])[0] - agg(by[N][arm])[0] for N in Ns]
        print(f"  {LABEL[arm]:>14}: {max(reg):.3f}")


if __name__ == "__main__":
    main()
