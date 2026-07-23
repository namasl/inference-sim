#!/usr/bin/env python3
"""Analyze the topology & provisioning matrix (structural ablation, experiment 2).

Reads topo_matrix.csv (topology,prefill,decode,archetype,rate,arm,seed,goodput)
and emits BOTH paper framings:

  A) Topology robustness: worst-case regret per (topology, arm), where the
     reference in each (topology, archetype) cell is the best arm there. Printed
     as a table; drift+VaR's worst-case regret should stay low across topologies,
     matching its 1P2D headline (~0.05). Discharges "one topology per regime".

  B) P:D provisioning adaptivity: goodput vs provisioned P:D ratio per archetype,
     one line per arm. drift+VaR should track the per-provisioning best WITHOUT
     retuning -- the TaiChi-adaptivity answer. Figure: figures/pd_provisioning.

Color: Okabe-Ito CVD-safe palette, policies in FIXED order. One y-axis per panel.

Usage:
  python3 campaigns/edpp-study/analyze/topology_matrix.py \
    --csv campaigns/edpp-study/out/topo_matrix/topo_matrix.csv \
    --out campaigns/edpp-study/out/topo_matrix/pd_provisioning.png
"""
import argparse
import collections
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "never": "#CC79A7", "always": "#999999", "prefix": "#999999",
    "kairos": "#E69F00", "least-ttft": "#D55E00", "dpp": "#0072B2", "dpVaR": "#009E73",
}
LABEL = {
    "never": "never", "always": "always", "prefix": "prefix-thr.", "kairos": "Kairos",
    "least-ttft": "least-TTFT", "dpp": "drift+penalty", "dpVaR": "drift+VaR",
}
ARMS = ["never", "always", "prefix", "kairos", "least-ttft", "dpp", "dpVaR"]
# provisioning order: prefill-lean -> decode-lean (by P:D ratio).
TOPO_ORDER = ["1P3D", "2P2D", "3P1D"]


def load(path):
    # by[topology][archetype][arm] = list of goodputs over seeds
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    rates = {}
    for r in csv.DictReader(open(path)):
        by[r["topology"]][r["archetype"]][r["arm"]].append(float(r["goodput"]))
        rates[r["archetype"]] = r["rate"]
    return by, rates


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def framing_a(by):
    """worst-case regret per (topology, arm) across archetypes."""
    topos = [t for t in TOPO_ORDER if t in by] + [t for t in by if t not in TOPO_ORDER]
    print("\n=== Framing A: worst-case regret per topology (across archetypes) ===")
    header = "topology     " + "  ".join(f"{LABEL[a]:>13}" for a in ARMS if a != "prefix")
    print(header)
    for t in topos:
        archs = list(by[t])
        row = []
        for a in ARMS:
            if a == "prefix":
                continue
            regs = []
            for arch in archs:
                best = max(mean(by[t][arch][x]) for x in ARMS if by[t][arch][x])
                regs.append(best - mean(by[t][arch][a]))
            row.append(max(regs))
        print(f"{t:<12} " + "  ".join(f"{v:>13.3f}" for v in row))


def framing_b(by, rates, out):
    """goodput vs provisioning per archetype; drift+VaR tracks per-provisioning best."""
    topos = [t for t in TOPO_ORDER if t in by]
    archs = sorted({arch for t in by for arch in by[t]},
                   key=lambda a: ["decode", "mixed", "prefill_lean", "prefill_bound"].index(a)
                   if a in ["decode", "mixed", "prefill_lean", "prefill_bound"] else 99)
    x = list(range(len(topos)))
    fig, axes = plt.subplots(1, len(archs), figsize=(3.4 * len(archs), 3.8), sharey=True)
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        for a in ARMS:
            if a == "prefix":
                continue
            y = [mean(by[t][arch][a]) for t in topos]
            lw = 2.4 if a in ("dpVaR", "dpp") else 1.5
            ax.plot(x, y, marker="o", ms=5, lw=lw, color=PALETTE[a], label=LABEL[a], zorder=3)
        # per-provisioning best (dashed)
        best = [max(mean(by[t][arch][a]) for a in ARMS if by[t][arch][a]) for t in topos]
        ax.plot(x, best, ls="--", lw=1.3, color="#444444", label="per-provisioning best", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(topos)
        ax.set_title(f"{arch} (rate {rates.get(arch,'?')})", fontsize=9)
        ax.set_xlabel("provisioning (P:D)")
        ax.grid(True, alpha=0.25, lw=0.6)
    axes[0].set_ylabel("goodput")
    axes[0].set_ylim(0, 1.03)
    axes[-1].legend(fontsize=7, loc="lower center", framealpha=0.9)
    fig.suptitle("P:D provisioning adaptivity (GPU-matched, 4 instances)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    by, rates = load(args.csv)
    framing_a(by)
    framing_b(by, rates, args.out)


if __name__ == "__main__":
    main()
