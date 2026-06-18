#!/usr/bin/env python3
"""Plot composed (chat × code) rate-sweep results.

Reads blis-pd-sweep-rate_composed_threshold_vs_edpp/composed.csv and produces
one heatmap figure per metric. Each figure has 3 subplots (one per arm) with
chat_rate on the x-axis and code_rate on the y-axis. Color encodes the
seed-mean of the metric. Shared color scale per figure for easy cross-arm
comparison.
"""

import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./blis-pd-sweep-rate_composed_threshold_vs_edpp"
CSV_PATH = f"{DATA_DIR}/composed.csv"

ARMS = ["agg-4xtp4", "pd-thresh16", "pd-edpp"]

# (column, title, lower-is-better)
METRICS = [
    ("ttft_p99_ms",     "TTFT p99 (ms)",      True),
    ("ttft_mean_ms",    "TTFT mean (ms)",     True),
    ("e2e_p99_ms",      "E2E p99 (ms)",       True),
    ("e2e_mean_ms",     "E2E mean (ms)",      True),
    ("itl_p99_ms",      "ITL p99 (ms)",       True),
    ("itl_mean_ms",     "ITL mean (ms)",      True),
    ("throughput_rps",  "Throughput (rps)",   False),
    ("disagg_count",    "Disagg requests",    False),
]


def load():
    """Returns {arm: {(chat, code): {col: seed_mean}}} plus sorted axes."""
    rows = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            key = (r["arm"], float(r["chat_rate"]), float(r["code_rate"]))
            rows[key].append(r)

    data = defaultdict(dict)
    chats, codes = set(), set()
    for (arm, chat, code), rs in rows.items():
        chats.add(chat); codes.add(code)
        cell = {}
        for col in rs[0]:
            if col in ("arm", "chat_rate", "code_rate", "seed", "sat_level"):
                continue
            try:
                cell[col] = sum(float(r[col]) for r in rs) / len(rs)
            except ValueError:
                continue
        data[arm][(chat, code)] = cell
    return data, sorted(chats), sorted(codes)


def make_grid(arm_data, chats, codes, col):
    """Returns a (len(codes), len(chats)) array; rows are code (low→high),
    so imshow with origin='lower' shows code increasing upward."""
    g = np.full((len(codes), len(chats)), np.nan)
    for j, c in enumerate(chats):
        for i, co in enumerate(codes):
            cell = arm_data.get((c, co))
            if cell is not None and col in cell:
                g[i, j] = cell[col]
    return g


def plot_metric(data, chats, codes, col, title, lower_better, out):
    grids = [make_grid(data.get(a, {}), chats, codes, col) for a in ARMS]
    valid = np.concatenate([g[~np.isnan(g)] for g in grids if not np.all(np.isnan(g))])
    vmin, vmax = (valid.min(), valid.max()) if valid.size else (0, 1)
    cmap = "viridis_r" if lower_better else "viridis"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    im = None
    for ax, arm, g in zip(axes, ARMS, grids):
        im = ax.imshow(g, origin="lower", aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(chats)))
        ax.set_xticklabels([f"{c:g}" for c in chats], rotation=45, ha="right")
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels([f"{c:g}" for c in codes])
        ax.set_xlabel("chat_rate")
        ax.set_ylabel("code_rate")
        ax.set_title(arm)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if (v - vmin) / max(vmax - vmin, 1e-9) > 0.5 else "black")

    fig.suptitle(title, fontsize=14)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label=f"{title} ({'lower=better' if lower_better else 'higher=better'})")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_edpp_vs_baseline_delta(data, chats, codes, col, title, lower_better, out):
    """Per-cell edpp - threshold16 difference. Numbers carry the metric's
    natural sign; the colormap encodes who wins (blue = edpp wins) so the
    visual story is consistent across metrics regardless of polarity."""
    g_edpp = make_grid(data.get("pd-edpp", {}), chats, codes, col)
    g_t16 = make_grid(data.get("pd-thresh16", {}), chats, codes, col)
    delta = g_edpp - g_t16
    if lower_better:
        cmap = "RdBu_r"  # blue at low end → negative delta (edpp wins) is blue
        cbar_label = f"{title}: edpp − threshold16 (negative = edpp wins)"
    else:
        cmap = "RdBu"    # blue at high end → positive delta (edpp wins) is blue
        cbar_label = f"{title}: edpp − threshold16 (positive = edpp wins)"

    valid = delta[~np.isnan(delta)]
    if valid.size == 0:
        return
    vmax = max(abs(valid.min()), abs(valid.max()))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(delta, origin="lower", aspect="auto",
                   cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(chats)))
    ax.set_xticklabels([f"{c:g}" for c in chats], rotation=45, ha="right")
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([f"{c:g}" for c in codes])
    ax.set_xlabel("chat_rate")
    ax.set_ylabel("code_rate")
    ax.set_title(f"edpp vs threshold16 — {title}")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"No CSV at {CSV_PATH} — run the sweep first.")

    data, chats, codes = load()

    for col, title, lower_better in METRICS:
        plot_metric(data, chats, codes, col, title, lower_better,
                    f"{DATA_DIR}/composed-{col}-heatmap.png")
        if "pd-edpp" in data and "pd-thresh16" in data:
            plot_edpp_vs_baseline_delta(data, chats, codes, col, title, lower_better,
                                        f"{DATA_DIR}/composed-{col}-edpp-vs-thresh16.png")


if __name__ == "__main__":
    main()
