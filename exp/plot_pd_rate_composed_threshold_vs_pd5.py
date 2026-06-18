#!/usr/bin/env python3
"""Plot composed (chat × code × arrival) rate-sweep results for pd5.

Reads blis-pd-sweep-rate_composed_threshold_vs_pd5/composed.csv and
produces, for each metric, one heatmap-grid figure per arrival process
(poisson, weibull, ...). Each figure has 3 subplots (one per arm) with
chat_rate on the x-axis and code_rate on the y-axis. Color encodes the
seed-mean of the metric. Shared color scale per figure for cross-arm
comparison.

Also produces, per arrival process, a per-cell delta heatmap of pd5 vs
pd-thresh16 — the campaign's primary comparison.
"""

import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./blis-pd-sweep-rate_composed_threshold_vs_pd5"
CSV_PATH = f"{DATA_DIR}/composed.csv"

ARMS = ["agg-4xtp4", "pd-thresh16", "pd-pd5"]

# (column, title, lower-is-better)
METRICS = [
    ("goodput",         "Goodput-at-SLO",     False),
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
    """Returns {arrival: {arm: {(chat, code): {col: seed_mean}}}} plus axes."""
    rows = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            key = (r["arrival"], r["arm"], float(r["chat_rate"]), float(r["code_rate"]))
            rows[key].append(r)

    data = defaultdict(lambda: defaultdict(dict))
    chats, codes, arrivals = set(), set(), set()
    for (arrival, arm, chat, code), rs in rows.items():
        arrivals.add(arrival)
        chats.add(chat)
        codes.add(code)
        cell = {}
        for col in rs[0]:
            if col in ("arm", "chat_rate", "code_rate", "seed", "arrival", "sat_level"):
                continue
            try:
                cell[col] = sum(float(r[col]) for r in rs) / len(rs)
            except ValueError:
                continue
        data[arrival][arm][(chat, code)] = cell
    return data, sorted(chats), sorted(codes), sorted(arrivals)


def make_grid(arm_data, chats, codes, col):
    """(len(codes), len(chats)) array; rows are code (low→high)."""
    g = np.full((len(codes), len(chats)), np.nan)
    for j, c in enumerate(chats):
        for i, co in enumerate(codes):
            cell = arm_data.get((c, co))
            if cell is not None and col in cell:
                g[i, j] = cell[col]
    return g


def annotate(ax, g, vmin, vmax):
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g[i, j]
            if np.isnan(v):
                continue
            if abs(v) >= 10:
                txt = f"{v:.0f}"
            elif abs(v) >= 1:
                txt = f"{v:.2f}"
            else:
                txt = f"{v:.3f}"
            shade = (v - vmin) / max(vmax - vmin, 1e-9)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if shade > 0.5 else "black")


def plot_metric(arrival, data, chats, codes, col, title, lower_better, out):
    grids = [make_grid(data.get(a, {}), chats, codes, col) for a in ARMS]
    valid = np.concatenate([g[~np.isnan(g)] for g in grids if not np.all(np.isnan(g))])
    if valid.size == 0:
        return
    vmin, vmax = valid.min(), valid.max()
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
        annotate(ax, g, vmin, vmax)

    fig.suptitle(f"{title} — arrival={arrival}", fontsize=14)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label=f"{title} ({'lower=better' if lower_better else 'higher=better'})")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_pd5_vs_baseline_delta(arrival, data, chats, codes, col, title,
                               lower_better, baseline_arm, out):
    """Per-cell pd5 - <baseline> delta heatmap, signed so blue = pd5 wins."""
    g_pd5 = make_grid(data.get("pd-pd5", {}), chats, codes, col)
    g_base = make_grid(data.get(baseline_arm, {}), chats, codes, col)
    delta = g_pd5 - g_base
    if lower_better:
        cmap = "RdBu_r"
        cbar_label = f"{title}: pd5 − {baseline_arm} (negative = pd5 wins)"
    else:
        cmap = "RdBu"
        cbar_label = f"{title}: pd5 − {baseline_arm} (positive = pd5 wins)"

    valid = delta[~np.isnan(delta)]
    if valid.size == 0:
        return
    vmax = max(abs(valid.min()), abs(valid.max())) or 1e-9

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(delta, origin="lower", aspect="auto",
                   cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(chats)))
    ax.set_xticklabels([f"{c:g}" for c in chats], rotation=45, ha="right")
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([f"{c:g}" for c in codes])
    ax.set_xlabel("chat_rate")
    ax.set_ylabel("code_rate")
    ax.set_title(f"pd5 vs {baseline_arm} — {title} — arrival={arrival}")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}" if abs(v) < 1 else f"{v:+.1f}",
                        ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"No CSV at {CSV_PATH} — run the sweep first.")

    data, chats, codes, arrivals = load()

    for arrival in arrivals:
        arrival_data = data[arrival]
        for col, title, lower_better in METRICS:
            plot_metric(arrival, arrival_data, chats, codes, col, title, lower_better,
                        f"{DATA_DIR}/composed-{col}-{arrival}-heatmap.png")
            if "pd-pd5" in arrival_data and "pd-thresh16" in arrival_data:
                plot_pd5_vs_baseline_delta(
                    arrival, arrival_data, chats, codes, col, title, lower_better,
                    "pd-thresh16",
                    f"{DATA_DIR}/composed-{col}-{arrival}-pd5-vs-thresh16.png")
            if "pd-pd5" in arrival_data and "agg-4xtp4" in arrival_data:
                plot_pd5_vs_baseline_delta(
                    arrival, arrival_data, chats, codes, col, title, lower_better,
                    "agg-4xtp4",
                    f"{DATA_DIR}/composed-{col}-{arrival}-pd5-vs-agg.png")


if __name__ == "__main__":
    main()
