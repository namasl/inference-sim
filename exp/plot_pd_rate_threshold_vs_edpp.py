#!/usr/bin/env python3
"""Plot rate-sweep results from blis-pd-sweep-rate_threshold_vs_edpp/.

One figure per workload. Three arms (agg-4xtp4, pd-thresh16, pd-edpp) compared
across rates. Color encodes arm; linestyle encodes percentile. Seeds are
collapsed to per-(arm, rate) mean.
"""

import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DATA_DIR = "./blis-pd-sweep-rate_threshold_vs_edpp"
WORKLOADS = ["interactive-chat", "code-generation"]

# (arm label, color, legend description)
ARMS = [
    ("agg-4xtp4",   "tab:gray",   "agg-4xtp4 (4×TP=4, no PD)"),
    ("pd-thresh16", "tab:blue",   "pd-thresh16 (1P+3D, prefix-threshold=16)"),
    ("pd-edpp",     "tab:red",    "pd-edpp (1P+3D, edpp decider)"),
]

METRIC_GROUPS = [
    ("TTFT (ms)", [
        ("ttft_mean_ms", "mean", "-"),
        ("ttft_p90_ms",  "p90",  "-."),
        ("ttft_p95_ms",  "p95",  "--"),
        ("ttft_p99_ms",  "p99",  ":"),
    ]),
    ("E2E (ms)", [
        ("e2e_mean_ms", "mean", "-"),
        ("e2e_p90_ms",  "p90",  "-."),
        ("e2e_p95_ms",  "p95",  "--"),
        ("e2e_p99_ms",  "p99",  ":"),
    ]),
    ("ITL (ms)", [
        ("itl_mean_ms", "mean", "-"),
        ("itl_p90_ms",  "p90",  "-."),
        ("itl_p95_ms",  "p95",  "--"),
        ("itl_p99_ms",  "p99",  ":"),
    ]),
]
SCALAR_METRICS = [
    ("throughput_rps", "Throughput (rps)"),
    ("disagg_count",   "Disaggregated requests"),
]


def load_csv(path):
    """Group rows by (arm, rate) and average numeric columns across seeds.
    Returns {arm: {rate: {col: mean}}} with rate as float for sorting."""
    groups = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            groups[(row["arm"], float(row["rate"]))].append(row)

    out = defaultdict(dict)
    for (arm, rate), rows in groups.items():
        agg = {}
        for col in rows[0]:
            if col in ("arm", "rate", "seed", "sat_level"):
                continue
            try:
                agg[col] = sum(float(r[col]) for r in rows) / len(rows)
            except ValueError:
                continue
        out[arm][rate] = agg
    return out


def make_legend(ax):
    ax.axis("off")
    handles = []
    for _label, color, desc in ARMS:
        handles.append(mlines.Line2D([], [], color=color, marker="o",
                                     markersize=5, linestyle="-", label=desc))
    handles += [
        mlines.Line2D([], [], color="gray", linestyle="-",  label="mean"),
        mlines.Line2D([], [], color="gray", linestyle="-.", label="p90"),
        mlines.Line2D([], [], color="gray", linestyle="--", label="p95"),
        mlines.Line2D([], [], color="gray", linestyle=":",  label="p99"),
    ]
    ax.legend(handles=handles, loc="center", fontsize=9, frameon=False)


for name in WORKLOADS:
    csv_path = f"{DATA_DIR}/{name}.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping {name}: {csv_path} not found")
        continue

    data = load_csv(csv_path)
    rates = sorted({r for arm_data in data.values() for r in arm_data})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, (title, series) in zip(axes[0], METRIC_GROUPS):
        for arm, color, _desc in ARMS:
            if arm not in data:
                continue
            arm_data = data[arm]
            for col, _label, ls in series:
                y = [arm_data[r][col] for r in rates if r in arm_data and col in arm_data[r]]
                xr = [r for r in rates if r in arm_data and col in arm_data[r]]
                ax.plot(xr, y, color=color, linestyle=ls, marker="o", markersize=4)
        ax.set_title(title)
        ax.set_xlabel("aggregate_rate")
        ax.grid(True, alpha=0.3)

    for ax, (col, title) in zip(axes[1], SCALAR_METRICS):
        for arm, color, _desc in ARMS:
            if arm not in data:
                continue
            arm_data = data[arm]
            xr = [r for r in rates if r in arm_data and col in arm_data[r]]
            y = [arm_data[r][col] for r in xr]
            ax.plot(xr, y, color=color, linestyle="-", marker="o", markersize=4)
        ax.set_title(title)
        ax.set_xlabel("aggregate_rate")
        ax.grid(True, alpha=0.3)

    make_legend(axes[1, 2])

    fig.suptitle(f"PD rate sweep: agg vs threshold=16 vs edpp — {name}", fontsize=14)
    plt.tight_layout()
    out = f"{DATA_DIR}/{name}-rate-sweep.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
