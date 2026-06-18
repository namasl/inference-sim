#!/usr/bin/env python3
"""Plot TTFT-only rate-sweep results from blis-pd-sweep-rate_threshold_vs_edpp/.

One figure per workload showing TTFT across rates. Color encodes arm;
linestyle encodes percentile. Seeds collapsed to per-(arm, rate) mean.
"""

import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DATA_DIR = "./blis-pd-sweep-rate_threshold_vs_edpp"
WORKLOADS = ["interactive-chat"]

# (arm label, color, legend description) — red/blue swapped
ARMS = [
    ("pd-thresh16", "tab:red",  "Thresh16"),
    ("pd-edpp",     "tab:blue", "EDPP"),
]

TTFT_SERIES = [
    ("ttft_mean_ms", "mean", "-"),
    ("ttft_p90_ms",  "p90",  "-."),
    ("ttft_p95_ms",  "p95",  "--"),
    ("ttft_p99_ms",  "p99",  ":"),
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


def make_legend():
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
    return handles


for name in WORKLOADS:
    csv_path = f"{DATA_DIR}/{name}.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping {name}: {csv_path} not found")
        continue

    data = load_csv(csv_path)
    rates = sorted(r for arm_data in data.values() for r in arm_data if r <= 100)

    fig, ax = plt.subplots(figsize=(7, 5))

    for arm, color, _desc in ARMS:
        if arm not in data:
            continue
        arm_data = data[arm]
        for col, _label, ls in TTFT_SERIES:
            xr = [r for r in rates if r in arm_data and col in arm_data[r]]
            y = [arm_data[r][col] for r in xr]
            ax.plot(xr, y, color=color, linestyle=ls, marker="o", markersize=4)
    ax.set_ylabel("TTFT (ms)")
    ax.set_xlabel("Rate (QPS)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PD Decider: Threshold=16 vs Empirical Drift Plus Penalty\n1P+3D, Llama-3.3-70B-Instruct, {name}", fontsize=12)
    handles = make_legend()
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = f"{DATA_DIR}/{name}-rate-sweep_blog_ttft.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
