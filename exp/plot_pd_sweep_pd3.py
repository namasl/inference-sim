#!/usr/bin/env python3
"""Plot pd-prefix-threshold sweep results from blis-pd-sweep-thresh_pd3/.

One figure per workload. Color distinguishes threshold sweep (blue) from
reference deciders (pd3=red).
Linestyle encodes percentile.
"""

import csv
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DATA_DIR = "./blis-pd-sweep-thresh_pd3"
WORKLOADS = ["interactive-chat", "code-generation"]

SWEEP_COLOR = "tab:blue"

# (label, color, legend description). Files: <workload>-<label>.{json,log}
REFERENCES = [
    ("pd3", "tab:red", "pd3 (3-clause OR: qdGate=1, kvGate=0.04, ifrGate=19, longPrefillThreshold=10000)"),
]

# X-axis order: never disagg → always disagg
THRESHOLD_ORDER = ["never", "16384", "4096", "1024", "256", "64", "16", "0", "always"]

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
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["threshold"]] = {k: float(v) for k, v in row.items()
                                      if k not in ("threshold", "sat_level")}
    return rows


def load_reference(name, label):
    """Load a reference run's metrics. disagg_count comes from the .log file
    (PD metrics block in stdout); throughput_rps is aliased from
    responses_per_sec to match the CSV column name used by the sweep."""
    json_path = f"{DATA_DIR}/{name}-{label}.json"
    log_path  = f"{DATA_DIR}/{name}-{label}.log"
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    data["throughput_rps"] = data.get("responses_per_sec")
    if os.path.exists(log_path):
        with open(log_path) as f:
            m = re.search(r"Disaggregated Requests:\s+(\d+)", f.read())
            if m:
                data["disagg_count"] = int(m.group(1))
    return data


def draw_hlines(ax, series_or_col, refs):
    """Horizontal reference lines from each reference run, per percentile."""
    series = [(series_or_col, "", "-")] if isinstance(series_or_col, str) else series_or_col
    for ref_data, color in refs:
        if ref_data is None:
            continue
        for col, _label, ls in series:
            val = ref_data.get(col)
            if val is not None:
                ax.axhline(val, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)


def make_legend(ax):
    ax.axis("off")
    handles = [
        mlines.Line2D([], [], color=SWEEP_COLOR, linestyle="-", marker="o",
                      markersize=5, label="threshold sweep"),
    ]
    for _label, color, desc in REFERENCES:
        handles.append(mlines.Line2D([], [], color=color, linestyle="-",
                                     linewidth=1.5, label=desc))
    handles += [
        mlines.Line2D([], [], color="gray", linestyle="-",  label="mean"),
        mlines.Line2D([], [], color="gray", linestyle="-.", label="p90"),
        mlines.Line2D([], [], color="gray", linestyle="--", label="p95"),
        mlines.Line2D([], [], color="gray", linestyle=":",  label="p99"),
    ]
    ax.legend(handles=handles, loc="center", fontsize=10, frameon=False)


x = list(range(len(THRESHOLD_ORDER)))

for name in WORKLOADS:
    rows = load_csv(f"{DATA_DIR}/{name}.csv")
    refs = [(load_reference(name, label), color)
            for label, color, _desc in REFERENCES]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, (title, series) in zip(axes[0], METRIC_GROUPS):
        for col, _label, ls in series:
            y = [rows[t][col] for t in THRESHOLD_ORDER]
            ax.plot(x, y, color=SWEEP_COLOR, linestyle=ls, marker="o", markersize=4)
        draw_hlines(ax, series, refs)
        ax.set_xticks(x)
        ax.set_xticklabels(THRESHOLD_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for ax, (col, title) in zip(axes[1], SCALAR_METRICS):
        y = [rows[t][col] for t in THRESHOLD_ORDER]
        ax.plot(x, y, color=SWEEP_COLOR, linestyle="-", marker="o", markersize=4)
        draw_hlines(ax, col, refs)
        ax.set_xticks(x)
        ax.set_xticklabels(THRESHOLD_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    make_legend(axes[1, 2])

    fig.suptitle(f"PD prefix-threshold sweep — {name}", fontsize=14)
    plt.tight_layout()
    out = f"{DATA_DIR}/{name}-pd-sweep.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
