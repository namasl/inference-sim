#!/usr/bin/env python3
"""Plot rate-sweep results from blis-pd-sweep-all-workloads/.

One figure per workload. Four arms (agg-4xtp4, pd-1p3d-thresh16, pd-2p2d-thresh16,
pd-3p1d-thresh16) compared across rates. Color encodes arm; linestyle encodes
percentile. Seeds are collapsed to per-(arm, rate) mean.

The three edpp counterparts are listed in ARMS but commented out — uncomment
them (and re-enable in the sweep script) to plot edpp alongside threshold=16.
"""

import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import yaml

DATA_DIR = "./blis-pd-sweep-all-workloads"
WORKLOAD_DIR = "../workloads"
WORKLOADS = [
    "interactive-chat",
    "code-generation",
    "deep-research",
    "reasoning",
    "batch-summarization-rag",
    "batch-synthetic-data-generation",
]

# (arm label, color, legend description)
# Colors: aggregate is gray (neutral baseline). The three threshold=16 PD splits
# use a cool-to-warm progression as the prefill share grows (1P→3P).
ARMS = [
    ("agg-4xtp4",        "tab:gray",   "agg-4xtp4 (4×TP=4, no PD)"),
    ("pd-1p3d-thresh16", "tab:blue",   "pd-1p3d-thresh16 (1P+3D, prefix-threshold=16)"),
    ("pd-2p2d-thresh16", "tab:green",  "pd-2p2d-thresh16 (2P+2D, prefix-threshold=16)"),
    ("pd-3p1d-thresh16", "tab:orange", "pd-3p1d-thresh16 (3P+1D, prefix-threshold=16)"),
    # ("pd-1p3d-edpp",     "tab:red",    "pd-1p3d-edpp (1P+3D, edpp decider)"),
    # ("pd-2p2d-edpp",     "tab:purple", "pd-2p2d-edpp (2P+2D, edpp decider)"),
    # ("pd-3p1d-edpp",     "tab:brown",  "pd-3p1d-edpp (3P+1D, edpp decider)"),
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

def load_workload_slos(name):
    """Read goodput_slo_targets from the matching workloads/inference-perf-*.yaml.

    Returns a flat list of (class, metric, threshold_ms) tuples — one per
    (class, dimension) pair in the spec. BLIS evaluates these per-request and
    emits goodput_rps/slo_attainment in the metrics JSON; this function exists
    only to render the SLO description panel and the threshold reference lines.
    """
    path = f"{WORKLOAD_DIR}/inference-perf-{name}.yaml"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    targets = spec.get("goodput_slo_targets") or {}
    out = []
    for cls, dims in targets.items():
        for dim_key, value in (dims or {}).items():
            metric = dim_key.removesuffix("_ms")
            out.append((cls, metric, value))
    return out


def slo_description(slos):
    """Human-readable summary of the workload's SLOs for the description panel."""
    if not slos:
        return "No SLOs defined."
    lines = ["Goodput SLOs (BLIS-native, per-request):"]
    for cls, metric, threshold in slos:
        if threshold >= 1000:
            unit = f"{threshold / 1000:g} s"
        else:
            unit = f"{threshold} ms"
        lines.append(f"  [{cls}] {metric.upper():<5s} ≤ {unit}")
    return "\n".join(lines)


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
    ax.legend(handles=handles, loc="center", fontsize=8, frameon=False)


def make_slo_panel(ax, slos):
    ax.axis("off")
    ax.text(0.0, 1.0, slo_description(slos), family="monospace", fontsize=9,
            verticalalignment="top", horizontalalignment="left",
            transform=ax.transAxes)


for name in WORKLOADS:
    csv_path = f"{DATA_DIR}/{name}.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping {name}: {csv_path} not found")
        continue

    data = load_csv(csv_path)
    rates = sorted({r for arm_data in data.values() for r in arm_data})
    slos = load_workload_slos(name)

    # Detect whether the CSV carries BLIS-native goodput. Old runs (made before
    # SLOs landed in the workload YAMLs) have no goodput_rps column at all;
    # newer runs may have a numeric value. Anything else → native data missing.
    has_native_goodput = any(
        isinstance(arm_data.get(r, {}).get("goodput_rps"), (int, float))
        for arm_data in data.values() for r in arm_data
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    for ax, (title, series) in zip(axes[0], METRIC_GROUPS):
        for arm, color, _desc in ARMS:
            if arm not in data:
                continue
            arm_data = data[arm]
            for col, _label, ls in series:
                y = [arm_data[r][col] for r in rates if r in arm_data and col in arm_data[r]]
                xr = [r for r in rates if r in arm_data and col in arm_data[r]]
                ax.plot(xr, y, color=color, linestyle=ls, marker="o", markersize=4)
        # Overlay SLO thresholds for the matching metric (TTFT/E2E/ITL). BLIS
        # evaluates SLOs per-request, but a horizontal reference line still
        # helps eyeball where the latency curves cross the gate.
        metric_key = title.split()[0].lower()  # "TTFT (ms)" → "ttft"
        for cls, slo_metric, threshold in slos:
            if slo_metric == metric_key:
                ax.axhline(threshold, color="red", linestyle=(0, (1, 1)),
                           linewidth=1, alpha=0.7)
                ax.text(0.98, threshold, f"SLO[{cls}]", color="red",
                        fontsize=7, ha="right", va="bottom",
                        transform=ax.get_yaxis_transform())
        ax.set_title(title)
        ax.set_xlabel("aggregate_rate")
        ax.set_xscale("log")
        if title.startswith("TTFT"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

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
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")

    make_legend(axes[1, 2])

    # Bottom row: native goodput (rps), SLO attainment (fraction), SLO panel.
    # Both come from BLIS — goodput_rps and slo_attainment are populated by
    # cmd/goodput.go when the workload YAML carries goodput_slo_targets.
    ax_goodput = axes[2, 0]
    for arm, color, _desc in ARMS:
        if arm not in data:
            continue
        arm_data = data[arm]
        xr = [r for r in rates if r in arm_data and isinstance(arm_data[r].get("goodput_rps"), (int, float))]
        y = [arm_data[r]["goodput_rps"] for r in xr]
        ax_goodput.plot(xr, y, color=color, linestyle="-", marker="o", markersize=4)
    ax_goodput.set_title("Goodput (rps) — BLIS native")
    ax_goodput.set_xlabel("aggregate_rate")
    ax_goodput.set_xscale("log")
    ax_goodput.grid(True, alpha=0.3, which="both")
    if not has_native_goodput:
        ax_goodput.text(0.5, 0.5,
                        "No native goodput in CSV.\nRe-run sweep with SLO-bearing\nworkload specs.",
                        transform=ax_goodput.transAxes, ha="center", va="center",
                        color="gray", fontsize=10)

    ax_attain = axes[2, 1]
    for arm, color, _desc in ARMS:
        if arm not in data:
            continue
        arm_data = data[arm]
        xr = [r for r in rates if r in arm_data and isinstance(arm_data[r].get("slo_attainment"), (int, float))]
        y = [arm_data[r]["slo_attainment"] for r in xr]
        ax_attain.plot(xr, y, color=color, linestyle="-", marker="o", markersize=4)
    ax_attain.set_title("SLO attainment (fraction of requests meeting SLO)")
    ax_attain.set_xlabel("aggregate_rate")
    ax_attain.set_xscale("log")
    ax_attain.set_ylim(0.0, 1.05)
    ax_attain.grid(True, alpha=0.3, which="both")
    if not has_native_goodput:
        ax_attain.text(0.5, 0.5,
                       "No native attainment in CSV.\nRe-run sweep with SLO-bearing\nworkload specs.",
                       transform=ax_attain.transAxes, ha="center", va="center",
                       color="gray", fontsize=10)

    make_slo_panel(axes[2, 2], slos)

    fig.suptitle(f"PD rate sweep: agg vs 1P+3D vs 2P+2D vs 3P+1D (threshold=16) — {name}", fontsize=14)
    plt.tight_layout()
    out = f"{DATA_DIR}/{name}-rate-sweep.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
