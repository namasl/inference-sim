# PD Disaggregation Explorer

Multi-configuration comparison tool for aggregate vs prefill-decode disaggregated LLM serving using BLIS. Sweep arrival rates, prompt/output lengths, or prefill/decode splits to find the configuration that best fits your serving requirements.

## Prerequisites

- **Go** (for building BLIS)
- **Python 3.9+**

## Setup

1. Build BLIS from the repo root:

   ```bash
   cd /path/to/inference-sim
   go build -o blis main.go
   ```

2. Install Python dependencies:

   ```bash
   pip install -r demo/pd-disaggregation/requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run demo/pd-disaggregation/app.py
   ```

If your BLIS binary is in a non-standard location, set the `BLIS_BINARY` environment variable:

```bash
BLIS_BINARY=/path/to/blis streamlit run demo/pd-disaggregation/app.py
```

## Module Structure

The app is organized into five Python modules:

- `app.py` — tab orchestration and main entry point
- `blis_runner.py` — BLIS binary discovery, subprocess execution, output parsing
- `config_ui.py` — `SimConfig` dataclass and sidebar rendering
- `charts.py` — visualization functions
- `sweep.py` — BLIS subprocess execution and sweep logic

## Exploration Tabs

### Rate Sweep

Compare 2-4 named configurations as arrival rate increases. Shows the throughput-per-GPU vs throughput-per-user tradeoff curve with a Pareto front (gold stars marking points no other config dominates on both axes). Also shows latency comparison (TTFT P95, E2E P95) and KV transfer duration for PD configurations.

### Workload Sweep

Sweep token distributions at a fixed arrival rate. Two modes:

- **1D Sweep**: Sweep input or output token mean at a fixed arrival rate. Reveals the crossover point where PD disaggregation becomes advantageous as prompt or output length grows.
- **2D Heatmap**: Compare exactly 2 configs across a grid of input × output token combinations. Select input/output token ranges and number of grid points, then choose a metric (Throughput/GPU, TTFT P95, or E2E P95). The output is a heatmap where color shows the ratio of config A / config B — red means A wins, blue means B wins, and a dashed crossover line marks where ratio = 1.0. Estimated run count is shown before execution.

### Parallelism Explorer

Replaces the former Topology Explorer. Three sub-modes for finding the optimal GPU allocation:

- **P/D Split Sweep**: Given a fixed GPU budget, TP values per pool, and PD decider — enumerates all valid (P, D) instance pairs and shows throughput/GPU and TTFT P95 as bar/line charts with an aggregate baseline as a dashed reference line.
- **TP Sweep**: Fix P and D instance counts, then sweep TP values for both pools, prefill only, or decode only. Shows throughput/GPU and TTFT P95 vs TP for PD (solid) vs aggregate (dashed).
- **Replica Sweep**: Fix the P:D instance ratio and TP values, then sweep total instance count. Shows how PD and aggregate scale as more replicas are added.

## Multi-Configuration Comparison

Define 2-4 named configurations in the sidebar. Three preset starting points are provided:

- **N x Aggregate** — all instances handle both prefill and decode
- **Half/Half Always-PD** — equal prefill and decode pools with always-disaggregate decider
- **25%/75% Direct-Decode** — smaller prefill pool with direct-to-decode bypass for short prompts

Use Add/Remove to adjust the set (up to 4 configurations).

**Aggregate configs** specify: Name, Mode, Instances, TP degree, and a GPU budget line showing `N × TP = M GPUs`.

**Disaggregated configs** specify: Name, Mode, four primary controls (P Instances, P TP, D Instances, D TP), an auto-computed `num_instances = P + D`, and a GPU budget line showing `P×TP_p + D×TP_d = N GPUs`. PD-specific settings (decider, transfer parameters, interference factors) appear in a "PD Advanced" expander, with per-pool hardware overrides in a nested sub-expander.

Default preset configurations use a fixed 8 GPU budget to ensure fair comparison.

## PD Parameters

| Parameter | Description |
|-----------|-------------|
| PD Decider | always, prefix-threshold, direct-to-decode, never |
| Prefix Threshold | Non-cached token threshold for prefix-threshold decider |
| Direct-Decode Threshold | Input token threshold for direct-to-decode decider |
| Transfer Bandwidth (GB/s) | KV cache transfer bandwidth |
| Transfer Base Latency (ms) | Fixed overhead per transfer |
| KV Bytes/Token | KV cache size per token |
| Transfer Contention | Fair-share bandwidth when transfers overlap |
| Interference: Prefill Factor | Co-location slowdown for prefill-dominant batches |
| Interference: Decode Factor | Co-location slowdown for decode-dominant batches |
| Per-pool Hardware Overrides | TP, GPU type, latency model, routing scorers per pool |

## What the Charts Show

- **Throughput tradeoff**: X-axis = throughput per user (tokens/s/user), Y-axis = throughput per GPU (tokens/s/GPU). Effective concurrency computed via Little's Law: `rate x mean_e2e_seconds`.
- **Pareto front**: Gold stars mark points where no other configuration dominates on both axes simultaneously.
- **Latency**: TTFT P95 and E2E P95 plotted as solid vs dashed lines across configurations.
- **KV Transfer Duration**: Mean transfer time for PD configurations across the sweep.
- **2D Heatmap**: Ratio of metric_A / metric_B. Values > 1 (red) mean config A produces more throughput or higher latency than config B. Values < 1 (blue) mean config B wins. A dashed black line marks the crossover at ratio = 1.0.
- **Parallelism Explorer bar chart**: Each P/D split gets a distinct color; a gold bar marks the best split. A dashed gray line shows the aggregate baseline at the same GPU budget.
- **Results table**: Full numeric results downloadable as CSV.
