#!/usr/bin/env bash
# Sweep aggregate_rate for the inference-perf workloads, comparing PD and AGG
# cluster configurations on meta-llama/llama-3.3-70b-instruct with H100 GPUs.
#
# Configs use 12 H100 GPUs (same total GPU count for fair comparison).
# Rationale: campaign iter-3 confirmed PD-1P+2D-TP=4 vs AGG-3xTP=4 at 12 GPUs
# is the cleanest result — 98.1–99.6% TTFT reduction, zero E2E regression on all
# 3 seeds, 3000/3000 completions. The iter-3 rate sweep (30→100) was never run;
# this script completes it. 12 GPUs also aligns with the campaign's stated
# preference for fewer GPUs to ease real-hardware demonstration.
#
# Code-generation note: campaign iter-2 excluded code-gen at 10.3 GB/s because
# the shared 30k-token prefix warms the decode pod's KV cache after the first
# session, causing nearly all requests to fall below the prefix-threshold and
# leaving the prefill pod idle. The same mechanism applies to the multi-turn
# workload here. Code-gen runs are included to document this negative case.
#
# Rate range selection:
#   interactive-chat:  campaign ran rate=50 (saturated, 98%+ TTFT reduction).
#                      Sweep 10–100 to locate the load crossover below rate=50.
#   code-generation:   rate=0.1 is OVERLOADED on AGG-3xTP=4 (no PD benefit
#                      expected at 10.3 GB/s; see note above).
#
# Output: $OUTDIR (default ./blis-pd-rate-sweep)
#   <workload>-<config>-<rate>.json    # full metrics JSON
#   <workload>-<config>-<rate>.log     # blis stdout
#   <workload>.csv                     # one row per (config, rate)
#
# Usage:
#   examples/sweep-pd-rate.sh
#   OUTDIR=./results examples/sweep-pd-rate.sh
#   CHAT_RATES="20 50 100" CODE_RATES="0.02 0.05 0.1" examples/sweep-pd-rate.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-./blis-pd-rate-sweep}"
# Chat: 10→100 covers unsaturated through deep saturation at 12 GPUs.
# Code-gen: 0.01→0.1 covers the full range (already overloaded at 0.1).
CHAT_RATES="${CHAT_RATES:-10 20 30 50 80 100 150 200}"
CODE_RATES="${CODE_RATES:-0.01 0.025 0.05 0.075 0.1 0.25 0.5 1 5 10 20}"
MODEL="meta-llama/llama-3.3-70b-instruct"

mkdir -p "$OUTDIR"
[[ -x ./blis ]] || go build -o blis main.go

# Cluster configs: "label|blis-flags"
# All use 12 H100 GPUs (same total GPU count for fair comparison).
#   pd-1p2d:   PD disaggregation — 1 prefill + 2 decode, TP=4 each (12 GPUs)
#              threshold=16 disaggregates all chat requests (5k prefix >> 16);
#              this is the config confirmed in campaign iter-3.
#   agg-3xtp4: 3 co-located instances, TP=4 (12 GPUs) — the only fair AGG
#              baseline at 12 GPUs for a 70B model.
CONFIGS=(
  "pd-1p2d|--num-instances 3 --prefill-instances 1 --decode-instances 2 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider prefix-threshold --pd-prefix-threshold 16"
  "agg-3xtp4|--num-instances 3 --tp 4 --hardware H100"
)

CSV_HEADER="config,rate,throughput_rps,tokens_per_sec,completed,ttft_mean_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,e2e_mean_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,itl_mean_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,timeouts,preemptions,disagg_count,sat_level,sat_score"

run_sweep() {
  local wl_name=$1 wl_path=$2
  shift 2
  local rates=("$@")
  local csv="$OUTDIR/${wl_name}.csv"

  echo "$CSV_HEADER" > "$csv"
  echo "=== $wl_name ==="

  for spec in "${CONFIGS[@]}"; do
    local cfg_label="${spec%%|*}"
    local cfg_flags="${spec#*|}"

    echo "  config: $cfg_label"
    for rate in "${rates[@]}"; do
      local cfg_yaml="$OUTDIR/${wl_name}-${cfg_label}-${rate}.yaml"
      local metrics="$OUTDIR/${wl_name}-${cfg_label}-${rate}.json"
      local log="$OUTDIR/${wl_name}-${cfg_label}-${rate}.log"

      sed "s/^aggregate_rate: .*/aggregate_rate: $rate/" "$wl_path" > "$cfg_yaml"
      printf "    rate=%-8s ... " "$rate"

      # shellcheck disable=SC2086
      ./blis run \
        --model "$MODEL" \
        --workload-spec "$cfg_yaml" \
        $cfg_flags \
        --post-hoc-detector composite \
        --metrics-path "$metrics" > "$log" 2>&1

      local disagg
      disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)

      jq -r --arg cfg "$cfg_label" --arg r "$rate" --argjson d "$disagg" '
        [$cfg, $r,
         (.responses_per_sec // 0),
         (.tokens_per_sec // 0),
         (.completed_requests // 0),
         (.ttft_mean_ms // 0), (.ttft_p90_ms // 0), (.ttft_p95_ms // 0), (.ttft_p99_ms // 0),
         (.e2e_mean_ms // 0),  (.e2e_p90_ms // 0),  (.e2e_p95_ms // 0),  (.e2e_p99_ms // 0),
         (.itl_mean_ms // 0),  (.itl_p90_ms // 0),  (.itl_p95_ms // 0),  (.itl_p99_ms // 0),
         (.timed_out_requests // 0),
         (.preemption_count // 0),
         $d,
         (.saturation.level // ""),
         (.saturation.score // 0)
        ] | @csv
      ' "$metrics" >> "$csv"

      local ttft e2e sat
      ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
      e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
      sat=$(jq -r  '.saturation.level // "?"' "$metrics")
      printf "ttft_p99=%8.1fms  e2e_p99=%9.1fms  sat=%-10s  disagg=%s\n" \
        "$ttft" "$e2e" "$sat" "$disagg"
    done
    echo ""
  done

  # Summary: all configs × rates, grouped by rate for easy crossover inspection
  echo "  Summary (sorted by rate, then TTFT p99):"
  printf "  %-12s %-8s %10s %10s %10s %10s %8s %10s\n" \
    "config" "rate" "ttft_p99" "e2e_p99" "itl_p99" "rps" "disagg" "sat"
  tail -n +2 "$csv" | sort -t, -k2 -g -k9 -g | awk -F, '
    function strip(s) { gsub(/"/, "", s); return s }
    {
      printf "  %-12s %-8s %10.1f %10.1f %10.1f %10.3f %8d %10s\n",
        strip($1), strip($2), $9, $13, $17, $3, $21, strip($22)
    }
  '
  echo ""
  echo "  All metrics: $csv"
  echo ""
}

read -ra chat_rates <<< "$CHAT_RATES"
read -ra code_rates <<< "$CODE_RATES"

run_sweep "interactive-chat" "examples/inference-perf-interactive-chat.yaml" "${chat_rates[@]}"
run_sweep "code-generation"  "examples/inference-perf-code-generation.yaml"  "${code_rates[@]}"

echo "All results in: $OUTDIR"
