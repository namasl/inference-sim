#!/usr/bin/env bash
# Sweep aggregate_rate × seed for all six inference-perf workloads, comparing
# four 16-GPU cluster configurations on meta-llama/llama-3.3-70b-instruct,
# H100 TP=4:
#
#   agg-4xtp4         — 4 co-located instances, TP=4 (no PD disaggregation)
#   pd-1p3d-thresh16  — 1P+3D TP=4, prefix-threshold decider at threshold=16
#   pd-2p2d-thresh16  — 2P+2D TP=4, prefix-threshold decider at threshold=16
#   pd-3p1d-thresh16  — 3P+1D TP=4, prefix-threshold decider at threshold=16
#
# All four arms use 16 GPUs (4 instances × TP=4) for a GPU-matched comparison.
# The three edpp counterparts (1P+3D, 2P+2D, 3P+1D) are listed in the ARMS
# array but commented out — uncomment when blis-edpp is available and you want
# to re-enable them.
#
# For each workload, runs every (arm, rate, seed) cell and writes a per-workload
# CSV plus a stdout summary table. Rates are approximately geometric sequences
# tuned per workload to span moderately-loaded → heavily-loaded.
#
# Binaries:
#   blis      — standard BLIS build. Used for agg and all *-thresh16 arms.
#   blis-edpp — built externally. Required only when the edpp arms are enabled.
#
# Output: $OUTDIR (default ./exp/blis-pd-sweep-all-workloads, i.e. a directory
# next to this script)
#   <workload>-<arm>-<rate>-<seed>.json   # full metrics JSON per cell
#   <workload>-<arm>-<rate>-<seed>.log    # blis stdout (includes PD metrics)
#   <workload>.csv                        # one row per (arm, rate, seed)
#
# Usage:
#   exp/sweep-pd-rate_all-workloads.sh
#   OUTDIR=./results exp/sweep-pd-rate_all-workloads.sh
#   CHAT_RATES="50 100 150" SEEDS="42" exp/sweep-pd-rate_all-workloads.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# OUTDIR lives under exp/ (next to this script), not under the repo root.
OUTDIR="${OUTDIR:-./exp/blis-pd-sweep-all-workloads}"

# Per-workload rate sweeps. Approximately geometric, chosen to span
# moderately-loaded → heavily-loaded for 16 H100 GPUs (4×TP=4) on llama-3.3-70b.
# Override any of these via env var when invoking the script.
CHAT_RATES="${CHAT_RATES:-5 10 20 50 100 200 500}"
CODE_RATES="${CODE_RATES:-0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100}"
DEEPRESEARCH_RATES="${DEEPRESEARCH_RATES:-0.2 0.5 1 2 5 10 20 50 100}"
REASONING_RATES="${REASONING_RATES:-0.2 0.5 1 2 5 10 20 50 100 200}"
BATCHSUMMARIZATION_RATES="${BATCHSUMMARIZATION_RATES:-0.2 0.5 1 2 5 10 20 50 100}"
BATCHSYNTHETIC_RATES="${BATCHSYNTHETIC_RATES:-0.2 0.5 1 2 5 10 20 50 100}"

SEEDS="${SEEDS:-42 123 777}"
MODEL="meta-llama/llama-3.3-70b-instruct"

mkdir -p "$OUTDIR"

# Arms: "label|binary|flags". All use 16 GPUs (4 instances × TP=4).
# The edpp arms are commented out — uncomment to re-enable when blis-edpp is
# available.
ARMS=(
  "agg-4xtp4|./blis|--num-instances 4 --tp 4 --hardware H100"
  "pd-1p3d-thresh16|./blis|--num-instances 4 --prefill-instances 1 --decode-instances 3 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider prefix-threshold --pd-prefix-threshold 16"
  "pd-2p2d-thresh16|./blis|--num-instances 4 --prefill-instances 2 --decode-instances 2 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider prefix-threshold --pd-prefix-threshold 16"
  "pd-3p1d-thresh16|./blis|--num-instances 4 --prefill-instances 3 --decode-instances 1 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider prefix-threshold --pd-prefix-threshold 16"
  # "pd-1p3d-edpp|./exp/blis-edpp|--num-instances 4 --prefill-instances 1 --decode-instances 3 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider edpp"
  # "pd-2p2d-edpp|./exp/blis-edpp|--num-instances 4 --prefill-instances 2 --decode-instances 2 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider edpp"
  # "pd-3p1d-edpp|./exp/blis-edpp|--num-instances 4 --prefill-instances 3 --decode-instances 1 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider edpp"
)

CSV_HEADER="arm,rate,seed,throughput_rps,tokens_per_sec,completed,ttft_mean_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,e2e_mean_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,itl_mean_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,timeouts,preemptions,disagg_count,sat_level,sat_score,goodput_rps,slo_attainment"

run_sweep() {
  local wl_name=$1 wl_path=$2
  shift 2
  local rates=("$@")
  local csv="$OUTDIR/${wl_name}.csv"
  local failures=()

  echo "=== $wl_name ==="

  read -ra seeds <<< "$SEEDS"

  for spec in "${ARMS[@]}"; do
    local arm="${spec%%|*}"
    local rest="${spec#*|}"
    local bin="${rest%%|*}"
    local flags="${rest#*|}"

    echo "  arm: $arm  ($bin)"
    for rate in "${rates[@]}"; do
      local cfg_yaml="$OUTDIR/${wl_name}-${rate}.yaml"
      sed "s/^aggregate_rate: .*/aggregate_rate: $rate/" "$wl_path" > "$cfg_yaml"

      for seed in "${seeds[@]}"; do
        local metrics="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.json"
        local log="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.log"

        printf "    rate=%-8s seed=%-4s ... " "$rate" "$seed"

        # Skip if valid result already exists (idempotency).
        if [[ -f "$metrics" ]] && jq -e . "$metrics" > /dev/null 2>&1; then
          local s_ttft s_e2e s_sat s_disagg
          s_ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
          s_e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
          s_sat=$(jq -r  '.saturation.level // "?"' "$metrics")
          s_disagg=0
          if [[ -f "$log" ]]; then
            s_disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)
          fi
          printf "SKIP  ttft_p99=%8.1fms  e2e_p99=%9.1fms  sat=%-10s  disagg=%s\n" \
            "$s_ttft" "$s_e2e" "$s_sat" "$s_disagg"
          continue
        fi

        # shellcheck disable=SC2086
        local rc=0
        "$bin" run \
          --model "$MODEL" \
          --workload-spec "$cfg_yaml" \
          --seed "$seed" \
          $flags \
          --post-hoc-detector composite \
          --metrics-path "$metrics" > "$log" 2>&1 || rc=$?

        if (( rc != 0 )); then
          printf "FAILED (exit %d) — see %s\n" "$rc" "$log"
          local cell="${wl_name}  arm=${arm}  rate=${rate}  seed=${seed}  (exit ${rc})  log=${log}"
          failures+=("$cell")
          ALL_FAILURES+=("$cell")
          continue
        fi

        # PD metrics live in stdout, not the JSON. Aggregate arm has no PD
        # metrics; default disagg=0.
        local disagg ttft e2e sat
        disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)
        ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
        e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
        sat=$(jq -r  '.saturation.level // "?"' "$metrics")
        printf "ttft_p99=%8.1fms  e2e_p99=%9.1fms  sat=%-10s  disagg=%s\n" \
          "$ttft" "$e2e" "$sat" "$disagg"
      done
    done
    echo ""
  done

  # Rebuild the CSV from every valid result JSON for this workload, including
  # results from arms that are currently commented out. Arm/rate/seed are parsed
  # from the filename: strip the workload prefix, then peel the last two
  # dash-fields from the right (seed, then rate); the remainder is the arm label.
  echo "$CSV_HEADER" > "$csv"
  local jfile bname r_rest r_arm r_rate r_seed r_disagg r_log
  for jfile in "$OUTDIR/${wl_name}"-*.json; do
    [[ -f "$jfile" ]] || continue                        # handle empty glob
    jq -e . "$jfile" > /dev/null 2>&1 || continue       # skip corrupt/incomplete files
    bname=$(basename "$jfile" .json)
    r_rest="${bname#${wl_name}-}"                        # strip workload-name prefix
    r_seed="${r_rest##*-}"                               # last dash-field  → seed
    r_rest="${r_rest%-${r_seed}}"                        # drop seed
    r_rate="${r_rest##*-}"                               # last dash-field  → rate
    r_arm="${r_rest%-${r_rate}}"                         # remainder        → arm label
    [[ -n "$r_arm" && -n "$r_rate" && -n "$r_seed" ]] || continue
    r_disagg=0
    r_log="${jfile%.json}.log"
    if [[ -f "$r_log" ]]; then
      r_disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$r_log" | grep -oE '[0-9]+$' || echo 0)
    fi
    jq -r --arg arm "$r_arm" --arg r "$r_rate" --arg s "$r_seed" --argjson d "$r_disagg" '
      [$arm, $r, $s,
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
       (.saturation.score // 0),
       (.goodput_rps // ""),
       (.slo_attainment // "")
      ] | @csv
    ' "$jfile" >> "$csv"
  done

  # Per-workload failure banner.
  if (( ${#failures[@]} > 0 )); then
    printf "\n\033[1;31m%s\033[0m\n" "$(printf '=%.0s' {1..80})"
    printf "\033[1;31m  FAILED CELLS in workload '%s' — %d cell(s) failed (re-run to retry)\033[0m\n" \
      "$wl_name" "${#failures[@]}"
    printf "\033[1;31m%s\033[0m\n" "$(printf '=%.0s' {1..80})"
    for f in "${failures[@]}"; do
      printf "\033[1;31m  x  %s\033[0m\n" "$f"
    done
    printf "\033[1;31m%s\033[0m\n\n" "$(printf '=%.0s' {1..80})"
  fi

  # Summary: seed-mean per (arm, rate), sorted by rate then TTFT p99.
  # Reads the CSV and averages across seeds per (arm, rate) cell.
  echo "  Summary (seed-mean per arm × rate, sorted by rate then TTFT p99):"
  printf "  %-20s %-8s %10s %10s %10s %10s %8s %12s\n" \
    "arm" "rate" "ttft_p99" "e2e_p99" "itl_p99" "rps" "disagg" "sat(mode)"
  tail -n +2 "$csv" | awk -F, '
    function strip(s) { gsub(/"/, "", s); return s }
    {
      arm=strip($1); r=strip($2); k=arm","r
      n[k]++
      ttft[k]+=$10; e2e[k]+=$14; itl[k]+=$18; rps[k]+=$4; dis[k]+=$21
      satk=k","strip($22); satc[satk]++
    }
    END {
      for (k in n) {
        # Find modal sat level for this cell.
        best=""; bestc=0
        for (sk in satc) {
          if (index(sk, k",") == 1) {
            s=substr(sk, length(k)+2)
            if (satc[sk] > bestc) { bestc=satc[sk]; best=s }
          }
        }
        split(k, kk, ",")
        printf "%s\t%s\t%.1f\t%.1f\t%.1f\t%.3f\t%d\t%s\n",
          kk[1], kk[2], ttft[k]/n[k], e2e[k]/n[k], itl[k]/n[k],
          rps[k]/n[k], int(dis[k]/n[k]), best
      }
    }
  ' | sort -t$'\t' -k2 -g -k3 -g | awk -F'\t' '
    {
      printf "  %-20s %-8s %10.1f %10.1f %10.1f %10.3f %8d %12s\n",
        $1, $2, $3, $4, $5, $6, $7, $8
    }
  '
  echo ""
  echo "  All metrics: $csv"
  echo ""
}

read -ra chat_rates             <<< "$CHAT_RATES"
read -ra code_rates             <<< "$CODE_RATES"
read -ra deepresearch_rates     <<< "$DEEPRESEARCH_RATES"
read -ra reasoning_rates        <<< "$REASONING_RATES"
read -ra batchsummarization_rates <<< "$BATCHSUMMARIZATION_RATES"
read -ra batchsynthetic_rates   <<< "$BATCHSYNTHETIC_RATES"

ALL_FAILURES=()

run_sweep "interactive-chat"               "workloads/inference-perf-interactive-chat.yaml"               "${chat_rates[@]}"
run_sweep "code-generation"                "workloads/inference-perf-code-generation.yaml"                "${code_rates[@]}"
run_sweep "deep-research"                  "workloads/inference-perf-deep-research.yaml"                  "${deepresearch_rates[@]}"
run_sweep "reasoning"                      "workloads/inference-perf-reasoning.yaml"                      "${reasoning_rates[@]}"
run_sweep "batch-summarization-rag"        "workloads/inference-perf-batch-summarization-rag.yaml"        "${batchsummarization_rates[@]}"
run_sweep "batch-synthetic-data-generation" "workloads/inference-perf-batch-synthetic-data-generation.yaml" "${batchsynthetic_rates[@]}"

echo "All results in: $OUTDIR"

# Cross-workload failure summary — printed last so it's impossible to miss.
if (( ${#ALL_FAILURES[@]} > 0 )); then
  printf "\n"
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  printf "\033[1;31m##  SWEEP FINISHED WITH %d FAILURE(S) — re-run this script to retry%-*s##\033[0m\n" \
    "${#ALL_FAILURES[@]}" 1 ""
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  for f in "${ALL_FAILURES[@]}"; do
    printf "\033[1;31m  x  %s\033[0m\n" "$f"
  done
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  printf "\033[1;31m%s\033[0m\n\n" "$(printf '#%.0s' {1..80})"
  exit 1
fi
