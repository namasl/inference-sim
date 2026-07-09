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
# Cells (workload × arm × rate × seed) are independent and run in parallel up
# to JOBS at a time. Each cell writes its own .csvrow file; the per-workload
# CSV is assembled in deterministic order at the end so output is stable
# regardless of completion order.
#
# Binaries:
#   blis      — standard BLIS build. Used for agg and all *-thresh16 arms.
#   blis-edpp — built externally. Required only when the edpp arms are enabled.
#
# Output: $OUTDIR (default ./exp/blis-pd-sweep-all-workloads, i.e. a directory
# next to this script)
#   <workload>-<arm>-<rate>-<seed>.json    # full metrics JSON per cell
#   <workload>-<arm>-<rate>-<seed>.log     # blis stdout (includes PD metrics)
#   <workload>-<arm>-<rate>-<seed>.csvrow  # per-cell CSV row (intermediate)
#   <workload>.csv                         # one row per (arm, rate, seed)
#
# Usage:
#   exp/sweep-pd-rate_all-workloads.sh
#   JOBS=16 exp/sweep-pd-rate_all-workloads.sh
#   OUTDIR=./results JOBS=8 exp/sweep-pd-rate_all-workloads.sh
#   CHAT_RATES="50 100 150" SEEDS="42" exp/sweep-pd-rate_all-workloads.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# OUTDIR lives under exp/ (next to this script), not under the repo root.
OUTDIR="${OUTDIR:-./exp/blis-pd-sweep-all-workloads}"

# Maximum number of concurrent blis runs. Defaults to nproc.
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

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

# Workload definitions. Order is preserved in the final CSV/summary output.
WORKLOAD_ORDER=(
  "interactive-chat"
  "code-generation"
  "deep-research"
  "reasoning"
  "batch-summarization-rag"
  "batch-synthetic-data-generation"
)
declare -A WL_PATH=(
  ["interactive-chat"]="workloads/inference-perf-interactive-chat.yaml"
  ["code-generation"]="workloads/inference-perf-code-generation.yaml"
  ["deep-research"]="workloads/inference-perf-deep-research.yaml"
  ["reasoning"]="workloads/inference-perf-reasoning.yaml"
  ["batch-summarization-rag"]="workloads/inference-perf-batch-summarization-rag.yaml"
  ["batch-synthetic-data-generation"]="workloads/inference-perf-batch-synthetic-data-generation.yaml"
)
declare -A WL_RATES=(
  ["interactive-chat"]="$CHAT_RATES"
  ["code-generation"]="$CODE_RATES"
  ["deep-research"]="$DEEPRESEARCH_RATES"
  ["reasoning"]="$REASONING_RATES"
  ["batch-summarization-rag"]="$BATCHSUMMARIZATION_RATES"
  ["batch-synthetic-data-generation"]="$BATCHSYNTHETIC_RATES"
)

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

# Extract the per-cell CSV row from a metrics JSON and write it to $row.
# Shared by run_cell's fresh-run and skip paths so the two stay in lockstep.
emit_row() {
  local metrics=$1 disagg=$2 arm=$3 rate=$4 seed=$5 row=$6
  jq -r --arg arm "$arm" --arg r "$rate" --arg s "$seed" --argjson d "$disagg" '
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
  ' "$metrics" > "$row"
}

# Run a single (workload, arm, rate, seed) cell and write its .csvrow.
# All inputs come via positional args so this is safe to invoke from a
# background subshell.
#
# Idempotency: a cell whose metrics JSON already exists and parses is skipped
# (its .csvrow is regenerated so Phase 3 assembly still finds it). A cell whose
# blis run fails writes a .fail marker instead of aborting the pool — the marker
# is collected into the post-run failure banner. run_cell always returns 0 so
# the dispatch loop's `wait -n` never trips `set -e`.
run_cell() {
  local wl_name=$1 arm=$2 bin=$3 flags=$4 rate=$5 seed=$6
  local cfg_yaml="$OUTDIR/${wl_name}-${rate}.yaml"
  local metrics="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.json"
  local log="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.log"
  local row="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.csvrow"
  local fail="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.fail"
  local disagg ttft e2e sat

  # Skip if a valid result already exists (idempotency). Still regenerate the
  # .csvrow so Phase 3 assembly finds it, and clear any stale .fail marker.
  if [[ -f "$metrics" ]] && jq -e . "$metrics" > /dev/null 2>&1; then
    rm -f "$fail"
    disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" 2>/dev/null | grep -oE '[0-9]+$' || echo 0)
    emit_row "$metrics" "$disagg" "$arm" "$rate" "$seed" "$row"
    ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
    e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
    sat=$(jq -r  '.saturation.level // "?"' "$metrics")
    printf "[%-26s] %-20s rate=%-8s seed=%-4s SKIP  ttft_p99=%8.1fms e2e_p99=%9.1fms sat=%-10s disagg=%s\n" \
      "$wl_name" "$arm" "$rate" "$seed" "$ttft" "$e2e" "$sat" "$disagg" >&2
    return 0
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
    # Record the failure for the post-run banner; drop any stale row so Phase 3
    # doesn't assemble a partial result for this cell.
    rm -f "$row"
    printf '%s  arm=%s  rate=%s  seed=%s  (exit %d)  log=%s\n' \
      "$wl_name" "$arm" "$rate" "$seed" "$rc" "$log" > "$fail"
    printf "[%-26s] %-20s rate=%-8s seed=%-4s FAILED (exit %d) — see %s\n" \
      "$wl_name" "$arm" "$rate" "$seed" "$rc" "$log" >&2
    return 0
  fi
  rm -f "$fail"

  # PD metrics live in stdout, not the JSON. Aggregate arm has no PD metrics;
  # default disagg=0.
  disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)
  emit_row "$metrics" "$disagg" "$arm" "$rate" "$seed" "$row"

  ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
  e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
  sat=$(jq -r  '.saturation.level // "?"' "$metrics")
  # Per-cell progress line. Goes to stderr so any future stdout pipelining
  # of this script is unaffected. Lines may interleave across workers — that's
  # fine; the assembled CSV is deterministic.
  printf "[%-26s] %-20s rate=%-8s seed=%-4s ttft_p99=%8.1fms e2e_p99=%9.1fms sat=%-10s disagg=%s\n" \
    "$wl_name" "$arm" "$rate" "$seed" "$ttft" "$e2e" "$sat" "$disagg" >&2
}

# Phase 1: generate per-(workload, rate) cfg yamls upfront. Doing this here
# (sequentially, before any worker starts) avoids races when multiple cells
# share the same cfg file.
echo "Generating cfg yamls..."
for wl_name in "${WORKLOAD_ORDER[@]}"; do
  wl_path="${WL_PATH[$wl_name]}"
  read -ra rates <<< "${WL_RATES[$wl_name]}"
  for rate in "${rates[@]}"; do
    cfg_yaml="$OUTDIR/${wl_name}-${rate}.yaml"
    sed "s/^aggregate_rate: .*/aggregate_rate: $rate/" "$wl_path" > "$cfg_yaml"
  done
done

# Phase 2: dispatch all cells with a JOBS-deep job pool. wait -n returns when
# any single background job completes.
total_cells=0
for wl_name in "${WORKLOAD_ORDER[@]}"; do
  read -ra rates <<< "${WL_RATES[$wl_name]}"
  read -ra seeds <<< "$SEEDS"
  total_cells=$(( total_cells + ${#ARMS[@]} * ${#rates[@]} * ${#seeds[@]} ))
done
echo "Dispatching $total_cells cells across $JOBS workers..."

# Clear stale failure markers from a prior run so the post-run banner reflects
# only this invocation. (run_cell also clears its own marker on success/skip.)
rm -f "$OUTDIR"/*.fail 2>/dev/null || true

running=0
dispatched=0
for spec in "${ARMS[@]}"; do
  arm="${spec%%|*}"
  rest="${spec#*|}"
  bin="${rest%%|*}"
  flags="${rest#*|}"
  for wl_name in "${WORKLOAD_ORDER[@]}"; do
    read -ra rates <<< "${WL_RATES[$wl_name]}"
    read -ra seeds <<< "$SEEDS"
    for rate in "${rates[@]}"; do
      for seed in "${seeds[@]}"; do
        while (( running >= JOBS )); do
          wait -n
          running=$(( running - 1 ))
        done
        run_cell "$wl_name" "$arm" "$bin" "$flags" "$rate" "$seed" &
        running=$(( running + 1 ))
        dispatched=$(( dispatched + 1 ))
      done
    done
  done
done
wait
echo "All $dispatched cells complete."
echo ""

# Phase 3: assemble per-workload CSVs in deterministic order (arm × rate × seed)
# and print the summary table.
for wl_name in "${WORKLOAD_ORDER[@]}"; do
  csv="$OUTDIR/${wl_name}.csv"
  echo "$CSV_HEADER" > "$csv"
  echo "=== $wl_name ==="

  read -ra rates <<< "${WL_RATES[$wl_name]}"
  read -ra seeds <<< "$SEEDS"

  for spec in "${ARMS[@]}"; do
    arm="${spec%%|*}"
    for rate in "${rates[@]}"; do
      for seed in "${seeds[@]}"; do
        row="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.csvrow"
        if [[ -s "$row" ]]; then
          cat "$row" >> "$csv"
        else
          echo "  WARN: missing or empty $row" >&2
        fi
      done
    done
  done

  # Per-workload failure banner — collected from this workload's .fail markers.
  shopt -s nullglob
  wl_fails=("$OUTDIR/${wl_name}"-*.fail)
  shopt -u nullglob
  if (( ${#wl_fails[@]} > 0 )); then
    printf "\n\033[1;31m%s\033[0m\n" "$(printf '=%.0s' {1..80})"
    printf "\033[1;31m  FAILED CELLS in workload '%s' — %d cell(s) failed (re-run to retry)\033[0m\n" \
      "$wl_name" "${#wl_fails[@]}"
    printf "\033[1;31m%s\033[0m\n" "$(printf '=%.0s' {1..80})"
    for f in "${wl_fails[@]}"; do
      printf "\033[1;31m  x  %s\033[0m\n" "$(cat "$f")"
    done
    printf "\033[1;31m%s\033[0m\n\n" "$(printf '=%.0s' {1..80})"
  fi

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
done

echo "All results in: $OUTDIR"

# Cross-workload failure summary — printed last so it's impossible to miss.
# Each failed cell left a .fail marker (workers can't append to a shared array
# from background subshells, so we collect from the filesystem instead).
shopt -s nullglob
fail_markers=("$OUTDIR"/*.fail)
shopt -u nullglob
if (( ${#fail_markers[@]} > 0 )); then
  printf "\n"
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  printf "\033[1;31m##  SWEEP FINISHED WITH %d FAILURE(S) — re-run this script to retry%-*s##\033[0m\n" \
    "${#fail_markers[@]}" 1 ""
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  for f in "${fail_markers[@]}"; do
    printf "\033[1;31m  x  %s\033[0m\n" "$(cat "$f")"
  done
  printf "\033[1;31m%s\033[0m\n" "$(printf '#%.0s' {1..80})"
  printf "\033[1;31m%s\033[0m\n\n" "$(printf '#%.0s' {1..80})"
  exit 1
fi
