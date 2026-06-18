#!/usr/bin/env bash
# Sweep --pd-prefix-threshold for the inference-perf workloads on a 1P3D
# H100 TP=4 cluster running meta-llama/llama-3.3-70b-instruct, plus a single
# reference run with the pd3 algorithm (AdmissionAndLongPrefillAwarePrefixDecider,
# 3-clause OR + long-prefill early return) discovered by the pd3 campaign in
# .nous/pd3/report.md.
#
# For each workload (interactive-chat, code-generation), runs every threshold
# in THRESHOLDS plus the pd3 reference, and writes a per-workload CSV plus a
# stdout summary table sorted by ttft_p99 (lower = better).
#
# Binaries:
#   blis      — built from the current working directory (latest commit, or
#               with local modifications under test). Used for the threshold
#               sweep.
#   blis-pd3  — built from a temporary worktree pinned to PD3_BASE_COMMIT with
#               examples/pd3.patch applied. Used for the pd3 reference run.
#               The worktree is removed after the build.
#
# Output: $OUTDIR (default ./blis-pd-sweep-thresh_pd3)
#   <workload>.yaml             # rate-adjusted workload spec
#   <workload>-<thr>.json       # full metrics JSON (one per threshold)
#   <workload>-<thr>.log        # blis stdout (includes PD metrics)
#   <workload>-pd3.json         # full metrics JSON for the pd3 reference run
#   <workload>-pd3.log          # blis-pd3 stdout for the pd3 reference run
#   <workload>.csv              # one row per threshold
#
# Usage:
#   examples/sweep-pd-threshold_pd3.sh                  # default rates (50, 0.1)
#   CHAT_RATE=80 CODE_RATE=0.15 examples/sweep-pd-threshold_pd3.sh
#   OUTDIR=./results examples/sweep-pd-threshold_pd3.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-./blis-pd-sweep-thresh_pd3}"
CHAT_RATE="${CHAT_RATE:-50}"
CODE_RATE="${CODE_RATE:-0.1}"
MODEL="meta-llama/llama-3.3-70b-instruct"

mkdir -p "$OUTDIR"

# Standard blis: build from the current working directory. This is the binary
# used for the threshold sweep and reflects whatever state the repo is in —
# either the latest commit, or local modifications under test.
go build -o blis main.go

# pd3 binary: build from a temporary worktree pinned to the commit the patch
# was generated against, with examples/pd3.patch applied. The worktree is
# removed after the build. This isolates the pd3 algorithm from any local
# modifications to the working tree, so the threshold-sweep blis and the pd3
# reference always compare against a known baseline.
PD3_BASE_COMMIT="4e88b200794b2dfb1d9d4e4e8ca356d86edd4e2c"
PD3_PATCH="$PWD/examples/pd3.patch"
REPO_ROOT="$PWD"
PD3_WT="$(mktemp -d -t blis-pd3-XXXXXX)"
trap 'git worktree remove --force "$PD3_WT" >/dev/null 2>&1 || true; rm -rf "$PD3_WT"' EXIT
git worktree add --detach "$PD3_WT" "$PD3_BASE_COMMIT"
git -C "$PD3_WT" apply --check "$PD3_PATCH"
git -C "$PD3_WT" apply "$PD3_PATCH"
(cd "$PD3_WT" && go build -o "$REPO_ROOT/blis-pd3" main.go)
git worktree remove --force "$PD3_WT"
trap - EXIT

# Threshold sweep: log scale plus the two endpoints.
# Format: "<label>|<flags>" — label appears in the CSV; flags go to blis.
THRESHOLDS=(
  "never|--pd-decider never"
  "16384|--pd-decider prefix-threshold --pd-prefix-threshold 16384"
  "4096|--pd-decider prefix-threshold --pd-prefix-threshold 4096"
  "1024|--pd-decider prefix-threshold --pd-prefix-threshold 1024"
  "256|--pd-decider prefix-threshold --pd-prefix-threshold 256"
  "64|--pd-decider prefix-threshold --pd-prefix-threshold 64"
  "16|--pd-decider prefix-threshold --pd-prefix-threshold 16"
  "0|--pd-decider prefix-threshold --pd-prefix-threshold 0"
  "always|--pd-decider always"
)

CSV_HEADER="threshold,throughput_rps,tokens_per_sec,completed,ttft_mean_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,e2e_mean_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,itl_mean_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,timeouts,preemptions,disagg_count,sat_level,sat_score"

run_sweep() {
  local wl_name=$1 wl_path=$2 rate=$3
  local cfg="$OUTDIR/${wl_name}.yaml"
  local csv="$OUTDIR/${wl_name}.csv"

  sed "s/^aggregate_rate: .*/aggregate_rate: $rate/" "$wl_path" > "$cfg"
  echo "$CSV_HEADER" > "$csv"

  echo "=== $wl_name (aggregate_rate=$rate) ==="
  for spec in "${THRESHOLDS[@]}"; do
    local label="${spec%%|*}"
    local flags="${spec#*|}"
    local metrics="$OUTDIR/${wl_name}-${label}.json"
    local log="$OUTDIR/${wl_name}-${label}.log"

    printf "  threshold=%-7s ... " "$label"
    # shellcheck disable=SC2086
    ./blis run \
      --model "$MODEL" \
      --workload-spec "$cfg" \
      --num-instances 4 --prefill-instances 1 --decode-instances 3 \
      --prefill-tp 4 --decode-tp 4 --hardware H100 \
      --pd-transfer-bandwidth 10.3 \
      $flags \
      --post-hoc-detector composite \
      --metrics-path "$metrics" > "$log" 2>&1

    # PD metrics live in stdout, not the JSON.
    local disagg
    disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)

    # Build a CSV row from the metrics JSON. Disagg comes from stdout (PD
    # metrics block) and is coerced to a number so awk doesn't see quotes.
    jq -r --arg t "$label" --argjson d "$disagg" '
      [$t,
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

    local e2e ttft
    e2e=$(jq -r '.e2e_p99_ms // 0' "$metrics")
    ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
    printf "ttft_p99=%8.1fms  e2e_p99=%9.1fms  disagg=%s\n" "$ttft" "$e2e" "$disagg"
  done

  echo ""
  echo "Summary (sorted by ttft_p99 ascending):"
  # Sort by ttft_p99 (column 8 in the new schema).
  {
    head -1 "$csv"
    tail -n +2 "$csv" | sort -t, -k8 -g
  } | awk -F, '
    function strip(s) { gsub(/"/, "", s); return s }
    NR==1 {
      printf "  %-9s %10s %10s %10s %10s %10s %10s %8s %10s %8s %12s\n",
        "thresh", "ttft_p90", "ttft_p95", "ttft_p99", "e2e_p90", "e2e_p95", "e2e_p99", "itl_p99", "rps", "disagg", "sat"
      next
    }
    {
      printf "  %-9s %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %8.1f %10.2f %8d %12s\n",
        strip($1), $6, $7, $8, $10, $11, $12, $16, $2, $19, strip($20)
    }
  '
  echo ""

  # pd3 reference run — best algorithm per .nous/pd3/report.md.
  # AdmissionAndLongPrefillAwarePrefixDecider, simplified to 3-clause OR
  # (qdFires || kvFires || ifrFires) + long-prefill early return. admitFires
  # is left at its vacuous default (admit-threshold=2.0) per iter-60 finding
  # that it is byte-identical when kvFires+ifrFires are active on 1P+3D.
  # Parameters validated across seeds 42/123/777 on 1P+3D H100 TP=4.
  local pd3_metrics="$OUTDIR/${wl_name}-pd3.json"
  local pd3_log="$OUTDIR/${wl_name}-pd3.log"
  printf "  pd3                ... "
  ./blis-pd3 run \
    --model "$MODEL" \
    --workload-spec "$cfg" \
    --num-instances 4 --prefill-instances 1 --decode-instances 3 \
    --prefill-tp 4 --decode-tp 4 --hardware H100 \
    --pd-transfer-bandwidth 10.3 \
    --pd-decider admission-and-long-prefill-aware-prefix \
    --pd-load-gate-value 1 --pd-load-gate-mode selected \
    --pd-kv-gate-value 0.04 \
    --pd-ifr-gate 19 \
    --pd-long-prefill-threshold 10000 \
    --post-hoc-detector composite \
    --metrics-path "$pd3_metrics" > "$pd3_log" 2>&1
  local pd3_disagg
  pd3_disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$pd3_log" | grep -oE '[0-9]+$' || echo 0)
  local pd3_e2e pd3_ttft
  pd3_e2e=$(jq -r '.e2e_p99_ms // 0' "$pd3_metrics")
  pd3_ttft=$(jq -r '.ttft_p99_ms // 0' "$pd3_metrics")
  printf "ttft_p99=%8.1fms  e2e_p99=%9.1fms  disagg=%s\n" "$pd3_ttft" "$pd3_e2e" "$pd3_disagg"
}

run_sweep "interactive-chat" "examples/inference-perf-interactive-chat.yaml" "$CHAT_RATE"
run_sweep "code-generation"  "examples/inference-perf-code-generation.yaml"  "$CODE_RATE"

echo "All metrics in: $OUTDIR"
