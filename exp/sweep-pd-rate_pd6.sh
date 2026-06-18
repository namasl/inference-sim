#!/usr/bin/env bash
# Sweep (rate × arrival × seed) on a fixed PD topology, computing per-cell
# tuned-fixed-threshold goodput-at-SLO baselines plus an optional treatment
# arm for a candidate dynamic decider.
#
# TOPOLOGY (fixed):
#   qwen/qwen2.5-7b-instruct on 24 H100 GPUs:
#     PD-1P+11D, TP=2 (12 instances total)
#   This is the strong-win regime: PD beats AGG-12xTP=2 by ~99% on TTFT mean
#   at r=200 chat-mt, prefix=4900, Weibull cv=1.0 across seeds {42,123,777}.
#   The campaign holds topology constant and varies only the decider.
#
# WORKLOAD:
#   examples/inference-perf-interactive-chat-mt-pd6.yaml
#     (chat-mt, prefix=4900, accumulate, max_rounds=4, think_time=45s)
#   Per-cell variants are written into $OUTDIR with overridden aggregate_rate
#   and arrival; the source file is not mutated.
#
# GRID (defaults span sub-saturation through near-ceiling):
#   rates    : 4 20 50 100 150 200 300 500    req/s
#   arrivals : poisson, weibull cv=1.0, weibull cv=1.5
#   seeds    : 42, 123, 777
#
# BASELINE ARMS:
#   Per-cell tuned fixed prefix-threshold. For each (rate, arrival, seed) cell,
#   nine threshold arms are run and the cell's "tuned-baseline goodput-at-SLO"
#   is the max over arms. Thresholds:
#     0, 16, 64, 256, 1024, 4096, 16384, never (= no PD), always (= threshold 1)
#   "never" is implemented by --pd-decider always-prefill (no disagg).
#   "always" is implemented by --pd-prefix-threshold 1 (every request disagg).
#
# TREATMENT ARM (optional):
#   Set TREATMENT_FLAGS to the BLIS flags for the candidate decider. The
#   treatment arm is run on the same grid and labelled `pd-treatment` in the
#   CSV. If TREATMENT_FLAGS is empty, only the baseline arms are run.
#   Example:
#     TREATMENT_FLAGS="--pd-decider <name> --pd-<param> <value>" \
#       examples/sweep-pd-rate_pd6.sh
#
# OPTIMIZATION TARGET — goodput-at-SLO:
#   Per-request goodput = (ttft_ms <= 500 AND itl_ms <= 50 AND e2e_ms <= 30000)
#   Cell goodput = count_meeting_SLO / num_requests.
#   p99 metrics (TTFT_p99, ITL_p99, E2E_p99) are reported but not maximized;
#   the methodology layer enforces a no-regression cap (<= 10% degradation
#   vs the per-cell tuned-baseline arm).
#
# Output: $OUTDIR (default ./blis-pd-sweep-rate_pd6)
#   chat-mt-r<rate>-<arrival>.yaml                            # per-cell workload
#   chat-mt-<arm>-r<rate>-<arrival>-s<seed>.json              # full metrics
#   chat-mt-<arm>-r<rate>-<arrival>-s<seed>.log               # blis stdout
#   chat-mt.csv                                               # one row per (arm, rate, arrival, seed)
#
# Usage:
#   examples/sweep-pd-rate_pd6.sh
#   OUTDIR=./results examples/sweep-pd-rate_pd6.sh
#   RATES="50 200" ARRIVALS="weibull-cv10" SEEDS="42" \
#     examples/sweep-pd-rate_pd6.sh
#   TREATMENT_FLAGS="--pd-decider <my-decider>" \
#     examples/sweep-pd-rate_pd6.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-./blis-pd-sweep-rate_pd6}"
RATES="${RATES:-4 20 50 100 150 200 300 500}"
ARRIVALS="${ARRIVALS:-poisson weibull-cv10 weibull-cv15}"
SEEDS="${SEEDS:-42 123 777}"
NUM_REQUESTS="${NUM_REQUESTS:-30000}"
MODEL="${MODEL:-qwen/qwen2.5-7b-instruct}"
WORKLOAD_SRC="${WORKLOAD_SRC:-examples/inference-perf-interactive-chat-mt-pd6.yaml}"
TREATMENT_FLAGS="${TREATMENT_FLAGS:-}"

mkdir -p "$OUTDIR"
[[ -x ./blis ]] || go build -o blis main.go

# Common topology flags shared by all PD arms (1P+11D TP=2, 24 H100).
PD_TOPO="--num-instances 12 --prefill-instances 1 --decode-instances 11 \
--prefill-tp 2 --decode-tp 2 --hardware H100 --pd-transfer-bandwidth 10.3"

# Baseline arms: per-cell tuned fixed threshold sweep.
# "never" = always-prefill decider (no disaggregation).
# "always" = prefix-threshold=1 (effectively every request disaggregates).
BASE_ARMS=(
  "pd-thresh-never|$PD_TOPO --pd-decider always-prefill"
  "pd-thresh-0|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 0"
  "pd-thresh-16|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 16"
  "pd-thresh-64|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 64"
  "pd-thresh-256|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 256"
  "pd-thresh-1024|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 1024"
  "pd-thresh-4096|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 4096"
  "pd-thresh-16384|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 16384"
  "pd-thresh-always|$PD_TOPO --pd-decider prefix-threshold --pd-prefix-threshold 1"
)

ARMS=("${BASE_ARMS[@]}")
if [[ -n "$TREATMENT_FLAGS" ]]; then
  ARMS+=("pd-treatment|$PD_TOPO $TREATMENT_FLAGS")
fi

CSV_HEADER="arm,rate,arrival,seed,throughput_rps,tokens_per_sec,completed,goodput,ttft_mean_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,e2e_mean_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,itl_mean_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,timeouts,preemptions,disagg_count,sat_level,sat_score"

# Render per-cell workload yaml: override aggregate_rate and arrival block.
render_spec() {
  local rate=$1 arrival=$2 out=$3
  local arrival_block
  case "$arrival" in
    poisson)
      arrival_block="    arrival:\n      process: poisson"
      ;;
    weibull-cv10)
      arrival_block="    arrival:\n      process: weibull\n      cv: 1.0"
      ;;
    weibull-cv15)
      arrival_block="    arrival:\n      process: weibull\n      cv: 1.5"
      ;;
    *) echo "unknown arrival: $arrival" >&2; exit 1 ;;
  esac
  awk -v rate="$rate" -v ab="$arrival_block" '
    /^aggregate_rate:/ { print "aggregate_rate: " rate; next }
    /^    arrival:/    { print ab; in_arrival=1; next }
    in_arrival && /^      / { next }
    in_arrival         { in_arrival=0 }
    { print }
  ' "$WORKLOAD_SRC" > "$out"
}

# Per-row goodput-at-SLO from per-request `requests`:
#   count(ttft_ms<=500 AND itl_ms<=50 AND e2e_ms<=30000) / N
goodput_jq='
  if (.requests // []) | length == 0 then 0
  else
    ((.requests
        | map(select(.ttft_ms <= 500 and .itl_ms <= 50 and .e2e_ms <= 30000))
        | length)
     / (.requests | length))
  end
'

csv="$OUTDIR/chat-mt.csv"
echo "$CSV_HEADER" > "$csv"

read -ra rates    <<< "$RATES"
read -ra arrivals <<< "$ARRIVALS"
read -ra seeds    <<< "$SEEDS"

echo "=== chat-mt (rate × arrival × seed × decider) ==="

for spec in "${ARMS[@]}"; do
  arm="${spec%%|*}"
  flags="${spec#*|}"
  echo "  arm: $arm"

  for rate in "${rates[@]}"; do
    for arrival in "${arrivals[@]}"; do
      cfg_yaml="$OUTDIR/chat-mt-r${rate}-${arrival}.yaml"
      [[ -f "$cfg_yaml" ]] || render_spec "$rate" "$arrival" "$cfg_yaml"

      for seed in "${seeds[@]}"; do
        metrics="$OUTDIR/chat-mt-${arm}-r${rate}-${arrival}-s${seed}.json"
        log="$OUTDIR/chat-mt-${arm}-r${rate}-${arrival}-s${seed}.log"

        printf "    rate=%-4s arr=%-13s seed=%-4s ... " "$rate" "$arrival" "$seed"

        # shellcheck disable=SC2086
        ./blis run \
          --model "$MODEL" \
          --workload-spec "$cfg_yaml" \
          --num-requests "$NUM_REQUESTS" \
          --seed "$seed" \
          $flags \
          --post-hoc-detector composite \
          --metrics-path "$metrics" > "$log" 2>&1

        disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)
        goodput=$(jq -r "$goodput_jq" "$metrics")

        jq -r --arg arm "$arm" --arg r "$rate" --arg a "$arrival" --arg s "$seed" \
              --argjson d "$disagg" --argjson g "$goodput" '
          [$arm, $r, $a, $s,
           (.responses_per_sec // 0),
           (.tokens_per_sec // 0),
           (.completed_requests // 0),
           $g,
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

        ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
        e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
        sat=$(jq -r  '.saturation.level // "?"' "$metrics")
        printf "ttft_p99=%8.1fms  e2e_p99=%9.1fms  goodput=%5.3f  sat=%-10s  disagg=%s\n" \
          "$ttft" "$e2e" "$goodput" "$sat" "$disagg"
      done
    done
  done
  echo ""
done

# Summary: per-cell tuned-baseline goodput = max over baseline-arm seed-mean
# goodput; treatment goodput shown side-by-side when present.
echo "Summary (per-cell tuned-baseline vs treatment, seed-mean goodput-at-SLO):"
printf "  %-4s %-13s %-18s %8s %10s %10s\n" \
  "rate" "arrival" "best_baseline_arm" "base_gp" "treat_gp" "ttft_p99"
tail -n +2 "$csv" | awk -F, '
  function strip(s) { gsub(/"/, "", s); return s }
  {
    arm=strip($1); r=strip($2); a=strip($3)
    k=r","a","arm
    n[k]++; gp[k]+=$8; ttft[k]+=$12
  }
  END {
    for (k in n) {
      mean_gp[k]=gp[k]/n[k]
      mean_ttft[k]=ttft[k]/n[k]
      split(k, kk, ",")
      cell=kk[1]","kk[2]
      arm=kk[3]
      if (arm == "pd-treatment") {
        treat_gp[cell]=mean_gp[k]
        treat_ttft[cell]=mean_ttft[k]
      } else {
        if (mean_gp[k] > best_gp[cell]) {
          best_gp[cell]=mean_gp[k]
          best_arm[cell]=arm
          best_ttft[cell]=mean_ttft[k]
        }
      }
    }
    for (cell in best_gp) {
      split(cell, cc, ",")
      tg = (cell in treat_gp) ? sprintf("%.4f", treat_gp[cell]) : "-"
      tt = (cell in treat_ttft) ? sprintf("%.1f", treat_ttft[cell]) : "-"
      printf "%s\t%s\t%s\t%.4f\t%s\t%.1f / %s\n",
        cc[1], cc[2], best_arm[cell], best_gp[cell], tg, best_ttft[cell], tt
    }
  }
' | sort -t$'\t' -k1 -g -k2 | awk -F'\t' '
  {
    printf "  %-4s %-13s %-18s %8s %10s %10s\n", $1, $2, $3, $4, $5, $6
  }
'

echo ""
echo "All metrics: $csv"
echo "All results in: $OUTDIR"
