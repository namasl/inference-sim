#!/usr/bin/env bash
# Sweep aggregate_rate × seed comparing scorer profiles from the
# .nous/pd_scorer_reasoning campaign on the decode-heavy reasoning workload,
# with two AGG references for cost/architecture context. PD arms use 24 H100
# GPUs (6 instances × TP=4); AGG arms span 12–24 GPUs.
#
# Arms:
#   agg-3xtp4    — 3 co-located instances, TP=4 (12 GPUs).
#                  Per .nous/pd_scorer_reasoning/report.md RP-5: this beats
#                  PD 3P+3D (24 GPU) on decode-heavy reasoning at 10.3 GB/s.
#                  Reference for the half-hardware-still-wins finding.
#   agg-6xtp4    — 6 co-located instances, TP=4 (24 GPUs).
#                  GPU-matched reference: same hardware as PD arms.
#   pd-default   — 3P+3D TP=4 with explicit symmetric default scorer on BOTH
#                  pools (matched-RNG baseline; campaign A0).
#   pd-champ     — 3P+3D TP=4 with the campaign-final champion (RP-16):
#                    prefill: precise-prefix-cache:1
#                    decode:  queue-depth:1, kv-utilization:1
#                  Drops decode precise-prefix-cache entirely. 12/12 Pareto
#                  wins vs pd-default; ITL p95 −6 to −30%, TTFT p99 −2 to −9%.
#   pd-alt       — 3P+3D TP=4 with the report's acceptable alternative:
#                    prefill: precise-prefix-cache:1
#                    decode:  precise-prefix-cache:1, queue-depth:1,
#                             kv-utilization:1
#                  Also 12/12 wins, smaller ITL margin at r=1 vs pd-champ.
#
# Matched-RNG protocol: setting any --prefill-routing-scorers flag switches
# the prefill router onto its own PRNG stream. To keep the baseline comparable
# to the asymmetric arms, pd-default explicitly sets BOTH flags to the default
# values rather than relying on the no-flag default.
#
# Workload & rate envelope follow the campaign exactly:
#   reasoning: {1, 2, 3, 5} req/s
# Seeds: 42, 123, 777.
#
# Output: $OUTDIR (default ./blis-pd-sweep-rate_scorers_reasoning)
#   <workload>-<arm>-<rate>-<seed>.json   # full metrics JSON per cell
#   <workload>-<arm>-<rate>-<seed>.log    # blis stdout (PD metrics + LIR)
#   <workload>.csv                        # one row per (arm, rate, seed)
#
# Usage:
#   examples/sweep-pd-rate_scorers_reasoning.sh
#   OUTDIR=./results examples/sweep-pd-rate_scorers_reasoning.sh
#   RATES="1 5" SEEDS="42" examples/sweep-pd-rate_scorers_reasoning.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-./blis-pd-sweep-rate_scorers_reasoning}"
RATES="${RATES:-1 2 3 5}"
SEEDS="${SEEDS:-42 123 777}"
MODEL="meta-llama/llama-3.3-70b-instruct"

mkdir -p "$OUTDIR"
[[ -x ./blis ]] || go build -o blis main.go

# Default symmetric scorer (sim/routing_scorers.go DefaultScorerConfigs()).
DEFAULT_SCORERS="precise-prefix-cache:2,queue-depth:1,kv-utilization:1"

# Campaign-final champion (RP-16): drop decode precise-prefix-cache entirely.
CHAMP_PREFILL_SCORERS="precise-prefix-cache:1"
CHAMP_DECODE_SCORERS="queue-depth:1,kv-utilization:1"

# Report's acceptable alternative: keep decode prec at weight 1.
ALT_PREFILL_SCORERS="precise-prefix-cache:1"
ALT_DECODE_SCORERS="precise-prefix-cache:1,queue-depth:1,kv-utilization:1"

# 3P+3D base flags shared by all PD arms.
PD_BASE="--num-instances 6 --prefill-instances 3 --decode-instances 3 \
--prefill-tp 4 --decode-tp 4 --hardware H100 \
--pd-transfer-bandwidth 10.3 \
--pd-decider prefix-threshold --pd-prefix-threshold 16"

# Arms: "label|flags".
ARMS=(
  "agg-3xtp4|--num-instances 3 --tp 4 --hardware H100"
  "agg-6xtp4|--num-instances 6 --tp 4 --hardware H100"
  "pd-default|$PD_BASE --prefill-routing-scorers $DEFAULT_SCORERS --decode-routing-scorers $DEFAULT_SCORERS"
  "pd-champ|$PD_BASE --prefill-routing-scorers $CHAMP_PREFILL_SCORERS --decode-routing-scorers $CHAMP_DECODE_SCORERS"
  "pd-alt|$PD_BASE --prefill-routing-scorers $ALT_PREFILL_SCORERS --decode-routing-scorers $ALT_DECODE_SCORERS"
)

CSV_HEADER="arm,rate,seed,throughput_rps,tokens_per_sec,completed,ttft_mean_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,e2e_mean_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,itl_mean_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,timeouts,preemptions,disagg_count,lir,sat_level,sat_score"

run_sweep() {
  local wl_name=$1 wl_path=$2
  shift 2
  local rates=("$@")
  local csv="$OUTDIR/${wl_name}.csv"

  echo "$CSV_HEADER" > "$csv"
  echo "=== $wl_name ==="

  read -ra seeds <<< "$SEEDS"

  for spec in "${ARMS[@]}"; do
    local arm="${spec%%|*}"
    local flags="${spec#*|}"

    echo "  arm: $arm"
    for rate in "${rates[@]}"; do
      local cfg_yaml="$OUTDIR/${wl_name}-${rate}.yaml"
      sed "s/^aggregate_rate: .*/aggregate_rate: $rate/" "$wl_path" > "$cfg_yaml"

      for seed in "${seeds[@]}"; do
        local metrics="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.json"
        local log="$OUTDIR/${wl_name}-${arm}-${rate}-${seed}.log"

        printf "    rate=%-5s seed=%-4s ... " "$rate" "$seed"

        # shellcheck disable=SC2086
        ./blis run \
          --model "$MODEL" \
          --workload-spec "$cfg_yaml" \
          --seed "$seed" \
          $flags \
          --post-hoc-detector composite \
          --metrics-path "$metrics" > "$log" 2>&1

        # PD metrics live in stdout, not the JSON. Aggregate arms have no PD
        # metrics; default disagg=0 and lir="" (unset).
        local disagg lir
        disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)
        lir=$(grep -oE 'Load Imbalance Ratio: [0-9.]+' "$log" | grep -oE '[0-9.]+$' || echo "")

        jq -r --arg arm "$arm" --arg r "$rate" --arg s "$seed" \
              --argjson d "$disagg" --arg lir "$lir" '
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
           $lir,
           (.saturation.level // ""),
           (.saturation.score // 0)
          ] | @csv
        ' "$metrics" >> "$csv"

        local ttft e2e sat
        ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
        e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
        sat=$(jq -r  '.saturation.level // "?"' "$metrics")
        printf "ttft_p99=%8.1fms  e2e_p99=%9.1fms  sat=%-10s  disagg=%-4s  lir=%s\n" \
          "$ttft" "$e2e" "$sat" "$disagg" "${lir:-N/A}"
      done
    done
    echo ""
  done

  # Summary: seed-mean per (arm, rate), grouped by rate so all arms appear
  # together for head-to-head comparison at each load.
  echo "  Summary (seed-mean per arm × rate, grouped by rate):"
  printf "  %-12s %-6s %10s %10s %10s %10s %8s %8s %12s\n" \
    "arm" "rate" "ttft_p99" "e2e_p99" "itl_p99" "rps" "disagg" "lir" "sat(mode)"
  tail -n +2 "$csv" | awk -F, '
    function strip(s) { gsub(/"/, "", s); return s }
    {
      arm=strip($1); r=strip($2); k=arm","r
      n[k]++
      ttft[k]+=$10; e2e[k]+=$14; itl[k]+=$18; rps[k]+=$4; dis[k]+=$21
      lirv=strip($22)
      if (lirv != "") { lir[k]+=lirv; lirn[k]++ }
      satk=k","strip($23); satc[satk]++
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
        lirstr = (lirn[k] > 0) ? sprintf("%.3f", lir[k]/lirn[k]) : "N/A"
        printf "%s\t%s\t%.1f\t%.1f\t%.1f\t%.3f\t%d\t%s\t%s\n",
          kk[1], kk[2], ttft[k]/n[k], e2e[k]/n[k], itl[k]/n[k],
          rps[k]/n[k], int(dis[k]/n[k]), lirstr, best
      }
    }
  ' | sort -t$'\t' -k2 -g -k1 | awk -F'\t' '
    {
      printf "  %-12s %-6s %10.1f %10.1f %10.1f %10.3f %8d %8s %12s\n",
        $1, $2, $3, $4, $5, $6, $7, $8, $9
    }
  '
  echo ""
  echo "  All metrics: $csv"
  echo ""
}

read -ra rates <<< "$RATES"

run_sweep "reasoning" "examples/inference-perf-reasoning.yaml" "${rates[@]}"

echo "All results in: $OUTDIR"
