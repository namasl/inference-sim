#!/usr/bin/env bash
# Sweep aggregate_rate × seed comparing the symmetric default scorer profile
# against the asymmetric champion profile from the .nous/pd_scorer campaign,
# on the 3P+3D-3g topology that campaign was tuned for. All arms use 24 H100
# GPUs (6 instances × TP=4) on meta-llama/llama-3.3-70b-instruct, with the
# 3-prefix-group workloads inference-perf-{interactive-chat,code-generation}-3g.yaml.
#
# Arms:
#   agg-6xtp4    — 6 co-located instances, TP=4 (no PD disaggregation).
#                  Reference for PD-advantage confirmation.
#   pd-default   — 3P+3D TP=4 with explicit symmetric default scorer on BOTH
#                  pools (matched-RNG baseline; see RP-7/RP-10).
#   pd-asym      — 3P+3D TP=4 with the campaign-final asymmetric profile:
#                    prefill: prefix-affinity:0.1, precise-prefix-cache:1,
#                             queue-depth:1, kv-utilization:1   (pa_eff ≈ 0.032)
#                    decode:  precise-prefix-cache:2, queue-depth:1,
#                             kv-utilization:1                  (default)
#                  Per .nous/pd_scorer/report.md: −20% to −26% TTFT p99 on
#                  chat-r400, −27% to −51% on code-r40/r100 vs symmetric default.
#
# Matched-RNG protocol (RP-7, RP-10): setting any --prefill-routing-scorers
# flag switches the prefill router onto its own PRNG stream. To keep the
# baseline comparable to the asymmetric arm, the pd-default arm explicitly
# sets BOTH --prefill-routing-scorers and --decode-routing-scorers to their
# default values rather than relying on the no-flag default.
#
# Workloads & rate envelope follow the campaign exactly:
#   interactive-chat-3g: {200, 300, 400} req/s
#   code-generation-3g:  {40, 80, 100, 200} req/s
# Seeds: 42, 123, 777.
#
# Output: $OUTDIR (default ./blis-pd-sweep-rate_scorers)
#   <workload>-<arm>-<rate>-<seed>.json   # full metrics JSON per cell
#   <workload>-<arm>-<rate>-<seed>.log    # blis stdout (PD metrics + LIR)
#   <workload>.csv                        # one row per (arm, rate, seed)
#
# Usage:
#   examples/sweep-pd-rate_scorers.sh
#   OUTDIR=./results examples/sweep-pd-rate_scorers.sh
#   CHAT_RATES="200 400" CODE_RATES="40 100" SEEDS="42" \
#     examples/sweep-pd-rate_scorers.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-./blis-pd-sweep-rate_scorers}"
CHAT_RATES="${CHAT_RATES:-200 300 400}"
CODE_RATES="${CODE_RATES:-40 80 100 200}"
SEEDS="${SEEDS:-42 123 777}"
MODEL="meta-llama/llama-3.3-70b-instruct"

mkdir -p "$OUTDIR"
[[ -x ./blis ]] || go build -o blis main.go

# Default symmetric scorer (sim/routing_scorers.go DefaultScorerConfigs()).
DEFAULT_SCORERS="precise-prefix-cache:2,queue-depth:1,kv-utilization:1"

# Campaign-final asymmetric prefill profile (pa_eff ≈ 0.032, mid-plateau).
ASYM_PREFILL_SCORERS="prefix-affinity:0.1,precise-prefix-cache:1,queue-depth:1,kv-utilization:1"
ASYM_DECODE_SCORERS="$DEFAULT_SCORERS"

# 3P+3D base flags shared by pd-default and pd-asym.
PD_BASE="--num-instances 6 --prefill-instances 3 --decode-instances 3 \
--prefill-tp 4 --decode-tp 4 --hardware H100 \
--pd-transfer-bandwidth 10.3 \
--pd-decider prefix-threshold --pd-prefix-threshold 16"

# Arms: "label|flags". 24 GPUs each.
ARMS=(
  "agg-6xtp4|--num-instances 6 --tp 4 --hardware H100"
  "pd-default|$PD_BASE --prefill-routing-scorers $DEFAULT_SCORERS --decode-routing-scorers $DEFAULT_SCORERS"
  "pd-asym|$PD_BASE --prefill-routing-scorers $ASYM_PREFILL_SCORERS --decode-routing-scorers $ASYM_DECODE_SCORERS"
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

        # PD metrics live in stdout, not the JSON. Aggregate arm has no PD
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

  # Summary: seed-mean per (arm, rate), grouped by rate so all three arms
  # appear together for head-to-head comparison at each load.
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

read -ra chat_rates <<< "$CHAT_RATES"
read -ra code_rates <<< "$CODE_RATES"

run_sweep "interactive-chat-3g" "examples/inference-perf-interactive-chat-3g.yaml" "${chat_rates[@]}"
run_sweep "code-generation-3g"  "examples/inference-perf-code-generation-3g.yaml"  "${code_rates[@]}"

echo "All results in: $OUTDIR"
