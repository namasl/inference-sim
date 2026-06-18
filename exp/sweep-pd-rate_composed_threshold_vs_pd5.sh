#!/usr/bin/env bash
# Sweep (chat_rate × code_rate × arrival_process × seed) for a *composed*
# mixed-tenancy workload (chat + code combined via `blis compose`), comparing
# three 16-GPU cluster configurations on meta-llama/llama-3.3-70b-instruct,
# H100 TP=4:
#
#   agg-4xtp4    — 4 co-located instances, TP=4 (no PD disaggregation)
#                  Latest BLIS.
#   pd-thresh16  — 1P+3D TP=4, prefix-threshold decider at threshold=16
#                  Latest BLIS. This is the current llm-d-parity default.
#   pd-pd5       — 1P+3D TP=4, RollingInputRateAdaptivePrefixDecider
#                  with rt=25, windowUs=2s, contentThreshold=10000,
#                  lowThreshold=0, highThreshold=4096, ringSize=128.
#                  This is the campaign's best decider (.nous/pd5/report.md):
#                  +46% goodput-at-SLO over per-cell-tuned baseline at
#                  c100-co10-poisson, 96% of A0 at c100-co10-weibull, and
#                  byte-identical to A0 in chat-only cells.
#
# Workloads match the pd5 campaign exactly:
#   - inference-perf-interactive-chat.yaml   (chat client)
#   - inference-perf-code-generation.yaml    (code client)
#   composed via `blis compose` with rate split chat:code matching the
#   per-cell aggregate.
#
# Binaries:
#   blis      — built from CWD HEAD, used for agg-4xtp4 and pd-thresh16.
#   blis-pd5  — built from a temporary worktree at HEAD with
#               examples/pd5.patch applied; used for the pd-pd5 arm.
#
# Output: $OUTDIR (default ./blis-pd-sweep-rate_composed_threshold_vs_pd5)
#   composed-<chat>-<code>-<arrival>.yaml             # composed workload spec per cell
#   composed-<arm>-<chat>-<code>-<arrival>-<seed>.json
#   composed-<arm>-<chat>-<code>-<arrival>-<seed>.log
#   composed.csv                                      # one row per (arm, chat, code, arrival, seed)
#
# The campaign uses --num-requests 1000 (front-loaded session pool), not
# --horizon, so we match that here for parity with .nous/pd5 results.
#
# Usage:
#   examples/sweep-pd-rate_composed_threshold_vs_pd5.sh
#   OUTDIR=./results examples/sweep-pd-rate_composed_threshold_vs_pd5.sh
#   CHAT_RATES="100 150" CODE_RATES="1 5" SEEDS="42" ARRIVALS="poisson" \
#     examples/sweep-pd-rate_composed_threshold_vs_pd5.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-./blis-pd-sweep-rate_composed_threshold_vs_pd5}"
# Defaults span the campaign's full pd5.yaml grid.
CHAT_RATES="${CHAT_RATES:-50 100 150 200}"
CODE_RATES="${CODE_RATES:-0.5 1 5 10 20}"
ARRIVALS="${ARRIVALS:-poisson weibull}"
SEEDS="${SEEDS:-42 123 777}"
NUM_REQUESTS="${NUM_REQUESTS:-1000}"
MODEL="meta-llama/llama-3.3-70b-instruct"
CHAT_SPEC="examples/inference-perf-interactive-chat.yaml"
CODE_SPEC="examples/inference-perf-code-generation.yaml"

mkdir -p "$OUTDIR"

# Standard blis: agg-4xtp4 and pd-thresh16, built from current HEAD.
go build -o blis main.go

# pd5 binary: worktree at current HEAD + examples/pd5.patch.
PD5_PATCH="$PWD/examples/pd5.patch"
REPO_ROOT="$PWD"
PD5_BASE_COMMIT="$(git rev-parse HEAD)"
PD5_WT="$(mktemp -d -t blis-pd5-XXXXXX)"
trap 'git worktree remove --force "$PD5_WT" >/dev/null 2>&1 || true; rm -rf "$PD5_WT"' EXIT
git worktree add --detach "$PD5_WT" "$PD5_BASE_COMMIT"
git -C "$PD5_WT" apply --check "$PD5_PATCH"
git -C "$PD5_WT" apply "$PD5_PATCH"
(cd "$PD5_WT" && go build -o "$REPO_ROOT/blis-pd5" main.go)
git worktree remove --force "$PD5_WT"
trap - EXIT

# Arms: "label|binary|flags". 16 GPUs each (4 instances × TP=4).
ARMS=(
  "agg-4xtp4|./blis|--num-instances 4 --tp 4 --hardware H100"
  "pd-thresh16|./blis|--num-instances 4 --prefill-instances 1 --decode-instances 3 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider prefix-threshold --pd-prefix-threshold 16"
  "pd-pd5|./blis-pd5|--num-instances 4 --prefill-instances 1 --decode-instances 3 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider rolling-input-rate-adaptive-prefix --pd-rolling-input-rate-content-threshold 10000 --pd-rolling-input-rate-window-us 2000000 --pd-rolling-input-rate-threshold 25 --pd-rolling-input-rate-low-threshold 0 --pd-rolling-input-rate-high-threshold 4096 --pd-rolling-input-rate-ring-size 128"
)

CSV_HEADER="arm,chat_rate,code_rate,arrival,seed,throughput_rps,tokens_per_sec,completed,goodput,ttft_mean_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,e2e_mean_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,itl_mean_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,timeouts,preemptions,disagg_count,sat_level,sat_score"

# Compose a mixed workload spec for one (chat, code, arrival) tuple. Writes
# per-cell rate-adjusted source YAMLs into $OUTDIR (so we don't mutate the
# originals) then runs `blis compose` to merge them. arrival.process is
# rewritten in both client specs.
compose_spec() {
  local chat=$1 code=$2 arrival=$3 out=$4
  local chat_yaml="$OUTDIR/_chat-${chat}-${arrival}.yaml"
  local code_yaml="$OUTDIR/_code-${code}-${arrival}.yaml"
  local arrival_block
  if [[ "$arrival" == "weibull" ]]; then
    arrival_block="    arrival:\n      process: weibull\n      cv: 2"
  elif [[ "$arrival" == "gamma" ]]; then
    arrival_block="    arrival:\n      process: gamma\n      shape: 2.0"
  else
    arrival_block="    arrival:\n      process: poisson"
  fi
  # Awk replaces the multi-line `arrival:\n  process: poisson` block with the
  # configured arrival, and overrides aggregate_rate. The source files have
  # `arrival:` blocks with exactly one nested `process:` line each, so a
  # state-machine approach is robust to indentation.
  awk -v rate="$chat" -v ab="$arrival_block" '
    /^aggregate_rate:/ { print "aggregate_rate: " rate; next }
    /^    arrival:/    { print ab; in_arrival=1; next }
    in_arrival && /^      / { next }
    in_arrival         { in_arrival=0 }
    { print }
  ' "$CHAT_SPEC" > "$chat_yaml"
  awk -v rate="$code" -v ab="$arrival_block" '
    /^aggregate_rate:/ { print "aggregate_rate: " rate; next }
    /^    arrival:/    { print ab; in_arrival=1; next }
    in_arrival && /^      / { next }
    in_arrival         { in_arrival=0 }
    { print }
  ' "$CODE_SPEC" > "$code_yaml"
  ./blis compose --from "$chat_yaml" --from "$code_yaml" > "$out"
}

# Compute per-row goodput-at-SLO from the per-request `requests` array in the
# metrics JSON: count(ttft_ms<=500 AND itl_ms<=50 AND e2e_ms<=30000) / N.
# This matches the campaign's goodput formula (see .nous/pd5/handoff.md).
goodput_jq='
  if (.requests // []) | length == 0 then 0
  else
    ((.requests
        | map(select(.ttft_ms <= 500 and .itl_ms <= 50 and .e2e_ms <= 30000))
        | length)
     / (.requests | length))
  end
'

csv="$OUTDIR/composed.csv"
echo "$CSV_HEADER" > "$csv"

read -ra chat_rates <<< "$CHAT_RATES"
read -ra code_rates <<< "$CODE_RATES"
read -ra arrivals   <<< "$ARRIVALS"
read -ra seeds      <<< "$SEEDS"

echo "=== composed (chat × code × arrival mixed-tenancy) ==="

for spec in "${ARMS[@]}"; do
  arm="${spec%%|*}"
  rest="${spec#*|}"
  bin="${rest%%|*}"
  flags="${rest#*|}"

  echo "  arm: $arm  ($bin)"
  for chat in "${chat_rates[@]}"; do
    for code in "${code_rates[@]}"; do
      for arrival in "${arrivals[@]}"; do
        cfg_yaml="$OUTDIR/composed-${chat}-${code}-${arrival}.yaml"
        [[ -f "$cfg_yaml" ]] || compose_spec "$chat" "$code" "$arrival" "$cfg_yaml"

        for seed in "${seeds[@]}"; do
          metrics="$OUTDIR/composed-${arm}-${chat}-${code}-${arrival}-${seed}.json"
          log="$OUTDIR/composed-${arm}-${chat}-${code}-${arrival}-${seed}.log"

          printf "    chat=%-5s code=%-5s arr=%-7s seed=%-4s ... " \
            "$chat" "$code" "$arrival" "$seed"

          # shellcheck disable=SC2086
          "$bin" run \
            --model "$MODEL" \
            --workload-spec "$cfg_yaml" \
            --num-requests "$NUM_REQUESTS" \
            --seed "$seed" \
            $flags \
            --post-hoc-detector composite \
            --metrics-path "$metrics" > "$log" 2>&1

          disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)
          goodput=$(jq -r "$goodput_jq" "$metrics")

          jq -r --arg arm "$arm" --arg c "$chat" --arg co "$code" \
                --arg a "$arrival" --arg s "$seed" \
                --argjson d "$disagg" --argjson g "$goodput" '
            [$arm, $c, $co, $a, $s,
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
  done
  echo ""
done

# Summary: seed-mean per (arm, chat, code, arrival), grouped by load cell so
# each block compares the three arms head-to-head on identical workload.
echo "Summary (seed-mean per arm × cell, grouped by chat × code × arrival):"
printf "  %-5s %-5s %-7s %-12s %8s %10s %10s %10s %10s %8s %12s\n" \
  "chat" "code" "arr" "arm" "goodput" "ttft_p99" "e2e_p99" "itl_p99" "rps" "disagg" "sat(mode)"
tail -n +2 "$csv" | awk -F, '
  function strip(s) { gsub(/"/, "", s); return s }
  {
    arm=strip($1); c=strip($2); co=strip($3); a=strip($4); k=c","co","a","arm
    n[k]++
    gp[k]+=$9
    ttft[k]+=$13; e2e[k]+=$17; itl[k]+=$21; rps[k]+=$6; dis[k]+=$24
    satk=k","strip($25); satc[satk]++
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
      printf "%s\t%s\t%s\t%s\t%.4f\t%.1f\t%.1f\t%.1f\t%.3f\t%d\t%s\n",
        kk[1], kk[2], kk[3], kk[4],
        gp[k]/n[k], ttft[k]/n[k], e2e[k]/n[k], itl[k]/n[k],
        rps[k]/n[k], int(dis[k]/n[k]), best
    }
  }
' | sort -t$'\t' -k1 -g -k2 -g -k3 -k4 | awk -F'\t' '
  {
    printf "  %-5s %-5s %-7s %-12s %8.4f %10.1f %10.1f %10.1f %10.3f %8d %12s\n",
      $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
  }
'

echo ""
echo "All metrics: $csv"
echo "All results in: $OUTDIR"
