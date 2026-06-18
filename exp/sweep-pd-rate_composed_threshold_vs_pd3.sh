#!/usr/bin/env bash
# Sweep (chat_rate × code_rate × seed) for a *composed* mixed-tenancy workload
# (chat + code combined via `blis compose`), comparing three 16-GPU cluster
# configurations on meta-llama/llama-3.3-70b-instruct, H100 TP=4:
#
#   agg-4xtp4    — 4 co-located instances, TP=4 (no PD disaggregation)
#   pd-thresh16  — 1P+3D TP=4, prefix-threshold decider at threshold=16
#   pd-pd3       — 1P+3D TP=4, AdmissionAndLongPrefillAwarePrefixDecider
#                  (3-clause OR: qdGate=1, kvGate=0.04, ifrGate=19,
#                   longPrefillThreshold=10000)
#
# This is the regime where pd3 was tuned and validated in .nous/pd3/report.md
# (10/12 Pareto vs threshold-16 on a chat ∈ {100,150} × code ∈ {1,5} × seeds
# {42,123,777} mixed grid). Single-workload sweeps don't exercise the kvFires
# /ifrFires clauses; mixed tenancy does.
#
# Binaries:
#   blis      — built from CWD, used for agg-4xtp4 and pd-thresh16.
#   blis-pd3  — built from a temporary worktree pinned to PD3_BASE_COMMIT with
#               examples/pd3.patch applied; used for the pd-pd3 arm.
#
# Output: $OUTDIR (default ./blis-pd-sweep-rate_composed_threshold_vs_pd3)
#   composed-<chat>-<code>.yaml             # composed workload spec per cell
#   composed-<arm>-<chat>-<code>-<seed>.json
#   composed-<arm>-<chat>-<code>-<seed>.log
#   composed.csv                            # one row per (arm, chat, code, seed)
#
# `blis compose` drops num_requests from the merged spec, so each run is
# bounded by --horizon (default 30s, override via HORIZON_S).
#
# Usage:
#   examples/sweep-pd-rate_composed_threshold_vs_pd3.sh
#   OUTDIR=./results examples/sweep-pd-rate_composed_threshold_vs_pd3.sh
#   CHAT_RATES="100 150" CODE_RATES="1 5" SEEDS="42" \
#     examples/sweep-pd-rate_composed_threshold_vs_pd3.sh
#   HORIZON_S=60 examples/sweep-pd-rate_composed_threshold_vs_pd3.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-./blis-pd-sweep-rate_composed_threshold_vs_pd3}"
# Defaults span the campaign's pd3 sweet spot plus light/heavy bracketing.
CHAT_RATES="${CHAT_RATES:-50 80 100 120 150}"
CODE_RATES="${CODE_RATES:-0.5 1 2 5}"
SEEDS="${SEEDS:-42 123 777}"
# `blis compose` drops num_requests from the merged spec, so generation must
# be bounded by --horizon. 30s × the configured rates gives 1500–4500 chat
# requests and 15–150 code requests per cell, which is enough for stable
# percentile estimation when averaged over 3 seeds.
HORIZON_S="${HORIZON_S:-30}"
HORIZON_TICKS=$((HORIZON_S * 1000000))
MODEL="meta-llama/llama-3.3-70b-instruct"
CHAT_SPEC="examples/inference-perf-interactive-chat.yaml"
CODE_SPEC="examples/inference-perf-code-generation.yaml"

mkdir -p "$OUTDIR"

# Standard blis: agg-4xtp4 and pd-thresh16.
go build -o blis main.go

# pd3 binary: worktree+patch (see sweep-pd-threshold_pd3.sh for rationale).
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

# Arms: "label|binary|flags". 16 GPUs each (4 instances × TP=4).
ARMS=(
  "agg-4xtp4|./blis|--num-instances 4 --tp 4 --hardware H100"
  "pd-thresh16|./blis|--num-instances 4 --prefill-instances 1 --decode-instances 3 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider prefix-threshold --pd-prefix-threshold 16"
  "pd-pd3|./blis-pd3|--num-instances 4 --prefill-instances 1 --decode-instances 3 --prefill-tp 4 --decode-tp 4 --hardware H100 --pd-transfer-bandwidth 10.3 --pd-decider admission-and-long-prefill-aware-prefix --pd-load-gate-value 1 --pd-load-gate-mode selected --pd-kv-gate-value 0.04 --pd-ifr-gate 19 --pd-long-prefill-threshold 10000"
)

CSV_HEADER="arm,chat_rate,code_rate,seed,throughput_rps,tokens_per_sec,completed,ttft_mean_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,e2e_mean_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,itl_mean_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,timeouts,preemptions,disagg_count,sat_level,sat_score"

# Compose a mixed workload spec for one (chat, code) pair. Writes per-cell
# rate-adjusted source YAMLs into $OUTDIR (so we don't mutate the originals)
# then runs `blis compose` to merge them.
compose_spec() {
  local chat=$1 code=$2 out=$3
  local chat_yaml="$OUTDIR/_chat-${chat}.yaml"
  local code_yaml="$OUTDIR/_code-${code}.yaml"
  sed "s/^aggregate_rate: .*/aggregate_rate: $chat/" "$CHAT_SPEC" > "$chat_yaml"
  sed "s/^aggregate_rate: .*/aggregate_rate: $code/" "$CODE_SPEC" > "$code_yaml"
  ./blis compose --from "$chat_yaml" --from "$code_yaml" > "$out"
}

csv="$OUTDIR/composed.csv"
echo "$CSV_HEADER" > "$csv"

read -ra chat_rates <<< "$CHAT_RATES"
read -ra code_rates <<< "$CODE_RATES"
read -ra seeds      <<< "$SEEDS"

echo "=== composed (chat × code mixed-tenancy) ==="

for spec in "${ARMS[@]}"; do
  arm="${spec%%|*}"
  rest="${spec#*|}"
  bin="${rest%%|*}"
  flags="${rest#*|}"

  echo "  arm: $arm  ($bin)"
  for chat in "${chat_rates[@]}"; do
    for code in "${code_rates[@]}"; do
      cfg_yaml="$OUTDIR/composed-${chat}-${code}.yaml"
      [[ -f "$cfg_yaml" ]] || compose_spec "$chat" "$code" "$cfg_yaml"

      for seed in "${seeds[@]}"; do
        metrics="$OUTDIR/composed-${arm}-${chat}-${code}-${seed}.json"
        log="$OUTDIR/composed-${arm}-${chat}-${code}-${seed}.log"

        printf "    chat=%-5s code=%-5s seed=%-4s ... " "$chat" "$code" "$seed"

        # shellcheck disable=SC2086
        "$bin" run \
          --model "$MODEL" \
          --workload-spec "$cfg_yaml" \
          --seed "$seed" \
          --horizon "$HORIZON_TICKS" \
          $flags \
          --post-hoc-detector composite \
          --metrics-path "$metrics" > "$log" 2>&1

        disagg=$(grep -oE 'Disaggregated Requests: [0-9]+' "$log" | grep -oE '[0-9]+$' || echo 0)

        jq -r --arg arm "$arm" --arg c "$chat" --arg co "$code" --arg s "$seed" --argjson d "$disagg" '
          [$arm, $c, $co, $s,
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

        ttft=$(jq -r '.ttft_p99_ms // 0' "$metrics")
        e2e=$(jq -r  '.e2e_p99_ms  // 0' "$metrics")
        sat=$(jq -r  '.saturation.level // "?"' "$metrics")
        printf "ttft_p99=%8.1fms  e2e_p99=%9.1fms  sat=%-10s  disagg=%s\n" \
          "$ttft" "$e2e" "$sat" "$disagg"
      done
    done
  done
  echo ""
done

# Summary: seed-mean per (arm, chat, code), grouped by (chat, code) so each
# block compares the three arms head-to-head on identical load.
echo "Summary (seed-mean per arm × chat × code, grouped by load):"
printf "  %-5s %-5s %-12s %10s %10s %10s %10s %8s %12s\n" \
  "chat" "code" "arm" "ttft_p99" "e2e_p99" "itl_p99" "rps" "disagg" "sat(mode)"
tail -n +2 "$csv" | awk -F, '
  function strip(s) { gsub(/"/, "", s); return s }
  {
    arm=strip($1); c=strip($2); co=strip($3); k=c","co","arm
    n[k]++
    ttft[k]+=$11; e2e[k]+=$15; itl[k]+=$19; rps[k]+=$5; dis[k]+=$22
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
      printf "%s\t%s\t%s\t%.1f\t%.1f\t%.1f\t%.3f\t%d\t%s\n",
        kk[1], kk[2], kk[3], ttft[k]/n[k], e2e[k]/n[k], itl[k]/n[k],
        rps[k]/n[k], int(dis[k]/n[k]), best
    }
  }
' | sort -t$'\t' -k1 -g -k2 -g -k3 | awk -F'\t' '
  {
    printf "  %-5s %-5s %-12s %10.1f %10.1f %10.1f %10.3f %8d %12s\n",
      $1, $2, $3, $4, $5, $6, $7, $8, $9
  }
'

echo ""
echo "All metrics: $csv"
echo "All results in: $OUTDIR"
