#!/usr/bin/env bash
# campaigns/edpp-study/sweep.sh — bake one trace per (workload,rate), replay across
# the never@4 baseline plus the disaggregating arms over P/D splits.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (inference-sim)

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
NINST=4
OUT="campaigns/edpp-study/out"; TRACES="$OUT/traces"
mkdir -p "$OUT" "$TRACES"

WORKLOADS=("rag" "synth")
RATES=(0.5 1.0 1.5 2.0 2.5 3.0)
SPLITS=("1 3" "2 2" "3 1")          # P D, summing to NINST=4

for wl in "${WORKLOADS[@]}"; do
  for r in "${RATES[@]}"; do
    spec="campaigns/edpp-study/specs/${wl}_rate${r}.yaml"
    trprefix="$TRACES/${wl}_rate${r}"
    th="${trprefix}.yaml"; td="${trprefix}.csv"
    echo "== bake $wl rate $r =="
    ./blis run --model "$MODEL" --workload-spec "$spec" \
      --num-instances "$NINST" --trace-output "$trprefix"

    base_replay() { # $1=suffix  $2... extra flags
      local sfx="$1"; shift
      echo "-- replay $wl rate $r [$sfx] --"
      ./blis replay --model "$MODEL" --trace-header "$th" --trace-data "$td" \
        --num-instances "$NINST" \
        --results-path "$OUT/results_${wl}_rate${r}_${sfx}.json" "$@"
    }

    # never / all-local baseline: homogeneous 4 collocated (no P/D split).
    base_replay "never_agg" --pd-decider never

    # disaggregating arms swept over P/D splits.
    for split in "${SPLITS[@]}"; do
      set -- $split; P="$1"; D="$2"; tag="${P}P${D}D"
      base_replay "edpp_${tag}" \
        --prefill-instances "$P" --decode-instances "$D" \
        --pd-decider edpp --edpp-coeffs "$COEFFS" \
        --trace-level decisions \
        --edpp-decision-trace "$OUT/decisions_${wl}_rate${r}_edpp_${tag}.csv"
      base_replay "always_${tag}" \
        --prefill-instances "$P" --decode-instances "$D" --pd-decider always
      base_replay "prefix-threshold_${tag}" \
        --prefill-instances "$P" --decode-instances "$D" --pd-decider prefix-threshold
    done
  done
done
echo "DONE -> $OUT"
