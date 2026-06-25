#!/usr/bin/env bash
# campaigns/edpp-study/sweep.sh — bake one trace per (workload,rate), replay across deciders.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (inference-sim)

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
NINST=4
NPREFILL=2
NDECODE=2
OUT="campaigns/edpp-study/out"; TRACES="$OUT/traces"
mkdir -p "$OUT" "$TRACES"

WORKLOADS=("rag" "synth")
RATES=(0.5 1.0 1.5 2.0 2.5 3.0)

for wl in "${WORKLOADS[@]}"; do
  for r in "${RATES[@]}"; do
    spec="campaigns/edpp-study/specs/${wl}_rate${r}.yaml"
    trprefix="$TRACES/${wl}_rate${r}"
    th="${trprefix}.yaml"; td="${trprefix}.csv"
    echo "== bake $wl rate $r =="
    ./blis run --model "$MODEL" --workload-spec "$spec" \
      --num-instances "$NINST" --trace-output "$trprefix"

    # Replay each decider against the identical trace.
    # edpp, prefix-threshold, always require --prefill-instances/--decode-instances.
    # never uses collocated mode (no split).
    replay() { # $1=label  $2... extra flags
      local label="$1"; shift
      echo "-- replay $wl rate $r [$label] --"
      ./blis replay --model "$MODEL" --trace-header "$th" --trace-data "$td" \
        --num-instances "$NINST" \
        --results-path "$OUT/results_${wl}_rate${r}_${label}.json" "$@"
    }
    replay edpp \
      --prefill-instances "$NPREFILL" --decode-instances "$NDECODE" \
      --pd-decider edpp --edpp-coeffs "$COEFFS" \
      --trace-level decisions \
      --edpp-decision-trace "$OUT/decisions_${wl}_rate${r}_edpp.csv"
    replay prefix-threshold \
      --prefill-instances "$NPREFILL" --decode-instances "$NDECODE" \
      --pd-decider prefix-threshold
    replay never  --pd-decider never
    replay always \
      --prefill-instances "$NPREFILL" --decode-instances "$NDECODE" \
      --pd-decider always
  done
done
echo "DONE -> $OUT"
