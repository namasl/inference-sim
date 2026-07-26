#!/usr/bin/env bash
# Fig 1 (work model) — sweep the Stage B work-model validation across offered load.
#
# Stage B (repro_stage_b.sh) validates the corrected EDPP work model at one rate.
# This sweep repeats that validation across rates 0.5..3.0 for each workload so we
# can plot work-model relative error vs load. Expected story:
#   - single-chunk prefill + decode work: float-exact at EVERY load (per-request
#     closed form; load-invariant).
#   - chunked-prefill residual (C_attn*(a_p^2 - sum s_r^2)/2): grows with load
#     because higher load -> more chunking -> larger residual, but stays bounded.
#
# Usage (from inference-sim/ repo root):
#   bash campaigns/edpp-study/repro_work_model_sweep.sh
#
# Outputs under campaigns/edpp-study/out/work_sweep/ (out/ gitignored):
#   <wl>_rate<r>_work.csv   --edpp-work-trace (realized vs closed-form work)
#   <wl>_rate<r>_bias.json  work_model_validation.py report
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
OUT="campaigns/edpp-study/out/work_sweep"
RATES=(0.5 1.0 1.5 2.0 2.5 3.0)
mkdir -p "$OUT"

if [[ ! -x ./blis ]]; then
  echo "building blis..." >&2
  go build -o blis main.go
fi

# Per-workload SLO/tau flags (match repro_stage_b.sh / sweep.sh).
slo_flags() {
  case "$1" in
    synth) echo --slo-ttft "batch=2s" --slo-itl "batch=150ms" --edpp-tau-ttft 2s --edpp-tau-itl 150ms ;;
    rag)   echo --slo-ttft "standard=500ms,batch=5s" --slo-itl "standard=150ms,batch=200ms" \
                --edpp-tau-ttft-classes "standard=500ms,batch=5s" --edpp-tau-itl-classes "standard=150ms,batch=200ms" ;;
  esac
}

run_point() {
  local wl="$1" rate="$2"
  local spec="campaigns/edpp-study/specs/${wl}_rate${rate}.yaml"
  local tag="${wl}_rate${rate}"
  if [[ ! -f "$spec" ]]; then
    echo "skip: missing spec $spec" >&2
    return
  fi

  echo "[$tag 1/3] baking -> $OUT/${tag}.{yaml,csv}" >&2
  ./blis run --model "$MODEL" --workload-spec "$spec" \
    --num-instances 4 --trace-output "$OUT/${tag}"

  echo "[$tag 2/3] replaying 2P2D edpp -> $OUT/${tag}_work.csv" >&2
  # shellcheck disable=SC2046
  ./blis replay --model "$MODEL" \
    --trace-header "$OUT/${tag}.yaml" --trace-data "$OUT/${tag}.csv" \
    --num-instances 4 --prefill-instances 2 --decode-instances 2 \
    --pd-decider edpp --edpp-coeffs "$COEFFS" \
    $(slo_flags "$wl") \
    --edpp-work-trace "$OUT/${tag}_work.csv" \
    >/dev/null

  echo "[$tag 3/3] analyzing -> $OUT/${tag}_bias.json" >&2
  python3 campaigns/edpp-study/analyze/work_model_validation.py \
    --work "$OUT/${tag}_work.csv" --out "$OUT/${tag}_bias.json"
}

for wl in synth rag; do
  for r in "${RATES[@]}"; do
    run_point "$wl" "$r"
  done
done
echo "done. reports in $OUT/*_bias.json" >&2
echo "plot with: python3 campaigns/edpp-study/analyze/work_model_sweep.py --sweep-dir $OUT --plots $OUT/fig1_work_model.png" >&2
