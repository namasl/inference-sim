#!/usr/bin/env bash
# Reproduce the Stage B work-model validation (see FINDINGS.md "Stage B").
#
# Validates that the corrected EDPP work model (Wp = C_pf·a_p + C_attn·a_p·(a_r+a_p/2),
# the trained-physics basis; Wd = exact discrete decode sum) equals the per-request
# realized trajectory work the active latency model charges. Runs BOTH synth
# (no shared prefix) and rag (shared prefix) at 2P2D under EDPP.
#
# Usage (from inference-sim/ repo root):
#   bash campaigns/edpp-study/repro_stage_b.sh
#
# Outputs under campaigns/edpp-study/out/stage_b/ (out/ gitignored):
#   <wl>_work.csv   --edpp-work-trace (realized vs closed-form work, per request)
#   <wl>_bias.json  work_model_validation.py report
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
OUT="campaigns/edpp-study/out/stage_b"
mkdir -p "$OUT"

if [[ ! -x ./blis ]]; then
  echo "building blis..." >&2
  go build -o blis main.go
fi

# Per-workload SLO/tau flags (match the study harness sweep.sh).
slo_flags() {
  case "$1" in
    synth) echo --slo-ttft "batch=2s" --slo-itl "batch=150ms" --edpp-tau-ttft 2s --edpp-tau-itl 150ms ;;
    rag)   echo --slo-ttft "standard=500ms,batch=5s" --slo-itl "standard=150ms,batch=200ms" \
                --edpp-tau-ttft-classes "standard=500ms,batch=5s" --edpp-tau-itl-classes "standard=150ms,batch=200ms" ;;
  esac
}

run_wl() {
  local wl="$1" spec="campaigns/edpp-study/specs/${1}_rate2.0.yaml"
  echo "[$wl 1/3] baking -> $OUT/${wl}.{yaml,csv}" >&2
  ./blis run --model "$MODEL" --workload-spec "$spec" \
    --num-instances 4 --trace-output "$OUT/${wl}"

  echo "[$wl 2/3] replaying 2P2D edpp -> $OUT/${wl}_work.csv" >&2
  # shellcheck disable=SC2046
  ./blis replay --model "$MODEL" \
    --trace-header "$OUT/${wl}.yaml" --trace-data "$OUT/${wl}.csv" \
    --num-instances 4 --prefill-instances 2 --decode-instances 2 \
    --pd-decider edpp --edpp-coeffs "$COEFFS" \
    $(slo_flags "$wl") \
    --edpp-work-trace "$OUT/${wl}_work.csv" \
    >/dev/null

  echo "[$wl 3/3] analyzing -> $OUT/${wl}_bias.json" >&2
  python3 campaigns/edpp-study/analyze/work_model_validation.py \
    --work "$OUT/${wl}_work.csv" --out "$OUT/${wl}_bias.json"
}

for wl in synth rag; do
  run_wl "$wl"
done
echo "done. reports: $OUT/synth_bias.json, $OUT/rag_bias.json" >&2
