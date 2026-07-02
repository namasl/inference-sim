#!/usr/bin/env bash
# Reproduce the Stage A estimator-validation anchor (see FINDINGS.md "Stage A").
#
# Measures the bias of the SHIPPED (waiting-only, occupancy-blind) EDPP forward
# TTFT/admission estimators against realized outcomes, on synth @ 2P2D rate 2.0.
# Measurement-only: does not change the decider or routing.
#
# Usage (from the inference-sim/ repo root):
#   ./blis ...            # build first: go build -o blis main.go
#   bash campaigns/edpp-study/repro_stage_a.sh
#
# Outputs land in campaigns/edpp-study/out/stage_a/ (out/ is gitignored):
#   synth2.0.{yaml,csv}  baked trace (num-instances 4, topology-independent request stream)
#   decisions.csv        --edpp-decision-trace (forward estimates, per decision)
#   outcome.csv          --pd-outcome-trace   (realized T_adm/TTFT/ITL/E2E, per request)
#   bias.json            estimator_validation.py report (predicted vs realized)
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
SPEC="campaigns/edpp-study/specs/synth_rate2.0.yaml"   # 5000 reqs, aggregate_rate 2.0
OUT="campaigns/edpp-study/out/stage_a"
mkdir -p "$OUT"

if [[ ! -x ./blis ]]; then
  echo "building blis..." >&2
  go build -o blis main.go
fi

# 1. Bake the request stream once at num-instances 4 (topology-independent).
echo "[1/3] baking trace -> $OUT/synth2.0.{yaml,csv}" >&2
./blis run --model "$MODEL" --workload-spec "$SPEC" \
  --num-instances 4 --trace-output "$OUT/synth2.0"

# 2. Replay at 2P2D under EDPP, emitting BOTH traces.
#    --pd-outcome-trace needs no --trace-level; --edpp-decision-trace needs --trace-level decisions.
#    EDPP decisions happen in replay; per-class SLO/tau match the study harness (synth batch class).
echo "[2/3] replaying 2P2D edpp -> $OUT/{decisions,outcome}.csv" >&2
./blis replay --model "$MODEL" \
  --trace-header "$OUT/synth2.0.yaml" --trace-data "$OUT/synth2.0.csv" \
  --num-instances 4 --prefill-instances 2 --decode-instances 2 \
  --pd-decider edpp --edpp-coeffs "$COEFFS" \
  --edpp-tau-ttft 2s --edpp-tau-itl 150ms \
  --slo-ttft "batch=2s" --slo-itl "batch=150ms" \
  --trace-level decisions \
  --edpp-decision-trace "$OUT/decisions.csv" \
  --pd-outcome-trace "$OUT/outcome.csv" \
  >/dev/null

# 3. Join and report bias (JSON to stdout + file). Add --plots <png> for a scatter.
echo "[3/3] analyzing -> $OUT/bias.json" >&2
python3 campaigns/edpp-study/analyze/estimator_validation.py \
  --outcome "$OUT/outcome.csv" --decision "$OUT/decisions.csv" \
  --out "$OUT/bias.json"

echo "done. report: $OUT/bias.json" >&2
