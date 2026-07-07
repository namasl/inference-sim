#!/usr/bin/env bash
# Reproduce the decode-pool utilization sweep (see FINDINGS.md "Utilization sweep").
#
# Hardens the single-point Stage C result into a curve: measure decode-pool
# admission-estimator bias across bounded, stationary operating points below
# capacity. Topology = Stage C T1 (1P1D, EDPP, huge transfer penalty ⇒ all-local
# on the single decode engine). Steps:
#   1. Locate λ* — coarse-scan aggregate_rate until the composite saturation
#      detector leaves STABLE. λ* = lowest non-STABLE rate.
#   2. Sweep ρ = {0.5,0.7,0.85,0.9,0.95,0.98} → offered rate ρ·λ*; per point emit
#      --edpp-admission-trace + --post-hoc-detector composite; run admission_ablation.py.
#   3. Aggregate → sweep.json (+ sweep.png): bias vs measured ρ̂, drift, verdict.
#
# Usage (from inference-sim/ repo root):  bash campaigns/edpp-study/repro_utilization_sweep.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
BASE_SPEC="campaigns/edpp-study/specs/synth_rate1.0.yaml"
OUT="campaigns/edpp-study/out/utilization_sweep"; mkdir -p "$OUT"
ANALYZE="campaigns/edpp-study/analyze"
SCAN_RATES=(0.1 0.25 0.5 0.75 1.0 1.5 2.0)
RHO_TARGETS=(0.5 0.7 0.85 0.9 0.95 0.98)

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi

T1_COMMON=(--model "$MODEL"
  --num-instances 2 --prefill-instances 1 --decode-instances 1
  --pd-decider edpp --edpp-coeffs "$COEFFS"
  --edpp-tau-ttft 2s --edpp-tau-itl 150ms --slo-ttft "batch=2s" --slo-itl "batch=150ms"
  --edpp-c-xfer 100s --post-hoc-detector composite)

# spec_for_rate <rate> -> path to a temp spec with aggregate_rate rewritten
spec_for_rate() {
  local rate="$1"; local dst="$OUT/spec_rate${rate}.yaml"
  sed -E "s/^aggregate_rate:.*/aggregate_rate: ${rate}/" "$BASE_SPEC" > "$dst"
  echo "$dst"
}

echo "[1/3] locating λ* by coarse scan..." >&2
LAMBDA_STAR=""
for rate in "${SCAN_RATES[@]}"; do
  spec="$(spec_for_rate "$rate")"
  m="$OUT/scan_rate${rate}.metrics.json"
  # --metrics-path writes the clean cluster-level MetricsOutput JSON the analyzer
  # consumes; stdout is human-decorated (=== headers + per-instance blocks) and
  # not directly JSON-parseable, so it goes to /dev/null.
  ./blis run "${T1_COMMON[@]}" --workload-spec "$spec" --metrics-path "$m" >/dev/null 2>/dev/null
  if python3 "$ANALYZE/utilization_sweep.py" scan-verdict --metrics "$m" >/dev/null; then
    echo "   rate $rate: STABLE" >&2
  else
    echo "   rate $rate: NON-STABLE -> λ* = $rate" >&2
    LAMBDA_STAR="$rate"; break
  fi
done
if [[ -z "$LAMBDA_STAR" ]]; then
  echo "ERROR: all scan rates STABLE; widen SCAN_RATES upward and re-run." >&2; exit 1
fi
echo "$LAMBDA_STAR" > "$OUT/lambda_star.txt"

echo "[2/3] sweeping ρ grid at λ* = $LAMBDA_STAR ..." >&2
for rho in "${RHO_TARGETS[@]}"; do
  rate="$(python3 -c "print(round(${rho}*${LAMBDA_STAR}, 6))")"
  spec="$(spec_for_rate "$rate")"
  echo "   ρ=$rho -> rate=$rate" >&2
  ./blis run "${T1_COMMON[@]}" --workload-spec "$spec" \
    --edpp-admission-trace "$OUT/pt_rate${rate}.admission.csv" \
    --metrics-path "$OUT/pt_rate${rate}.metrics.json" >/dev/null 2>/dev/null
  python3 "$ANALYZE/admission_ablation.py" --admission "$OUT/pt_rate${rate}.admission.csv" \
    --out "$OUT/pt_rate${rate}.ablation.json" >/dev/null
done

echo "[3/3] aggregating..." >&2
python3 "$ANALYZE/utilization_sweep.py" aggregate --sweep-dir "$OUT" \
  --lambda-star "$LAMBDA_STAR" --pool local \
  --out "$OUT/sweep.json" --plots "$OUT/sweep.png" >/dev/null
echo "done. λ*=$LAMBDA_STAR ; report: $OUT/sweep.json ; figure: $OUT/sweep.png" >&2
