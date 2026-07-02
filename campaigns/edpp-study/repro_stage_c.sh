#!/usr/bin/env bash
# Reproduce the Stage C admission-estimator fidelity microbenchmark (see FINDINGS.md "Stage C").
#
# Two forced-routing isolation topologies on a single engine per role, so the
# predicted-vs-realized admission-delay gap is purely estimator quality:
#   T1 (ttft_d isolation): 1P1D, EDPP with a HUGE transfer penalty (--edpp-c-xfer 100s)
#      → EDPP keeps (nearly) everything LOCAL on the single decode engine.
#   T2 (ttft_p + ttft_d):  1P1D, EDPP with ZERO transfer penalty (--edpp-c-xfer 0s)
#      → EDPP disaggregates (nearly) everything → single prefill + single decode engine.
# NB: --pd-decider edpp is required (only EDPP's Decide assembles the estimator contexts);
# `never`/`always` deciders do not run it. The c-xfer knob forces the routing while
# EDPP still runs, so all six estimator predictions are logged per request×pool.
#
# Usage (from inference-sim/ repo root):
#   bash campaigns/edpp-study/repro_stage_c.sh
# Outputs under campaigns/edpp-study/out/stage_c/ (out/ gitignored):
#   t1_admission.csv / t2_admission.csv  (--edpp-admission-trace: realized + 6 predictions per request×pool)
#   t1_ablation.json / t2_ablation.json  (admission_ablation.py bias report)
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
SPEC="campaigns/edpp-study/specs/synth_rate2.0.yaml"  # saturating on a single decode engine
OUT="campaigns/edpp-study/out/stage_c"
mkdir -p "$OUT"

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi

COMMON=(--model "$MODEL" --workload-spec "$SPEC"
  --num-instances 2 --prefill-instances 1 --decode-instances 1
  --pd-decider edpp --edpp-coeffs "$COEFFS"
  --edpp-tau-ttft 2s --edpp-tau-itl 150ms --slo-ttft "batch=2s" --slo-itl "batch=150ms")

echo "[T1] force-local (c-xfer 100s) -> $OUT/t1_admission.csv" >&2
./blis run "${COMMON[@]}" --edpp-c-xfer 100s  --edpp-admission-trace "$OUT/t1_admission.csv" >/dev/null 2>&1
python3 campaigns/edpp-study/analyze/admission_ablation.py --admission "$OUT/t1_admission.csv" --out "$OUT/t1_ablation.json" >/dev/null

echo "[T2] force-disagg (c-xfer 0s) -> $OUT/t2_admission.csv" >&2
./blis run "${COMMON[@]}" --edpp-c-xfer 0s    --edpp-admission-trace "$OUT/t2_admission.csv" >/dev/null 2>&1
python3 campaigns/edpp-study/analyze/admission_ablation.py --admission "$OUT/t2_admission.csv" --out "$OUT/t2_ablation.json" >/dev/null

echo "done. reports: $OUT/t1_ablation.json (ttft_d), $OUT/t2_ablation.json (ttft_p+ttft_d)" >&2
