#!/usr/bin/env bash
# Extract per-GPU EDPP latency-law coefficients (θ_i) for an arbitrary device,
# fit from the simulator's OWN trained-physics execution at that HWConfig.
#
# This generalizes repro_llama70b.sh (the H100/TP4 instantiation of the
# calibration procedure in scripts/calibration/README.md) to any device whose
# roofline is pinned via a single-pool `hw_config_by_gpu` policy bundle. Because
# the decider consumes these coefficients as its θ_i for that device, fitting
# them from the same trained-physics engine that will EXECUTE the device removes
# the model/hardware confound (design §3).
#
# ROUTE: CSV-tap (Step 2a). The step recorder (BLIS_STEP_CSV) taps correctly
# under a single-instance node-pool bundle that pins the target HWConfig, so we
# drive the real `blis run` calibration grid rather than the Go-sampler fallback.
#
# Usage:
#   bash scripts/calibration/repro_theta_by_gpu.sh LABEL TFLOPS BW [GPU_TYPE] [OUTDIR]
#
# Args:
#   LABEL     device label used in the output filename (e.g. a100crippled)
#   TFLOPS    tflops_peak for the target HWConfig
#   BW        bw_peak_tbs for the target HWConfig
#   GPU_TYPE  node-pool gpu_type key (default: A100)
#   OUTDIR    scratch dir for the calibration CSVs (default: /tmp/theta_<LABEL>)
#
# Output: scripts/calibration/coeffs-llama70b-<LABEL>-tp4.json
#
# Requires: ./blis built (go build -o blis main.go), python3 + numpy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # inference-sim/
cd "$HERE"

LABEL="${1:?usage: repro_theta_by_gpu.sh LABEL TFLOPS BW [GPU_TYPE] [OUTDIR]}"
TFLOPS="${2:?missing TFLOPS (tflops_peak)}"
BW="${3:?missing BW (bw_peak_tbs)}"
GPU_TYPE="${4:-A100}"
OUT="${5:-/tmp/theta_$LABEL}"
mkdir -p "$OUT"

M="meta-llama/llama-3.3-70b-instruct"   # same model as the H100 frozen file
BLIS=./blis
[ -x "$BLIS" ] || { echo "build blis first: go build -o blis main.go" >&2; exit 1; }

# Single-pool bundle pinning the target device's roofline via hw_config_by_gpu.
# TP defaults to 4, so the 4-GPU node holds exactly one instance.
BUNDLE="$OUT/bundle.yaml"
cat > "$BUNDLE" <<EOF
node_pools:
  - name: only
    gpu_type: $GPU_TYPE
    gpus_per_node: 4
    gpu_memory_gib: 80
    initial_nodes: 1
    min_nodes: 1
    max_nodes: 1
hw_config_by_gpu:
  $GPU_TYPE:
    tflops_peak: $TFLOPS
    bw_peak_tbs: $BW
    mfu_prefill: 0.5
    mfu_decode: 0.5
EOF

# Same flags as repro_llama70b.sh / the README, plus the single-pool bundle.
PIN=(--num-instances 1 --policy-config "$BUNDLE")

echo "== target: $LABEL  gpu=$GPU_TYPE  tflops_peak=$TFLOPS  bw_peak_tbs=$BW =="
echo "== decode runs (fit alpha, c0, c1) — spread (b_dec, kv) in 2-D =="
BLIS_STEP_CSV="$OUT/D1.csv" $BLIS run --model "$M" "${PIN[@]}" --num-requests 150 --rate 6  --prompt-tokens 128  --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null
BLIS_STEP_CSV="$OUT/D2.csv" $BLIS run --model "$M" "${PIN[@]}" --num-requests 150 --rate 24 --prompt-tokens 128  --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null
BLIS_STEP_CSV="$OUT/D3.csv" $BLIS run --model "$M" "${PIN[@]}" --num-requests 150 --rate 24 --prompt-tokens 2000 --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null
BLIS_STEP_CSV="$OUT/D4.csv" $BLIS run --model "$M" "${PIN[@]}" --num-requests 150 --rate 6  --prompt-tokens 2000 --output-tokens 500  --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null

echo "== prefill runs (fit alpha_p, c_pf, c_attn) — vary chunk AND prompt length =="
BLIS_STEP_CSV="$OUT/P1.csv" $BLIS run --model "$M" "${PIN[@]}" --num-requests 120 --rate 3 --prompt-tokens 1000  --max-num-scheduled-tokens 512  --max-model-len 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4 >/dev/null
BLIS_STEP_CSV="$OUT/P2.csv" $BLIS run --model "$M" "${PIN[@]}" --num-requests 120 --rate 3 --prompt-tokens 4000  --max-num-scheduled-tokens 2048 --max-model-len 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4 >/dev/null
BLIS_STEP_CSV="$OUT/P3.csv" $BLIS run --model "$M" "${PIN[@]}" --num-requests 120 --rate 3 --prompt-tokens 16000 --max-num-scheduled-tokens 4096 --max-model-len 20000 --prompt-tokens-max 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4 >/dev/null

OUTFILE="scripts/calibration/coeffs-llama70b-$LABEL-tp4.json"
echo "== fit -> $OUTFILE =="
python3 scripts/calibration/fit_coeffs.py "$OUT"/D*.csv "$OUT"/P*.csv -o "$OUTFILE" >/dev/null

echo "== fit quality (R2 should be ~1, cond_* well under 30) =="
python3 - "$OUTFILE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
d, p = c["decode"], c["prefill"]
print(f"  decode : n={d['n_rows']:6d}  R2={d['r2']:.8f}  cond_b_dec_kv={d['cond_b_dec_kv']:.4f}")
print(f"  prefill: n={p['n_rows']:6d}  R2={p['r2']:.8f}  cond_s_pf_pf_ctx={p['cond_s_pf_pf_ctx']:.4f}")
for half, k in (("decode", "WARNING"), ("prefill", "WARNING")):
    if k in c[half]:
        print(f"  [{half}] {c[half][k]}")
PY
