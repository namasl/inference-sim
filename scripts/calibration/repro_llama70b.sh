#!/usr/bin/env bash
# Reproduce the frozen EDPP latency-law coefficients
# (scripts/calibration/coeffs-llama70b-h100-tp4.json) from scratch.
#
# Provenance note: the committed coeffs file is named for its target
# (Llama-70B / H100 / TP4) and its source_csvs point at a since-deleted
# /tmp/llama70/*.csv set. The calibration README documents the *procedure*
# using qwen3-14b as a worked example; this script is the exact Llama-70B
# instantiation of that procedure. Running it regenerates every coefficient
# BIT-EXACTLY (calibration runs are deterministic), which is the trust-check.
#
# Usage:  bash scripts/calibration/repro_llama70b.sh [OUTDIR]
# Requires: ./blis built (go build -o blis main.go), python3 + numpy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # inference-sim/
cd "$HERE"
OUT="${1:-/tmp/llama70_repro}"
mkdir -p "$OUT"
M="meta-llama/llama-3.3-70b-instruct"   # -> H100, TP4 by defaults.yaml
HW=(--hardware H100 --tp 4)             # explicit; matches the frozen target
BLIS=./blis
[ -x "$BLIS" ] || { echo "build blis first: go build -o blis main.go" >&2; exit 1; }

echo "== decode runs (fit alpha, c0, c1) — spread (b_dec, kv) in 2-D =="
BLIS_STEP_CSV="$OUT/D1.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 150 --rate 6  --prompt-tokens 128  --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null
BLIS_STEP_CSV="$OUT/D2.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 150 --rate 24 --prompt-tokens 128  --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null
BLIS_STEP_CSV="$OUT/D3.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 150 --rate 24 --prompt-tokens 2000 --output-tokens 2000 --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null
BLIS_STEP_CSV="$OUT/D4.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 150 --rate 6  --prompt-tokens 2000 --output-tokens 500  --max-num-scheduled-tokens 8192 --max-model-len 8192 >/dev/null

echo "== prefill runs (fit alpha_p, c_pf, c_attn) — vary chunk AND prompt length =="
BLIS_STEP_CSV="$OUT/P1.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 120 --rate 3 --prompt-tokens 1000  --max-num-scheduled-tokens 512  --max-model-len 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4 >/dev/null
BLIS_STEP_CSV="$OUT/P2.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 120 --rate 3 --prompt-tokens 4000  --max-num-scheduled-tokens 2048 --max-model-len 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4 >/dev/null
BLIS_STEP_CSV="$OUT/P3.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 120 --rate 3 --prompt-tokens 16000 --max-num-scheduled-tokens 4096 --max-model-len 20000 --prompt-tokens-max 20000 --output-tokens 2 --output-tokens-stdev 0 --output-tokens-min 1 --output-tokens-max 4 >/dev/null

echo "== mixed runs (validate additivity across regimes) =="
BLIS_STEP_CSV="$OUT/M_decode.csv"  $BLIS run --model "$M" "${HW[@]}" --num-requests 250 --rate 22 --prompt-tokens 256  --output-tokens 1500 --max-num-scheduled-tokens 256 >/dev/null
BLIS_STEP_CSV="$OUT/M_ridge.csv"   $BLIS run --model "$M" "${HW[@]}" --num-requests 250 --rate 12 --prompt-tokens 1024 --output-tokens 512  --max-num-scheduled-tokens 1024 >/dev/null
BLIS_STEP_CSV="$OUT/M_prefill.csv" $BLIS run --model "$M" "${HW[@]}" --num-requests 200 --rate 5  --prompt-tokens 8000 --output-tokens 32 --output-tokens-stdev 8 --output-tokens-min 4 --output-tokens-max 64 --prompt-tokens-max 20000 --max-num-scheduled-tokens 8192 --max-model-len 12000 >/dev/null

echo "== fit + validate =="
python3 scripts/calibration/fit_coeffs.py "$OUT"/D*.csv "$OUT"/P*.csv \
  -o "$OUT/coeffs_repro.json" \
  --validate "$OUT/M_decode.csv" "$OUT/M_ridge.csv" "$OUT/M_prefill.csv" >/dev/null

echo "== diff reproduced vs frozen =="
python3 - "$OUT/coeffs_repro.json" scripts/calibration/coeffs-llama70b-h100-tp4.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); f = json.load(open(sys.argv[2]))
keys = [("decode","alpha_us"),("decode","c0_us_per_req"),("decode","c1_us_per_token"),
        ("prefill","alpha_p_us"),("prefill","c_pf_us_per_token"),("prefill","c_attn_us_per_unit")]
ok = True
print(f'{"coeff":22} {"reproduced":>18} {"frozen":>18} {"rel_err_%":>12}')
for half, k in keys:
    rv, fv = r[half][k], f[half][k]
    rel = 100*(rv-fv)/fv if fv else 0.0
    if abs(rel) > 1e-6: ok = False
    print(f'{half+"."+k:22} {rv:18.8g} {fv:18.8g} {rel:12.4g}')
print("\nCHECKPOINT:", "PASS — bit-exact reproduction" if ok else "FAIL — coeffs drifted (see rel_err_%)")
sys.exit(0 if ok else 1)
PY
