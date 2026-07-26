#!/usr/bin/env bash
# ttft_p estimator study (single 1P1D disaggregated pipeline) — Phase 1.
#
# Mirror of repro_ttft_d_local.sh but for the DISAGGREGATED path: 1P1D with
# --edpp-c-xfer 0s ⇒ EDPP disaggregates (nearly) every request onto the single prefill +
# single decode engine (no pool/routing confound). Validates the disaggregated TTFT
# estimate ttft_p = prefill-pool admission + prefill compute + KV transfer + decode-side
# first-token, against realized outcomes, for fluid & rollforward, across load
# (ρ = offered/λ*, λ* = plateau throughput probe) and the three archetypes
# synth (decode-heavy) / mixed (balanced) / prefill (prefill-heavy).
#
# --pd-decider edpp is REQUIRED (only EDPP's Decide computes/logs ttft_p; never/always
# don't). Execution is identical across estimators here too, so per point we run fluid
# (capturing outcome + admission + decision) and rollforward (decision only).
#
# Downstream ttft_p_local.py emits two figures (decomposition option b):
#   fig_ttft_p     : realized_ttft vs estimated ttft_p (total, the bottom line)
#   fig_prefill_adm: realized prefill_t_adm vs estimated prefill-pool t_adm (load-sensitive)
#
# Usage (from inference-sim/ repo root):  bash campaigns/edpp-study/repro_ttft_p_local.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
OUT="campaigns/edpp-study/out/ttft_p_local"; mkdir -p "$OUT"
ANALYZE="campaigns/edpp-study/analyze"
WORKLOADS=(synth mixed prefill)
# (macOS bash 3.2: functions, not associative arrays.)
probe_rate() { case "$1" in synth) echo 2.5 ;; mixed) echo 6.0 ;; prefill) echo 8.0 ;; *) echo 3.0 ;; esac; }
num_req()    { echo 3000; }
RHO_TARGETS=(0.5 0.75 1.0 1.25 1.5)

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi

COMMON=(--model "$MODEL"
  --num-instances 2 --prefill-instances 1 --decode-instances 1
  --pd-decider edpp --edpp-coeffs "$COEFFS" --edpp-c-xfer 0s
  --post-hoc-detector composite --trace-level decisions)

slo_flags() { echo --slo-ttft "batch=2s" --slo-itl "batch=150ms" --edpp-tau-ttft 2s --edpp-tau-itl 150ms; }

spec_for_rate() {  # <wl> <rate> <dir>
  local wl="$1" rate="$2" dir="$3" base="campaigns/edpp-study/specs/${1}_rate1.0.yaml" nreq
  nreq="$(num_req "$wl")"; local dst="$dir/spec_rate${rate}.yaml"
  sed -E -e "s/^aggregate_rate:.*/aggregate_rate: ${rate}/" \
         -e "s/^num_requests:.*/num_requests: ${nreq}/" "$base" > "$dst"; echo "$dst"
}

sweep_wl() {
  local wl="$1"; local dir="$OUT/$wl"; mkdir -p "$dir"
  # shellcheck disable=SC2046
  local slo; slo=($(slo_flags))

  local probe; probe="$(probe_rate "$wl")"
  echo "[$wl 1/2] probing λ* at overload rate $probe ..." >&2
  local pspec pm; pspec="$(spec_for_rate "$wl" "$probe" "$dir")"; pm="$dir/probe.metrics.json"
  ./blis run "${COMMON[@]}" "${slo[@]}" --workload-spec "$pspec" --metrics-path "$pm" >/dev/null 2>/dev/null
  local lam; lam="$(python3 -c "
import json
m=json.load(open('$pm')); m=m if isinstance(m,dict) else (m[0] if m else {})
print(f\"{float(m.get('responses_per_sec',0.0)):.4f}\")
")"
  python3 -c "import sys; sys.exit(0 if $lam > 0 else 1)" || { echo "ERROR($wl): probe λ*=0." >&2; exit 1; }
  echo "   -> λ* (plateau throughput) = $lam req/s" >&2
  echo "$lam" > "$dir/lambda_star.txt"

  echo "[$wl 2/2] sweeping ρ grid (incl. overload) at λ*=$lam ..." >&2
  for rho in "${RHO_TARGETS[@]}"; do
    local rate spec tag; rate="$(python3 -c "print(round(${rho}*${lam}, 6))")"
    spec="$(spec_for_rate "$wl" "$rate" "$dir")"; tag="$dir/rho${rho}"
    echo "   ρ=$rho -> rate=$rate" >&2
    ./blis run "${COMMON[@]}" "${slo[@]}" --edpp-tadm-estimator fluid --workload-spec "$spec" \
      --pd-outcome-trace "${tag}.outcome.csv" \
      --edpp-admission-trace "${tag}.admission.csv" \
      --edpp-decision-trace "${tag}.decision.fluid.csv" \
      --metrics-path "${tag}.metrics.json" >/dev/null 2>/dev/null
    ./blis run "${COMMON[@]}" "${slo[@]}" --edpp-tadm-estimator rollforward --workload-spec "$spec" \
      --edpp-decision-trace "${tag}.decision.rollforward.csv" >/dev/null 2>/dev/null
  done
}

for wl in "${WORKLOADS[@]}"; do
  sweep_wl "$wl"
done

echo "plotting -> $OUT/fig_{ttft_p,prefill_adm}.png" >&2
python3 "$ANALYZE/ttft_p_local.py" --sweep-root "$OUT" --rho "${RHO_TARGETS[*]}" --out-prefix "$OUT/fig"
echo "done. figures: $OUT/fig_ttft_p.png $OUT/fig_prefill_adm.png" >&2
