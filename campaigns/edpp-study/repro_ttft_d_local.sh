#!/usr/bin/env bash
# ttft_d local estimator study (single collocated instance) — 3-figure decomposition.
#
# On ONE collocated engine (1P1D, huge transfer penalty ⇒ every request local, no routing
# confound) we validate the LOCAL time-to-first-token estimate ttft_d = T_adm + prefill_time
# against realized outcomes, for the two deployable estimators fluid & rollforward, across
# load (sub-load → overload) and three workload archetypes:
#   synth = decode-heavy (prefill_decode_ratio 0.125)
#   rag   = prefill-heavy (ratio 30)
#   mixed = balanced (fabricated for this study; specs/mixed_rate1.0.yaml)
#
# Execution is IDENTICAL across estimators under forced-local (the transfer penalty, not the
# estimate, decides local), so per load point we run fluid (capturing outcome + decision +
# admission traces) and rollforward (decision trace only — outcome/admission are identical).
#
# Downstream ttft_d_local.py joins the traces per request and emits three figures:
#   Fig A admission delay, Fig B prefill time, Fig C ttft (each = realized vs fluid/rollforward).
#
# Usage (from inference-sim/ repo root):  bash campaigns/edpp-study/repro_ttft_d_local.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="meta-llama/llama-3.3-70b-instruct"
COEFFS="scripts/calibration/coeffs-llama70b-h100-tp4.json"
OUT="campaigns/edpp-study/out/ttft_d_local"; mkdir -p "$OUT"
ANALYZE="campaigns/edpp-study/analyze"
# Three tractable single-instance archetypes spanning the prefill/decode spectrum:
#   synth = decode-heavy, mixed = balanced, prefill = prefill-heavy (synthetic, sane prompt
#   sizes). Real rag (15k–80k-tok prompts) is intractable to sweep on one instance.
WORKLOADS=(synth mixed prefill)
# λ* = PLATEAU THROUGHPUT (max sustainable rps), measured with one deliberately-overloaded
# probe run per workload: at an offered rate well above capacity, completions saturate at μ.
# This is warmup-insensitive (unlike a throughput-ratio threshold) and gives a clean μ to
# normalize against. Probe rates are per-workload (rag's huge prompts saturate at low req/s).
# (macOS bash 3.2 has no associative arrays — use functions.)
# Probe rate ~1.7× capacity: enough to reach the throughput plateau (μ) without a huge
# backlog. Higher multiples (esp. for rag's 15k–80k-tok prompts) make the sim crawl.
probe_rate() { case "$1" in synth) echo 2.0 ;; mixed) echo 5.0 ;; prefill) echo 6.0 ;; *) echo 3.0 ;; esac; }
num_req() { echo 3000; }
# Offered load intensity grid ρ = offered/λ*, spanning underload → clear overload.
RHO_TARGETS=(0.5 0.75 1.0 1.25 1.5)

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi

COMMON=(--model "$MODEL"
  --num-instances 2 --prefill-instances 1 --decode-instances 1
  --pd-decider edpp --edpp-coeffs "$COEFFS" --edpp-c-xfer 100s
  --post-hoc-detector composite --trace-level decisions)

# Per-workload SLO/tau flags (match repro_stage_b.sh; rag has two classes).
slo_flags() {
  case "$1" in
    rag) echo --slo-ttft "standard=500ms,batch=5s" --slo-itl "standard=150ms,batch=200ms" \
              --edpp-tau-ttft-classes "standard=500ms,batch=5s" --edpp-tau-itl-classes "standard=150ms,batch=200ms" ;;
    *)   echo --slo-ttft "batch=2s" --slo-itl "batch=150ms" --edpp-tau-ttft 2s --edpp-tau-itl 150ms ;;
  esac
}

spec_for_rate() {  # <wl> <rate> -> temp spec path with aggregate_rate rewritten
  local wl="$1" rate="$2" dir="$3"; local base="campaigns/edpp-study/specs/${wl}_rate1.0.yaml"
  local dst="$dir/spec_rate${rate}.yaml" nreq; nreq="$(num_req "$wl")"
  # Rewrite rate and trim request count (ample for median estimator curves; keeps the
  # slow large-prompt rag runs tractable across the sweep).
  sed -E -e "s/^aggregate_rate:.*/aggregate_rate: ${rate}/" \
         -e "s/^num_requests:.*/num_requests: ${nreq}/" "$base" > "$dst"; echo "$dst"
}

sweep_wl() {
  local wl="$1"; local dir="$OUT/$wl"; mkdir -p "$dir"
  # shellcheck disable=SC2046
  local slo; slo=($(slo_flags "$wl"))

  # λ* = plateau throughput from one overloaded probe run (max sustainable rps = μ).
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
    # fluid run: capture outcome + admission + decision (outcome/admission are estimator-independent).
    ./blis run "${COMMON[@]}" "${slo[@]}" --edpp-tadm-estimator fluid --workload-spec "$spec" \
      --pd-outcome-trace "${tag}.outcome.csv" \
      --edpp-admission-trace "${tag}.admission.csv" \
      --edpp-decision-trace "${tag}.decision.fluid.csv" \
      --metrics-path "${tag}.metrics.json" >/dev/null 2>/dev/null
    # rollforward run: decision trace only.
    ./blis run "${COMMON[@]}" "${slo[@]}" --edpp-tadm-estimator rollforward --workload-spec "$spec" \
      --edpp-decision-trace "${tag}.decision.rollforward.csv" >/dev/null 2>/dev/null
  done
}

for wl in "${WORKLOADS[@]}"; do
  sweep_wl "$wl"
done

echo "plotting -> $OUT/fig_{admission,prefill,ttft}.png" >&2
python3 "$ANALYZE/ttft_d_local.py" --sweep-root "$OUT" --rho "${RHO_TARGETS[*]}" --out-prefix "$OUT/fig"
echo "done. figures: $OUT/fig_admission.png $OUT/fig_prefill.png $OUT/fig_ttft.png" >&2
