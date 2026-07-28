#!/usr/bin/env bash
# Traced runs at the grid-v2 knee operating points for the SLO-miss decomposition
# and placement analysis (2026-07-28). Requires the local-record trace fix.
# Cells x {never, always, kairos, least-ttft, lt-joint, dpp, dpvar, gp-dep}, seed 42.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/policy_curves
OUT="${OUT:-campaigns/edpp-study/out/decomp}"
SEED=42
mkdir -p "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

spec(){ python3 - "$1" "$2" "$3" "$4" "$5" > "$6" <<'PY'
import sys, math
inp,out,rate,n,seed = sys.argv[1:6]
mu = math.log(float(out)) - 0.08
print(f"""version: "2"
seed: {seed}
category: language
aggregate_rate: {rate}
num_requests: {n}
clients:
  - {{id: w, tenant_id: t, slo_class: standard, rate_fraction: 1.0, streaming: false, arrival: {{process: poisson}}, input_distribution: {{type: constant, params: {{value: {inp}}}}}, output_distribution: {{type: lognormal, params: {{mu: {mu:.4f}, sigma: 0.4, min: 4, max: {int(float(out))*8+16}}}}}, prefix_group: g, prefix_length: 0}}""")
PY
}

T_1P2M="--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers queue-depth:1 --max-num-running-reqs 16"
T_HET="--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers queue-depth:1 --policy-config $D/bundle.yaml --max-num-running-reqs 16"

# cell in out rate T I E   (knee operating points + grid-v2 SLOs)
CELLS_DEF="decode 256 512 3.0 259 67 68320
mixed 2048 128 12.0 428 68 12537
prefill_lean 8192 64 12.0 1198 71 7202
prefill_bound 16000 16 8.0 2144 73 2788
hetero 256 64 6.0 259 67 6208
hetero_s10 256 64 10.0 259 67 6208
hetero_s12 256 64 12.0 259 67 6208
hetero_s14 256 64 14.0 259 67 6208"

arm_flags(){ local a=$1 T=$2 I=$3 E=$4
  local EC="--edpp-coeffs $COEF --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl ${I}ms"
  # dpvar = THE paper's rule: congestion + SLO deficits + the goodput penalty
  # R(a) = VaR(a) - good_r(a) (--edpp-var-goodput), deployable N-hat_out, floored norm.
  local VVF="--pd-decider edpp $EC --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft ${T}ms --edpp-tau-e2e ${E}ms"
  case $a in
    never)      echo "--pd-decider never" ;;
    always)     echo "--pd-decider always" ;;
    kairos)     echo "--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-rule kairos --kairos-beta 0.5" ;;
    lt-joint)   echo "--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-rule least-ttft --edpp-joint" ;;
    dpvar)      echo "$VVF" ;;
  esac; }

while read -r cell IN O R T I E; do
  [[ -z "$cell" ]] && continue
  N=$(python3 -c "import math;print(max(int(math.ceil($R*max(120, 10*$E/1000.0))), 600))")
  WF="$D/wd_${cell}.yaml"; spec "$IN" "$O" "$R" "$N" "$SEED" "$WF"
  TOPO="$T_1P2M"; [[ "$cell" == hetero* ]] && TOPO="$T_HET"
  SLO="--slo-ttft standard=${T}ms --slo-itl standard=${I}ms --slo-e2e standard=${E}ms"
  for a in never always kairos lt-joint dpvar; do
    ./blis run --model "$MODEL" --workload-spec "$WF" ${TOPO} ${SLO} $(arm_flags $a $T $I $E) --seed $SEED \
      --pd-outcome-trace "$OUT/${cell}_${a}.csv" --metrics-path "$OUT/${cell}_${a}.json" >/dev/null 2>&1 || true
    echo "traced $cell $a" >&2
  done
  echo "== $cell (T=${T}ms I=${I}ms E=${E}ms) =="
  python3 campaigns/edpp-study/analyze/decomposition.py "$T" "$I" "$E" "$OUT/${cell}"_*.csv
done <<< "$CELLS_DEF"
