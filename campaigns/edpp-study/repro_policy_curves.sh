#!/usr/bin/env bash
# Policy capacity curves — the analysis layer behind grid v2 (2026-07-28).
# For every cell and every policy, sweep offered rate across the knee and record
# achieved throughput + TTFT/ITL/E2E p99 + goodput per run, one CSV row per run.
# Also runs the aggregated reference "never@3M" (3 mixed instances, no prefill
# instance: the equal-hardware aggregated-serving baseline the fixed 1P2M
# topology never provides).
#
# Output: campaigns/edpp-study/out/policy_curves/curves.csv with columns
#   cell,policy,topology,rate,seed,achieved_rps,goodput,ttft_p99_ms,itl_p99_ms,e2e_p99_ms,completed,injected
# Single seed (42) for the curves; the grid-v2 operating points carry the seeds.
# Usage: CELLS="decode ..." bash campaigns/edpp-study/repro_policy_curves.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/policy_curves
OUT="${OUT:-campaigns/edpp-study/out/policy_curves}"
SEED="${SEED:-42}"
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

CSV="$OUT/curves.csv"
[[ -f "$CSV" ]] || echo "cell,policy,topology,rate,seed,achieved_rps,goodput,ttft_p99_ms,itl_p99_ms,e2e_p99_ms,completed,injected" > "$CSV"

row(){ python3 - "$1" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
pc=m.get("per_class",{}).get("standard",{})
print(f"{m.get('responses_per_sec',0):.4f},{pc.get('slo_attainment',0):.4f},{m.get('ttft_p99_ms',0):.1f},{m.get('itl_p99_ms',0):.1f},{m.get('e2e_p99_ms',0):.1f},{m.get('completed_requests',0)},{m.get('injected_requests',0)}")
PY
}

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

cat > "$D/bundle.yaml" <<'YAML'
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 400.0,  bw_peak_tbs: 0.7,  mfu_prefill: 0.5, mfu_decode: 0.5}
coeffs_by_gpu:
  H100: scripts/calibration/coeffs-llama70b-h100-tp4.json
  A100: scripts/calibration/coeffs-llama70b-a100crippled-tp4.json
YAML

# Topologies. agg3m = 3 mixed engines with NO PD disaggregation at all (plain
# cluster mode, no --pd-decider / pool flags): the aggregated-serving reference.
T_1P2M="--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers queue-depth:1 --max-num-running-reqs 16"
T_AGG3M="--num-instances 3 --routing-scorers queue-depth:1 --max-num-running-reqs 16"
T_HET="--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers queue-depth:1 --policy-config $D/bundle.yaml --max-num-running-reqs 16"

# SLOs per cell (grid-v2 derivation: 5x/5x/3x of rate-0.2 baseline p99s; cells.txt values).
slo_of(){ case $1 in
  decode)        echo "259 84 68320" ;;
  mixed)         echo "428 84 12537" ;;
  prefill_lean)  echo "1198 89 7202" ;;
  prefill_bound) echo "2144 91 2788" ;;
  hetero)        echo "259 84 6208" ;;
esac; }

rates_of(){ case $1 in
  decode)        echo "1.5 2.1 2.5 3.0 3.5 4.0" ;;
  mixed)         echo "6 8.4 10 12 14 16" ;;
  prefill_lean)  echo "6 8.4 10 12 14 16" ;;
  prefill_bound) echo "3 4 5.6 7 8 10" ;;
  hetero)        echo "2 3 4.2 5 6 8" ;;
esac; }

wl_of(){ case $1 in
  decode)        echo "256 512" ;;
  mixed)         echo "2048 128" ;;
  prefill_lean)  echo "8192 64" ;;
  prefill_bound) echo "16000 16" ;;
  hetero)        echo "256 64" ;;
esac; }

# policy flag sets (grid-v2 arm definitions; floored normalization = deployed default)
arm_flags(){ local a=$1 T=$2 I=$3 E=$4
  local EC="--edpp-coeffs $COEF --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl ${I}ms"
  local VVF="--pd-decider edpp $EC --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-tau-ttft ${T}ms --edpp-tau-e2e ${E}ms"
  case $a in
    never)      echo "--pd-decider never" ;;
    always)     echo "--pd-decider always" ;;
    kairos)     echo "--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-rule kairos --kairos-beta 0.5" ;;
    least-ttft) echo "--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-rule least-ttft" ;;
    lt-joint)   echo "--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-rule least-ttft --edpp-joint" ;;
    dpp)        echo "--pd-decider edpp $EC --edpp-tau-ttft ${T}ms --edpp-joint" ;;
    dpvar)      echo "$VVF" ;;
    gp-dep)     echo "$VVF --edpp-var-goodput" ;;
  esac; }

ARMS="never always kairos least-ttft lt-joint dpp dpvar gp-dep"
CELLS="${CELLS:-decode mixed prefill_lean prefill_bound hetero}"

for cell in $CELLS; do
  set -- $(wl_of $cell); IN=$1; O=$2
  set -- $(slo_of $cell); T=$1; I=$2; E=$3
  SLO="--slo-ttft standard=${T}ms --slo-itl standard=${I}ms --slo-e2e standard=${E}ms"
  for R in $(rates_of $cell); do
    N=$(python3 -c "import math;print(max(int(math.ceil($R*max(120, 10*$E/1000.0))), 600))")
    WF="$D/w_${cell}_${R}.yaml"; spec "$IN" "$O" "$R" "$N" "$SEED" "$WF"
    TOPO="$T_1P2M"; [[ "$cell" == hetero ]] && TOPO="$T_HET"
    for a in $ARMS; do
      MP="$OUT/${cell}_${a}_${R}.json"
      if [[ ! -s "$MP" ]]; then
        ./blis run --model "$MODEL" --workload-spec "$WF" ${TOPO} ${SLO} $(arm_flags $a $T $I $E) --seed "$SEED" --metrics-path "$MP" >/dev/null 2>&1 || true
      fi
      [[ -s "$MP" ]] && echo "$cell,$a,1P2M,$R,$SEED,$(row $MP)" | sed "s/1P2M/$([[ $cell == hetero ]] && echo 1P2M-het || echo 1P2M)/" >> "$CSV"
      echo "done $cell $a rate=$R" >&2
    done
    # aggregated reference (homogeneous cells only; hetero agg reference = fast-only fleet question, separate)
    if [[ "$cell" != hetero ]]; then
      MP="$OUT/${cell}_agg3m_${R}.json"
      if [[ ! -s "$MP" ]]; then
        ./blis run --model "$MODEL" --workload-spec "$WF" ${T_AGG3M} ${SLO} --seed "$SEED" --metrics-path "$MP" >/dev/null 2>&1 || true
      fi
      [[ -s "$MP" ]] && echo "$cell,never@3M,AGG3M,$R,$SEED,$(row $MP)" >> "$CSV"
      echo "done $cell never@3M rate=$R" >&2
    fi
  done
done
echo "curves -> $CSV"
