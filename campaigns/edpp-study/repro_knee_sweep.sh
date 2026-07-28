#!/usr/bin/env bash
# Phase B/C of the eval-protocol rebuild: per-cell baseline probe + rate sweep to
# locate the capacity knee (achieved/offered >= 0.95 criterion, per
# slo-methodology-research.md). Arms: always and never (fleet capacity ~ max of both).
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="meta-llama/llama-3.3-70b-instruct"
COEF="scripts/calibration/coeffs-llama70b-h100-tp4.json"
S=campaigns/edpp-study/out
D=$S/knee_v2; OUT=$D/out; mkdir -p "$D" "$OUT"

jget(){ python3 -c "import json;m=json.load(open('$1'));print(m.get('$2',0))" 2>/dev/null||echo 0; }

spec(){ # in out rate n seed file
  python3 - "$1" "$2" "$3" "$4" "$5" > "$6" <<'PY'
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

HTOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 16)
NOSLO=(--slo-ttft "standard=9999s" --slo-itl "standard=9999s" --slo-e2e "standard=9999s")

# hetero bundle (unified batch cap 16 now)
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
XTOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" --policy-config "$D/bundle.yaml" --max-num-running-reqs 16)

probe_and_sweep(){ # name in out topo... ; rates via env RATES
  local NAME=$1 IN=$2 O=$3; shift 3
  local TOPO=("$@")
  # baseline probe: rate 0.2, 60 requests
  spec "$IN" "$O" 0.2 60 42 "$D/w.yaml"
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${TOPO[@]}" "${NOSLO[@]}" --pd-decider always \
    --seed 42 --metrics-path "$OUT/base_$NAME.json" >/dev/null 2>&1
  local bt bi be
  bt=$(jget "$OUT/base_$NAME.json" ttft_p99_ms); bi=$(jget "$OUT/base_$NAME.json" itl_p99_ms); be=$(jget "$OUT/base_$NAME.json" e2e_p99_ms)
  echo "BASELINE $NAME ttft_p99=${bt}ms itl_p99=${bi}ms e2e_p99=${be}ms"
  for R in $RATES; do
    for ARM in always never; do
      spec "$IN" "$O" "$R" 600 42 "$D/w.yaml"
      ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${TOPO[@]}" "${NOSLO[@]}" --pd-decider $ARM \
        --seed 42 --metrics-path "$OUT/sw_${NAME}_${ARM}_${R}.json" >/dev/null 2>&1 || true
      local ach e2e
      ach=$(jget "$OUT/sw_${NAME}_${ARM}_${R}.json" responses_per_sec); e2e=$(jget "$OUT/sw_${NAME}_${ARM}_${R}.json" e2e_p99_ms)
      echo "SWEEP $NAME $ARM rate=$R achieved=$ach e2e_p99=${e2e}ms"
    done
  done
}

RATES="1.5 2.5 3.0 3.5 4.0 5.0"          probe_and_sweep decode        256   512 "${HTOPO[@]}"
RATES="6 9 12 14 16 20"                  probe_and_sweep mixed         2048  128 "${HTOPO[@]}"
RATES="6 9 12 14 16 20"                  probe_and_sweep prefill_lean  8192  64  "${HTOPO[@]}"
RATES="4 6 8 10 12 14"                   probe_and_sweep prefill_bound 16000 16  "${HTOPO[@]}"
RATES="1.0 1.5 2.0 2.5 3.0 4.0"          probe_and_sweep hetero        256   64  "${XTOPO[@]}"
echo DONE
