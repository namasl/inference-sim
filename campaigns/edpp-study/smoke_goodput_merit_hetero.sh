#!/usr/bin/env bash
# MERIT SMOKE TEST (heterogeneous cell) for --edpp-var-goodput.
# Heterogeneous hardware (fast H100 + crippled-A100 decode) is where the own-good reward term
# CAN differentiate candidates (θ_i differ), so it is the goodput reframing's best shot. Same
# arms as the homogeneous smoke: B=deployable dpVaR, G=+goodput+oracle, Go=+goodput deployable.
set -euo pipefail
cd /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/goodput_merit
OUT="${OUT:-campaigns/edpp-study/out/goodput_merit}"
SEEDS="${SEEDS:-42 7}"
VAROUT="${VAROUT:-0.4}"
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go
gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['$2']['slo_attainment'])" 2>/dev/null||echo NA; }
mean(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }
mn(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%min(xs) if xs else 'NA')" "$@"; }
outdist(){ if [ "$VAROUT" = "0" ]; then echo "{type: constant, params: {value: $1}}"; else
  local mu; mu=$(python3 -c "import math;print('%.4f'%(math.log($1)-$VAROUT*$VAROUT/2))")
  local mx; mx=$(python3 -c "print(int($1*8)+16)")
  echo "{type: lognormal, params: {mu: $mu, sigma: $VAROUT, min: 4, max: $mx}}"; fi; }
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl 100ms)

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
cat > "$D/hetero.yaml" <<YAML
version: "2"
seed: 42
category: language
aggregate_rate: 10
num_requests: 400
clients:
  - {id: hw, tenant_id: batch-jobs, slo_class: batch, rate_fraction: 1.0, streaming: false, arrival: {process: poisson}, input_distribution: {type: constant, params: {value: 256}}, output_distribution: $(outdist 64), prefix_group: hw, prefix_length: 0}
YAML
XT=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --policy-config "$D/bundle.yaml" --max-num-running-reqs 8)
XSLO=(--slo-ttft batch=60s --slo-e2e batch=8s --slo-itl batch=500ms)
xvv(){ local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$D/hetero.yaml" "${XT[@]}" "${XSLO[@]}" \
    --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-tau-e2e 8s \
    --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize \
    --edpp-var-congestion-weight 1 --edpp-var-deployable \
    "$@" --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json" batch; }

echo "GOODPUT-MERIT SMOKE (heterogeneous H100+cripA100)  sigma=$VAROUT  seeds=[$SEEDS]" >&2
declare -a B=() G=() GO=()
for s in $SEEDS; do
  B+=("$(xvv hb --seed $s)")
  G+=("$(xvv hg --seed $s --edpp-var-goodput --edpp-oracle-output-len)")
  GO+=("$(xvv hgo --seed $s --edpp-var-goodput)")
done
printf "%-14s| %-14s %-14s %-14s\n" "arm" "B (dpVaR)" "G (goodput+orc)" "Go (goodput dep)" >&2
printf "%-14s| %-6s(%-6s) %-6s(%-6s) %-6s(%-6s)\n" "heterogeneous" \
  "$(mean "${B[@]}")" "$(mn "${B[@]}")" "$(mean "${G[@]}")" "$(mn "${G[@]}")" "$(mean "${GO[@]}")" "$(mn "${GO[@]}")" >&2
echo "done   (mean(min) over seeds)" >&2
