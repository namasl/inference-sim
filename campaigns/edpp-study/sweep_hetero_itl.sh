#!/usr/bin/env bash
# HETEROGENEOUS-REGIME ITL SENSITIVITY. The paper's headline (drift-plus-VaR
# worst-case regret 0.05 vs 0.38-0.92) rests on the heterogeneous-hardware regime,
# where the published rules collapse. That regime binds on ITL, so it is where the
# ITL knob could change the story. We sweep the ITL target and watch whether the
# hardware-aware joint rule keeps its advantage over the blind load-balancer and
# over least-TTFT.
#
# Fleet: 1P2D, decode instance_1 = H100 (~17ms/step), instance_2 = crippled A100
# (~73-77ms/step). The ITL target that makes the hardware gap BIND must sit
# between the two step times. We sweep across and past that window:
#   40ms  (below fast=17? no; between) 50ms (paper) 60ms 73ms (== slow) 100ms 200ms
# PASS/finding = identify the ITL window where joint's advantage exists and where
# it vanishes (slow hardware starts meeting the target), so the paper can state the
# contingency honestly.
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
THETA_H100="scripts/calibration/coeffs-llama70b-h100-tp4.json"
THETA_A100="scripts/calibration/coeffs-llama70b-a100crippled-tp4.json"
SPECDIR="campaigns/edpp-study/specs/hetitl"
OUT="${OUT:-campaigns/edpp-study/out/hetitl}"
SEEDS="${SEEDS:-42 7 123}"
ITLS="${ITLS:-40 50 60 73 100 200}"
mkdir -p "$SPECDIR" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

cat > "$SPECDIR/bundle_theta.yaml" <<YAML
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 400.0,  bw_peak_tbs: 0.7,  mfu_prefill: 0.5, mfu_decode: 0.5}
coeffs_by_gpu:
  H100: $THETA_H100
  A100: $THETA_A100
YAML
cat > "$SPECDIR/bundle.yaml" <<'YAML'
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 400.0,  bw_peak_tbs: 0.7,  mfu_prefill: 0.5, mfu_decode: 0.5}
YAML
cat > "$SPECDIR/synth.yaml" <<'YAML'
version: "2"
seed: 42
category: language
aggregate_rate: 1.0
num_requests: 60
clients:
  - {id: hw, tenant_id: batch-jobs, slo_class: batch, rate_fraction: 1.0, streaming: false, arrival: {process: poisson}, input_distribution: {type: constant, params: {value: 256}}, output_distribution: {type: constant, params: {value: 512}}, prefix_group: hw, prefix_length: 0}
YAML

BUNDLE="$SPECDIR/bundle.yaml"; TBUNDLE="$SPECDIR/bundle_theta.yaml"; SPEC="$SPECDIR/synth.yaml"
TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --policy-config "$BUNDLE")
gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['batch']['slo_attainment'])" 2>/dev/null || echo NA; }
mean3(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x!='NA'];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }

echo "HETERO ITL SWEEP  1P2D (H100 fast ~17ms/step, A100 crippled ~73-77ms/step), decode-bound 256/512" >&2
echo "arm = mean goodput over seeds [$SEEDS].  TTFT=10s loose, no E2E; ITL is the sole binding target." >&2
printf "   %-7s| %-14s %-14s %-14s %-14s\n" "ITL" "theta-joint" "blind-loadbal" "least-ttft" "optimum(allfast)" >&2

# all-fast optimum plan
python3 -c "
import csv
w=csv.DictWriter(open('$OUT/allfast.csv','w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for i in range(60): w.writerow({'request_id':f'request_{i}','decode_instance':'instance_1','prefill_instance':'instance_0'})"

for ITL in $ITLS; do
  SLO=(--slo-ttft "batch=10s" --slo-itl "batch=${ITL}ms")
  EC=(--edpp-coeffs "$COEFFS" --edpp-tau-ttft 10s --edpp-tau-itl "${ITL}ms" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware)
  TJ=(); BL=(); LT=(); OP=()
  for s in $SEEDS; do
    ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider edpp "${EC[@]}" --edpp-joint --policy-config "$TBUNDLE" "${SLO[@]}" --metrics-path "$OUT/tj_${ITL}_$s.json" >/dev/null 2>&1 || true
    ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider always --decode-routing-scorers "load-balance:1" "${SLO[@]}" --metrics-path "$OUT/bl_${ITL}_$s.json" >/dev/null 2>&1 || true
    ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider edpp "${EC[@]}" --edpp-rule least-ttft --decode-routing-scorers "queue-depth:1" "${SLO[@]}" --metrics-path "$OUT/lt_${ITL}_$s.json" >/dev/null 2>&1 || true
    ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-plan "$OUT/allfast.csv" "${SLO[@]}" --metrics-path "$OUT/op_${ITL}_$s.json" >/dev/null 2>&1 || true
    TJ+=("$(gp "$OUT/tj_${ITL}_$s.json")"); BL+=("$(gp "$OUT/bl_${ITL}_$s.json")")
    LT+=("$(gp "$OUT/lt_${ITL}_$s.json")"); OP+=("$(gp "$OUT/op_${ITL}_$s.json")")
  done
  printf "   %-7s| %-14s %-14s %-14s %-14s\n" "${ITL}ms" "$(mean3 "${TJ[@]}")" "$(mean3 "${BL[@]}")" "$(mean3 "${LT[@]}")" "$(mean3 "${OP[@]}")" >&2
done
echo "done -> $OUT" >&2
