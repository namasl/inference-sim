#!/usr/bin/env bash
# T-A hardware-heterogeneity opportunity test (see FINDINGS.md "Hardware-θ
# opportunity test"). Serves a 1P2D deployment with the two DECODE instances on
# DIFFERENT hardware (fast H100 vs crippled A100) via a --policy-config bundle
# carrying hw_config_by_gpu, then asks: does the fixed-plan OPTIMUM beat every
# hardware-blind routing policy, and where do reduced- vs joint-EDPP land?
#
# Emits its own specs (campaigns/edpp-study/specs/ is git-ignored scratch), so it
# is self-contained from tracked files. Requires the hw_config_by_gpu bundle
# wiring (branch feat/edpp-estimator-validation, commits 60dcdaa..b07a79f).
#
# Placement note: llama-70b default TP=4 => 4 GPUs/instance on one node. The fast
# pool node has 8 GPUs (absorbs instance_0 prefill + instance_1 decode); the slow
# pool node has 4 (instance_2 decode spills there). First-fit, declaration order.
#
# Usage (from repo root):  bash campaigns/edpp-study/repro_hetero_hw.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
SPECDIR="campaigns/edpp-study/specs/hetero_hw"
OUT="${OUT:-campaigns/edpp-study/out/hetero_hw}"
SEEDS="${SEEDS:-42 7 123 2024}"
mkdir -p "$SPECDIR" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

# --- emit the heterogeneous bundle: instance_1=H100 fast, instance_2=A100 slow ---
cat > "$SPECDIR/bundle_1p2d.yaml" <<'YAML'
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: 400.0,  bw_peak_tbs: 0.7,  mfu_prefill: 0.5, mfu_decode: 0.5}
YAML

# --- emit the homogeneous decode-bound workload (only lever = which decode instance) ---
cat > "$SPECDIR/synth_hw.yaml" <<'YAML'
version: "2"
seed: 42
category: language
aggregate_rate: 1.0
num_requests: 60
clients:
  - id: "homogeneous-decode"
    tenant_id: "batch-jobs"
    slo_class: "batch"
    rate_fraction: 1.0
    streaming: false
    arrival: { process: poisson }
    input_distribution:  { type: constant, params: { value: 256 } }
    output_distribution: { type: constant, params: { value: 512 } }
    prefix_group: "hw"
    prefix_length: 0
YAML

BUNDLE="$SPECDIR/bundle_1p2d.yaml"; SPEC="$SPECDIR/synth_hw.yaml"
SLO=(--slo-ttft "batch=10s" --slo-itl "batch=50ms")   # fast~17ms MEETS, slow~74ms VIOLATES
TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --policy-config "$BUNDLE")
EC=(--edpp-coeffs "$COEFFS" --edpp-tau-ttft 10s --edpp-tau-itl 50ms --edpp-tadm-estimator rollforward)
gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['batch']['slo_attainment'])" 2>/dev/null || echo NA; }

# all-fast fixed plan = the optimum at this SLO (goodput = fast-fraction is linear)
python3 -c "
import csv
with open('$OUT/plan_allfast.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
    for i in range(60): w.writerow({'request_id':f'request_{i}','decode_instance':'instance_1','prefill_instance':'instance_0'})
"

echo "seed | reduced(dflt) | joint(dflt) | best-blind(loadbal) | optimum(all-fast)" >&2
for s in $SEEDS; do
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider edpp "${EC[@]}" "${SLO[@]}" --metrics-path "$OUT/r_$s.json" >/dev/null 2>&1; R=$(gp "$OUT/r_$s.json")
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider edpp "${EC[@]}" --edpp-joint "${SLO[@]}" --metrics-path "$OUT/j_$s.json" >/dev/null 2>&1; J=$(gp "$OUT/j_$s.json")
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider always --decode-routing-scorers "load-balance:1" "${SLO[@]}" --metrics-path "$OUT/lb_$s.json" >/dev/null 2>&1; L=$(gp "$OUT/lb_$s.json")
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-plan "$OUT/plan_allfast.csv" "${SLO[@]}" --metrics-path "$OUT/o_$s.json" >/dev/null 2>&1; O=$(gp "$OUT/o_$s.json")
  printf "%-5s| %-13s| %-11s| %-19s| %s\n" "$s" "$R" "$J" "$L" "$O" >&2
done
echo "done (under-capacity) -> $OUT" >&2

# ---------------------------------------------------------------------------
# SATURATING regime (SAT=1): cap per-instance concurrency and push arrival rate
# so the fast node saturates and the optimum is a NON-degenerate interior split
# (~86% fast). Shows reactive joint == blind load-balance (both undershoot the
# optimum) — the concrete case for per-instance θ_i (T-B). See FINDINGS.md
# "Saturating-regime follow-up".
# ---------------------------------------------------------------------------
[[ "${SAT:-0}" == "1" ]] || { echo "(set SAT=1 for the saturating-regime comparison)" >&2; exit 0; }

RATE="${RATE:-10}"; CAPN="${CAPN:-8}"; OPT="${OPT:-86}"   # optimal fast-fraction at rate 10
cat > "$SPECDIR/sat.yaml" <<YAML
version: "2"
seed: 42
category: language
aggregate_rate: $RATE
num_requests: 400
clients:
  - {id: hw, tenant_id: batch-jobs, slo_class: batch, rate_fraction: 1.0, streaming: false, arrival: {process: poisson}, input_distribution: {type: constant, params: {value: 256}}, output_distribution: {type: constant, params: {value: 64}}, prefix_group: hw, prefix_length: 0}
YAML
SSPEC="$SPECDIR/sat.yaml"
SSLO=(--slo-ttft "batch=60s" --slo-e2e "batch=8s" --slo-itl "batch=500ms")
SCAP=(--max-num-running-reqs "$CAPN")
python3 -c "
import csv
w=csv.DictWriter(open('$OUT/opt.csv','w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for i in range(400): w.writerow({'request_id':f'request_{i}','decode_instance':('instance_1' if i%100<$OPT else 'instance_2'),'prefill_instance':'instance_0'})"
echo "SATURATING rate=$RATE cap=$CAPN. seed | joint | blind-loadbal | reduced-loadbal | optimum($OPT% fast)" >&2
for s in $SEEDS; do
  J=$(./blis run --model "$MODEL" --workload-spec "$SSPEC" --seed "$s" "${TOPO[@]}" "${SCAP[@]}" --pd-decider edpp "${EC[@]}" --edpp-joint "${SSLO[@]}" --metrics-path "$OUT/sj_$s.json" >/dev/null 2>&1; gp "$OUT/sj_$s.json")
  B=$(./blis run --model "$MODEL" --workload-spec "$SSPEC" --seed "$s" "${TOPO[@]}" "${SCAP[@]}" --pd-decider always --decode-routing-scorers "load-balance:1" "${SSLO[@]}" --metrics-path "$OUT/sb_$s.json" >/dev/null 2>&1; gp "$OUT/sb_$s.json")
  R=$(./blis run --model "$MODEL" --workload-spec "$SSPEC" --seed "$s" "${TOPO[@]}" "${SCAP[@]}" --pd-decider edpp "${EC[@]}" --decode-routing-scorers "load-balance:1" "${SSLO[@]}" --metrics-path "$OUT/sr_$s.json" >/dev/null 2>&1; gp "$OUT/sr_$s.json")
  O=$(./blis run --model "$MODEL" --workload-spec "$SSPEC" --seed "$s" "${TOPO[@]}" "${SCAP[@]}" --pd-plan "$OUT/opt.csv" "${SSLO[@]}" --metrics-path "$OUT/so_$s.json" >/dev/null 2>&1; gp "$OUT/so_$s.json")
  printf "%-5s| %-6s| %-14s| %-16s| %s\n" "$s" "$J" "$B" "$R" "$O" >&2
done
echo "done (saturating) -> $OUT" >&2
