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

# --- THETA=1: emit a SECOND bundle that additionally carries per-instance θ_i (T-B).
# Same node_pools + hw_config_by_gpu as bundle_1p2d.yaml, PLUS coeffs_by_gpu keyed by
# gpu_type: the fast H100 file and the slow-A100 file fit (from the SAME 400 TFLOPS /
# 0.7 TB/s HWConfig used by the slow pool above) so the decider's θ_i matches execution.
# The joint decider (--edpp-joint) uses this to score each decode candidate with its own
# hardware work model, proactively over-weighting the fast node toward the optimal split.
THETA_H100="${THETA_H100:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
THETA_A100="${THETA_A100:-scripts/calibration/coeffs-llama70b-a100crippled-tp4.json}"
if [[ "${THETA:-0}" == "1" ]]; then
  cat > "$SPECDIR/bundle_1p2d_theta.yaml" <<YAML
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
fi
TBUNDLE="$SPECDIR/bundle_1p2d_theta.yaml"

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
# split(): parse the per-instance stdout blocks captured for an arm and report the
# realized fast/slow DECODE split. instance_1 = fast H100 decode, instance_2 = slow
# A100 decode (instance_0 = prefill). Every completed request decodes exactly once, so
# instance_1.completed / instance_2.completed IS the realized decode allocation.
# Prints "fast/slow (NN% fast)"; "NA" if the capture has no per-instance blocks.
split(){ python3 -c "
import re,sys
try: txt=open('$1').read()
except OSError: print('NA'); sys.exit()
m={iid:int(c) for iid,c in re.findall(r'\"instance_id\":\s*\"(instance_\d+)\".*?\"completed_requests\":\s*(\d+)', txt, re.S)}
f=m.get('instance_1',0); s=m.get('instance_2',0); t=f+s
print('%d/%d (%.0f%% fast)'%(f,s,100.0*f/t) if t else 'NA')
" 2>/dev/null || echo NA; }
# arm(): run one blis arm, capturing stdout (per-instance blocks) to <base>.out and the
# aggregate metrics to <base>.json, then echo "goodput  fast/slow (NN% fast)".
arm(){ local base="$1"; shift; ./blis run "$@" --metrics-path "$base.json" >"$base.out" 2>/dev/null; printf '%s  %s' "$(gp "$base.json")" "$(split "$base.out")"; }

# all-fast fixed plan = the optimum at this SLO (goodput = fast-fraction is linear)
python3 -c "
import csv
with open('$OUT/plan_allfast.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
    for i in range(60): w.writerow({'request_id':f'request_{i}','decode_instance':'instance_1','prefill_instance':'instance_0'})
"

# THETA=1 adds a θ_i-joint column here to confirm the under-capacity regime (where
# reactive joint already wins ~0.97) is NOT regressed by per-instance θ_i (design §7).
if [[ "${THETA:-0}" == "1" ]]; then
  echo "seed | reduced(dflt) | joint(dflt) | theta-joint | best-blind(loadbal) | optimum(all-fast)" >&2
else
  echo "seed | reduced(dflt) | joint(dflt) | best-blind(loadbal) | optimum(all-fast)" >&2
fi
for s in $SEEDS; do
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider edpp "${EC[@]}" "${SLO[@]}" --metrics-path "$OUT/r_$s.json" >/dev/null 2>&1; R=$(gp "$OUT/r_$s.json")
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider edpp "${EC[@]}" --edpp-joint "${SLO[@]}" --metrics-path "$OUT/j_$s.json" >/dev/null 2>&1; J=$(gp "$OUT/j_$s.json")
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider always --decode-routing-scorers "load-balance:1" "${SLO[@]}" --metrics-path "$OUT/lb_$s.json" >/dev/null 2>&1; L=$(gp "$OUT/lb_$s.json")
  ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-plan "$OUT/plan_allfast.csv" "${SLO[@]}" --metrics-path "$OUT/o_$s.json" >/dev/null 2>&1; O=$(gp "$OUT/o_$s.json")
  if [[ "${THETA:-0}" == "1" ]]; then
    ./blis run --model "$MODEL" --workload-spec "$SPEC" --seed "$s" "${TOPO[@]}" --pd-decider edpp "${EC[@]}" --edpp-joint --policy-config "$TBUNDLE" "${SLO[@]}" --metrics-path "$OUT/tj_$s.json" >/dev/null 2>&1; TJ=$(gp "$OUT/tj_$s.json")
    printf "%-5s| %-13s| %-11s| %-11s| %-19s| %s\n" "$s" "$R" "$J" "$TJ" "$L" "$O" >&2
  else
    printf "%-5s| %-13s| %-11s| %-19s| %s\n" "$s" "$R" "$J" "$L" "$O" >&2
  fi
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
# Each arm now reports "goodput  fast/slow (NN% fast)" (goodput from the aggregate
# metrics file; the realized decode split from the per-instance stdout blocks).
# THETA=1 adds the θ_i-joint arm: the SAME joint decider but with a coeffs_by_gpu bundle
# (per-instance θ_i) — the T-B acceptance experiment (design §7). PASS = it shifts the
# realized fast-share ~77%→~86% and goodput ~0.82→~0.96, beating reduced and blind.
SCOMMON=(--model "$MODEL" --workload-spec "$SSPEC" "${TOPO[@]}" "${SCAP[@]}" "${SSLO[@]}")
if [[ "${THETA:-0}" == "1" ]]; then
  echo "SATURATING rate=$RATE cap=$CAPN. arm = goodput  fast/slow (NN% fast)" >&2
  echo "seed | theta-joint (T-B) | joint(homog) | blind-loadbal | reduced-loadbal | optimum($OPT% fast)" >&2
else
  echo "SATURATING rate=$RATE cap=$CAPN. arm = goodput  fast/slow (NN% fast)" >&2
  echo "seed | joint(homog) | blind-loadbal | reduced-loadbal | optimum($OPT% fast)" >&2
fi
for s in $SEEDS; do
  J=$(arm "$OUT/sj_$s" "${SCOMMON[@]}" --seed "$s" --pd-decider edpp "${EC[@]}" --edpp-joint)
  B=$(arm "$OUT/sb_$s" "${SCOMMON[@]}" --seed "$s" --pd-decider always --decode-routing-scorers "load-balance:1")
  R=$(arm "$OUT/sr_$s" "${SCOMMON[@]}" --seed "$s" --pd-decider edpp "${EC[@]}" --decode-routing-scorers "load-balance:1")
  O=$(arm "$OUT/so_$s" "${SCOMMON[@]}" --seed "$s" --pd-plan "$OUT/opt.csv")
  if [[ "${THETA:-0}" == "1" ]]; then
    T=$(arm "$OUT/st_$s" "${SCOMMON[@]}" --seed "$s" --pd-decider edpp "${EC[@]}" --edpp-joint --policy-config "$TBUNDLE")
    printf "%-5s| %-18s| %-19s| %-19s| %-19s| %s\n" "$s" "$T" "$J" "$B" "$R" "$O" >&2
  else
    printf "%-5s| %-19s| %-19s| %-19s| %s\n" "$s" "$J" "$B" "$R" "$O" >&2
  fi
done
echo "done (saturating) -> $OUT" >&2
