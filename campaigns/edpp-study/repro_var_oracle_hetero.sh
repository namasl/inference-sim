#!/usr/bin/env bash
# Heterogeneous-θ_i VaR ORACLE test (E12 follow-up; companion to repro_hetero_hw.sh SAT=1 THETA=1).
# Design: docs/superpowers/specs/2026-07-21-edpp-var-oracle-design.md.
#
# THE QUESTION: on a SATURATING 1P2D deployment with two decode instances on DIFFERENT hardware
# (fast H100 + crippled A100), the goodput optimum is a NON-degenerate interior decode split
# (~OPT% fast). E5 found that work-currency θ_i-joint OVER-corrects here — the drift term
# overwhelms the congestion signal and it does not reach the optimum. Does re-pricing the drift
# term in VALUE-AT-RISK fix that? A fast node destroys LESS goodput per unit work (its co-residents
# complete sooner even with the added load), so a goodput-currency externality should push toward
# the fast node up to — but not past — the point where the fast node's own co-residents start
# flipping. That is exactly the interior optimum work-currency drift overshoots.
#
# SETUP (mirrors repro_hetero_hw.sh SAT=1 THETA=1): fast/slow node pools + hw_config_by_gpu +
# coeffs_by_gpu (per-instance θ_i), joint (decode,prefill) argmin, saturating batch workload.
# The VaR arms add --edpp-rule var + --edpp-var-metric + --edpp-tau-e2e (matching the goodput SLO),
# composed with --edpp-oracle-output-len for a clean ceiling (VAROR=1 default). Each arm reports
# goodput AND the realized fast/slow decode split, so we see whether VaR reaches the optimum's mix.
#
# Usage (from repo root):
#   bash campaigns/edpp-study/repro_var_oracle_hetero.sh
#   RATE=10 CAPN=8 OPT=86 SEEDS="42 7 123 2024" bash campaigns/edpp-study/repro_var_oracle_hetero.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
THETA_H100="${THETA_H100:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
THETA_A100="${THETA_A100:-scripts/calibration/coeffs-llama70b-a100crippled-tp4.json}"
SPECDIR="campaigns/edpp-study/specs/var_hetero"
OUT="${OUT:-campaigns/edpp-study/out/var_hetero}"
SEEDS="${SEEDS:-42 7 123 2024}"
RATE="${RATE:-10}"; CAPN="${CAPN:-8}"; OPT="${OPT:-86}"   # optimal fast-fraction at rate 10 (from E5)
mkdir -p "$SPECDIR" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

# VAROR=1 (default): compose VaR with --edpp-oracle-output-len for a fully clean ceiling.
VOR=(); VORLBL=""
if [[ "${VAROR:-1}" == "1" ]]; then VOR=(--edpp-oracle-output-len); VORLBL=" +oracle-o_r"; fi

# NORM=1: per-decision min-max auto-normalization ⇒ scale-free congestion weight (default w=1 then).
NRM=(); NORMLBL=""
if [[ "${NORM:-0}" == "1" ]]; then NRM=(--edpp-var-normalize); NORMLBL="+norm"; : "${VARCONGW:=1}"; fi
DCW="${VARCONGW:-10000}"

# --- per-instance θ_i bundle: instance_1=H100 fast decode, instance_2=A100 slow decode ---
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

# --- saturating decode-bound workload (only lever = which decode instance) ---
cat > "$SPECDIR/sat.yaml" <<YAML
version: "2"
seed: 42
category: language
aggregate_rate: $RATE
num_requests: 400
clients:
  - {id: hw, tenant_id: batch-jobs, slo_class: batch, rate_fraction: 1.0, streaming: false, arrival: {process: poisson}, input_distribution: {type: constant, params: {value: 256}}, output_distribution: {type: constant, params: {value: 64}}, prefix_group: hw, prefix_length: 0}
YAML

TBUNDLE="$SPECDIR/bundle_theta.yaml"; SSPEC="$SPECDIR/sat.yaml"
SSLO=(--slo-ttft "batch=60s" --slo-e2e "batch=8s" --slo-itl "batch=500ms")
TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --policy-config "$TBUNDLE")
SCAP=(--max-num-running-reqs "$CAPN")
# EDPP basis: rollforward estimator, real τ's matching the SLO, size-aware c_xfer.
EC=(--edpp-coeffs "$COEFFS" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware)
# VaR arms additionally match the E2E composite to the goodput SLO (batch e2e = 8s).
VC=("${EC[@]}" --edpp-tau-e2e 8s --edpp-rule var)

gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['batch']['slo_attainment'])" 2>/dev/null || echo NA; }
split(){ python3 -c "
import re,sys
try: txt=open('$1').read()
except OSError: print('NA'); sys.exit()
m={iid:int(c) for iid,c in re.findall(r'\"instance_id\":\s*\"(instance_\d+)\".*?\"completed_requests\":\s*(\d+)', txt, re.S)}
f=m.get('instance_1',0); s=m.get('instance_2',0); t=f+s
print('%d/%d (%.0f%% fast)'%(f,s,100.0*f/t) if t else 'NA')
" 2>/dev/null || echo NA; }
arm(){ local base="$1"; shift; ./blis run "$@" --metrics-path "$base.json" >"$base.out" 2>/dev/null; printf '%s  %s' "$(gp "$base.json")" "$(split "$base.out")"; }

# optimum fixed plan: first OPT of every 100 requests -> fast (instance_1), rest -> slow (instance_2).
python3 -c "
import csv
w=csv.DictWriter(open('$OUT/opt.csv','w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for i in range(400): w.writerow({'request_id':f'request_{i}','decode_instance':('instance_1' if i%100<$OPT else 'instance_2'),'prefill_instance':'instance_0'})"

echo "HETERO-θ_i drift-plus-VaR [JOINT +θ_i$VORLBL +c_xfer-size$NORMLBL w=$DCW]  saturating rate=$RATE cap=$CAPN  arm = goodput  fast/slow (NN% fast)" >&2
echo "(interior optimum ~$OPT% fast; θ_i-joint dpp OVER-corrects (E5); PURE var OVER-routes to fast (F17)." >&2
echo " DOES adding the congestion drift back (--edpp-var-congestion) stop the over-routing and reach the optimum?)" >&2
echo "seed | theta-joint(dpp) | var:util(pure)     | dpVaR:util(w=${VARCONGW:-10000}) | dpVaR:hazard(w=${VARCONGW:-10000})| blind-loadbal      | optimum($OPT% fast)" >&2
SCOMMON=(--model "$MODEL" --workload-spec "$SSPEC" "${TOPO[@]}" "${SCAP[@]}" "${SSLO[@]}")
for s in $SEEDS; do
  DPP=$(arm  "$OUT/tj_$s"  "${SCOMMON[@]}" --seed "$s" --pd-decider edpp "${EC[@]}" --edpp-joint)
  VU=$(arm   "$OUT/vu_$s"  "${SCOMMON[@]}" --seed "$s" --pd-decider edpp "${VC[@]}" --edpp-joint ${VOR[@]+"${VOR[@]}"} --edpp-var-metric util)
  CU=$(arm   "$OUT/cu_$s"  "${SCOMMON[@]}" --seed "$s" --pd-decider edpp "${VC[@]}" --edpp-joint ${VOR[@]+"${VOR[@]}"} ${NRM[@]+"${NRM[@]}"} --edpp-var-metric util   --edpp-var-congestion --edpp-var-congestion-weight "$DCW")
  CH=$(arm   "$OUT/ch_$s"  "${SCOMMON[@]}" --seed "$s" --pd-decider edpp "${VC[@]}" --edpp-joint ${VOR[@]+"${VOR[@]}"} ${NRM[@]+"${NRM[@]}"} --edpp-var-metric hazard --edpp-var-congestion --edpp-var-congestion-weight "$DCW")
  B=$(arm    "$OUT/b_$s"   "${SCOMMON[@]}" --seed "$s" --pd-decider always --decode-routing-scorers "load-balance:1")
  O=$(arm    "$OUT/o_$s"   "${SCOMMON[@]}" --seed "$s" --pd-plan "$OUT/opt.csv")
  printf "%-5s| %-18s| %-19s| %-19s| %-19s| %-19s| %s\n" "$s" "$DPP" "$VU" "$CU" "$CH" "$B" "$O" >&2
done
echo "done -> $OUT" >&2
