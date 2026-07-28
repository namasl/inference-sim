#!/usr/bin/env bash
# Policy-comparison grid v2 — rebuilt protocol after the 2026-07-27 eval audit
# (memory: edpp-eval-audit-2026-07-27). Differences from repro_var_dominance_goodput.sh:
#
#   1. RATES ARE KNEE-RELATIVE. Each cell's capacity knee is measured first
#      (rate sweep, achieved/offered >= 0.95 criterion, slo-methodology-research.md),
#      and the grid runs at 0.7x knee (operational) and 1.0x knee (stress).
#   2. SLOs ARE DERIVED UNIFORMLY, NO FLOORS. Baseline probe at rate 0.2 with no
#      contention; targets = 5x TTFT p99, 4x ITL p99, 3x E2E p99 (Method A,
#      Splitwise/Sarathi-style slowdown multiples). ITL is derived, not asserted.
#   3. STEADY-STATE HORIZONS. num_requests = rate x max(120 s, 10 x E2E SLO),
#      so goodput is horizon-stable (checked: n vs 2n within seed noise).
#   4. ONE SCORER CONFIG EVERYWHERE: --decode-routing-scorers queue-depth:1,
#      batch cap 16, on the heterogeneous cell too.
#   5. HETERO WORKLOAD RESEEDS PER SEED (spec seed = $s, not fixed 42).
#   6. FINAL ARM SET (2026-07-28, agreed with the paper's policy list): never,
#      always, kairos, lt-joint, dpvar. Everything is JOINT except the two static
#      corners and Kairos, which must pick a decode instance somehow and use the
#      load-balancing (queue-depth) scorer. Deliberately NOT included:
#        - least-TTFT (non-joint): its result is a property of whichever scorer we
#          pair with it, and the paper does not argue about scorers.
#        - prefix-threshold: numerically identical to always in every cell here.
#        - drift-plus-penalty: not a policy the paper claims.
#        - the output-length ORACLE arm: never deployable (INV-9), removed to
#          avoid any chance of an oracle number reaching a table.
#      Every arm below is deployable: no arm reads a realized output length.
#
# Knee/baseline numbers are filled in from the Phase-B sweep before running.
# Usage: bash campaigns/edpp-study/repro_grid_v2.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/grid_v2
OUT="${OUT:-campaigns/edpp-study/out/grid_v2}"
SEEDS="${SEEDS:-42 7 123 5 11}"
KBETAS="${KBETAS:-0.25 0.5 1.0}"
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['standard']['slo_attainment'])" 2>/dev/null||echo NA; }
mean(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }
mn(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%min(xs) if xs else 'NA')" "$@"; }
mx(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%max(xs) if xs else 'NA')" "$@"; }

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

# ---- Cells: name in out knee ttft_base_ms itl_base_ms e2e_base_ms ----
# knee + baselines filled from the Phase-B sweep (see audit scratchpad knee/).
# CELLS is populated below by the caller-editable block.
CELLS_FILE="${CELLS_FILE:-campaigns/edpp-study/specs/grid_v2/cells.txt}"
[[ -f "$CELLS_FILE" ]] || { echo "cells.txt missing: run the knee sweep and write $CELLS_FILE" >&2; exit 1; }

run_cell(){ # name in out topo_kind rate slo_ttft_ms slo_itl_ms slo_e2e_ms n loadtag
  local NAME=$1 IN=$2 O=$3 TK=$4 R=$5 T=$6 I=$7 E=$8 N=$9 LT=${10}
  local TOPO=("${HTOPO[@]}"); [[ "$TK" == hetero ]] && TOPO=("${XTOPO[@]}")
  local SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
  local EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
  # dpvar = THE paper's rule: congestion + SLO deficits + the goodput penalty
  # R(a) = VaR(a) - good_r(a) (--edpp-var-goodput), deployable N-hat_out, floored
  # normalization. This is exactly eq:penalty / eq:kv, so the evaluated rule and the
  # derived rule are the same object.
  local VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")
  declare -a NV=() A=() K=() LJ=() VV=()
  for s in $SEEDS; do
    # per-cell/load/seed spec path: chunked invocations of this script must not
    # race each other's workload files (the audit-session parallel run did).
    local WF="$D/w_${NAME}_${LT}_$s.yaml"
    spec "$IN" "$O" "$R" "$N" "$s" "$WF"
    r(){ local tag=$1; shift; ./blis run --model "$MODEL" --workload-spec "$WF" "${TOPO[@]}" "${SLO[@]}" "$@" --seed "$s" --metrics-path "$OUT/${NAME}_${LT}_${tag}_$s.json" >/dev/null 2>&1 || true; gp "$OUT/${NAME}_${LT}_${tag}_$s.json"; }
    NV+=("$(r nv --pd-decider never)")
    A+=("$(r a --pd-decider always)")
    kb=(); for bb in $KBETAS; do kb+=("$(r k$bb --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${T}ms" --edpp-rule kairos --kairos-beta $bb)"); done
    K+=("$(mx "${kb[@]}")")
    LJ+=("$(r lj --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${T}ms" --edpp-rule least-ttft --edpp-joint)")
    VV+=("$(r dpvar "${VVF[@]}")")
  done
  printf "%-13s %-4s %-6s| %-7s %-7s %-7s %-7s %-7s\n" \
    "$NAME" "$LT" "$R" "$(mean "${NV[@]}")" "$(mean "${A[@]}")" "$(mean "${K[@]}")" "$(mean "${LJ[@]}")" "$(mean "${VV[@]}")"
  printf "%-13s %-4s %-6s| %-7s %-7s %-7s %-7s %-7s (worst seed)\n" \
    "" "" "" "$(mn "${NV[@]}")" "$(mn "${A[@]}")" "$(mn "${K[@]}")" "$(mn "${LJ[@]}")" "$(mn "${VV[@]}")"
}

printf "%-13s %-4s %-6s| %-7s %-7s %-7s %-7s %-7s\n" \
  "cell" "load" "rate" "never" "always" "kairos*" "lt-joint" "dpvar"

while read -r NAME IN O TK KNEE BT BI BE; do
  [[ "$NAME" =~ ^# ]] && continue
  for LOAD in 0.7 1.0; do
    R=$(python3 -c "print(round($KNEE*$LOAD,2))")
    T=$(python3 -c "print(round(5*$BT))"); I=$(python3 -c "print(round(4*$BI))"); E=$(python3 -c "print(round(3*$BE))")
    N=$(python3 -c "import math;print(max(int(math.ceil($R*max(120, 10*$E/1000.0))), 600))")
    run_cell "$NAME" "$IN" "$O" "$TK" "$R" "$T" "$I" "$E" "$N" "$LOAD"
  done
done < "$CELLS_FILE"
echo "done -> $OUT"
