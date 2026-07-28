#!/usr/bin/env bash
# PAPER-FAITHFUL SLO-SENSITIVITY SWEEP (tab:grid config).
#
# This clones repro_var_dominance.sh EXACTLY (variable lognormal output sigma=0.4
# CV≈0.42, --edpp-c-xfer-size-aware, per-archetype rates decode/mixed/lean=16 and
# prefill_bound=8, the full 7-policy set INCLUDING the real deployable drift+VaR
# rule, Kairos tuned in its favor over KBETAS, 3 seeds, the heterogeneous H100 +
# crippled-A100 cell). The ONLY thing that changes across scenarios is where the
# SLO knobs land inside the MLPerf bracket. Everything else is byte-for-byte the
# generator that produced the paper's tab:grid.
#
# Reviewer question this answers: does the worst-case-regret ranking (drift+VaR
# lowest ~0.05, fixed corners / Kairos / least-TTFT / drift+penalty high) survive
# when TTFT and ITL move to the MLPerf endpoints? We vary one knob at a time
# around the paper operating point (TTFT floor 1000ms, homogeneous ITL 100ms,
# heterogeneous ITL 500ms):
#   TTFT floor   in {450 (MLPerf interactive), 1000 (paper), 2000 (MLPerf server)}
#   homog ITL    in {40  (MLPerf interactive), 100  (paper), 200  (MLPerf server)}
#   hetero ITL   in {100, 200, 500 (paper)}   <- the regime that binds hardest
# E2E stays probe-derived (2x idle p99) on the homogeneous cells and 8s on the
# heterogeneous cell, because E2E is the length-dependent target, not a knob we
# ground in MLPerf. TTFT stays SLO-scaled (3x idle) with the swept FLOOR.
#
# The heterogeneous cell binds on E2E under saturation with a loose 500ms ITL;
# sweeping its ITL down to 100ms tests whether drift+VaR's advantage there is an
# artifact of the loose ITL (it is not -- see the report).
#
# Factored to avoid redundant runs: 5 homogeneous grids (one per distinct
# FLOOR,ITL) + 3 heterogeneous grids (one per ITL). Worst-case regret is then
# computed for each one-factor-at-a-time scenario by combining the relevant
# homogeneous rows with the relevant heterogeneous row.
#
# Usage:  bash campaigns/edpp-study/sweep_slo_faithful.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/slo_faithful
OUT="${OUT:-campaigns/edpp-study/out/slo_faithful}"
SEEDS="${SEEDS:-42 7 123}"
VAROUT="${VAROUT:-0.4}"
KBETAS="${KBETAS:-0.25 0.5 1.0}"
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['$2']['slo_attainment'])" 2>/dev/null||echo NA; }
val(){ python3 -c "import json;print('%.0f'%json.load(open('$1'))['$2'])" 2>/dev/null||echo 0; }
mean(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }
mx(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%max(xs) if xs else 'NA')" "$@"; }

outdist(){ if [ "$VAROUT" = "0" ]; then echo "{type: constant, params: {value: $1}}"; else
  local mu; mu=$(python3 -c "import math;print('%.4f'%(math.log($1)-$VAROUT*$VAROUT/2))")
  local mmax; mmax=$(python3 -c "print(int($1*8)+16)")
  echo "{type: lognormal, params: {mu: $mu, sigma: $VAROUT, min: 4, max: $mmax}}"; fi; }

hspec(){ cat > "$D/w.yaml" <<YAML
version: "2"
seed: $4
category: language
aggregate_rate: $3
num_requests: 240
clients:
  - {id: w, tenant_id: t, slo_class: standard, rate_fraction: 1.0, streaming: false, arrival: {process: poisson}, input_distribution: {type: constant, params: {value: $1}}, output_distribution: $(outdist "$2"), prefix_group: g, prefix_length: 0}
YAML
}
HTOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 16)
# EC WITHOUT tau-itl (it is passed per-scenario so Kairos's TBT budget tracks the ITL SLO)
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware)

# auto_slo: idle probe -> SLO_E2E, SLO_TTFT. $3 = TTFT floor.
auto_slo(){ hspec "$1" "$2" 0.5 42
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" --pd-decider always \
    --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),$3))")
}
# hrun: uses SLO_TTFT/SLO_E2E and the current HITL (homogeneous ITL, ms).
hrun(){ local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=${HITL}ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json" standard
}

# ---- one homogeneous grid at (FLOOR, HITL); writes rows to $OUT/homo_<FLOOR>_<HITL>.tsv ----
homo_grid(){ local FLOOR=$1; HITL=$2
  local tsv="$OUT/homo_${FLOOR}_${HITL}.tsv"; : > "$tsv"
  for cell in "decode 256 512 16" "mixed 2048 128 16" "prefill_lean 8192 64 16" "prefill_bound 16000 16 8"; do
    set -- $cell; local NAME=$1 IN=$2 O=$3 R=$4
    auto_slo "$IN" "$O" "$FLOOR"
    local ITLFLAG=(--edpp-tau-itl "${HITL}ms")
    declare -a N=() A=() P=() K=() L=() DP=() VV=()
    for s in $SEEDS; do
      hspec "$IN" "$O" "$R" "$s"
      N+=("$(hrun n --pd-decider never --seed $s)")
      A+=("$(hrun a --pd-decider always --seed $s)")
      P+=("$(hrun p --pd-decider prefix-threshold --pd-prefix-threshold 16 --seed $s)")
      kb=(); for bb in $KBETAS; do kb+=("$(hrun k --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule kairos --kairos-beta $bb --seed $s)"); done
      K+=("$(mx "${kb[@]}")")
      L+=("$(hrun l --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft --seed $s)")
      DP+=("$(hrun dp --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-joint --seed $s)")
      VV+=("$(hrun vv --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --seed $s)")
    done
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" \
      "$(mean "${N[@]}")" "$(mean "${A[@]}")" "$(mean "${P[@]}")" "$(mean "${K[@]}")" \
      "$(mean "${L[@]}")" "$(mean "${DP[@]}")" "$(mean "${VV[@]}")" >> "$tsv"
    echo "  homo(FLOOR=$FLOOR,ITL=$HITL) $NAME done" >&2
  done
}

# ---- heterogeneous grid at XITL; writes single row to $OUT/hetero_<XITL>.tsv ----
hetero_grid(){ local XITL=$1
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
  local XT=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --policy-config "$D/bundle.yaml" --max-num-running-reqs 8)
  local XSLO=(--slo-ttft batch=60s --slo-e2e batch=8s --slo-itl "batch=${XITL}ms")
  local ITLFLAG=(--edpp-tau-itl "${XITL}ms")
  xrun(){ local tag="$1"; shift; ./blis run --model "$MODEL" --workload-spec "$D/hetero.yaml" "${XT[@]}" "${XSLO[@]}" "$@" --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true; gp "$OUT/$tag.json" batch; }
  declare -a A=() P=() K=() L=() DP=() VV=()
  for s in $SEEDS; do
    A+=("$(xrun xa --pd-decider always --seed $s)")
    P+=("$(xrun xp --pd-decider prefix-threshold --pd-prefix-threshold 16 --seed $s)")
    kb=(); for bb in $KBETAS; do kb+=("$(xrun xk --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft 60s --edpp-rule kairos --kairos-beta $bb --seed $s)"); done
    K+=("$(mx "${kb[@]}")")
    L+=("$(xrun xl --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft 60s --edpp-rule least-ttft --seed $s)")
    DP+=("$(xrun xdp --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft 60s --edpp-joint --seed $s)")
    VV+=("$(xrun xvv --pd-decider edpp "${EC[@]}" "${ITLFLAG[@]}" --edpp-tau-ttft 60s --edpp-tau-e2e 8s --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --seed $s)")
  done
  # 'never' has no meaning here (always route); use NA so it drops from row-best
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "hetero" \
    "NA" "$(mean "${A[@]}")" "$(mean "${P[@]}")" "$(mean "${K[@]}")" "$(mean "${L[@]}")" "$(mean "${DP[@]}")" "$(mean "${VV[@]}")" \
    > "$OUT/hetero_${XITL}.tsv"
  echo "  hetero(ITL=$XITL) done" >&2
}

echo "== PAPER-FAITHFUL SLO SWEEP (tab:grid config, seeds=[$SEEDS], CV≈0.42) ==" >&2
echo "== building 5 homogeneous grids + 3 heterogeneous grids ==" >&2
# distinct homogeneous settings (FLOOR ITL): paper + TTFT endpoints + ITL endpoints
homo_grid 1000 100      # paper operating point
homo_grid 450  100      # MLPerf interactive TTFT floor
homo_grid 2000 100      # MLPerf server TTFT floor
homo_grid 1000 40       # MLPerf interactive ITL
homo_grid 1000 200      # MLPerf server ITL
# heterogeneous ITL settings
hetero_grid 500         # paper
hetero_grid 200
hetero_grid 100

echo "== all grids built; computing worst-case regret per scenario ==" >&2
python3 - "$OUT" <<'PY'
import sys, os
OUT=sys.argv[1]
POLS=['never','always','prefix16','kairos','least-ttft','dpp','dpVaR']
def load(path):
    rows={}
    for l in open(path):
        f=l.rstrip('\n').split('\t');
        if not f or not f[0]: continue
        rows[f[0]]=[None if x=='NA' else float(x) for x in f[1:8]]
    return rows
def homo(fl,itl): return load(f"{OUT}/homo_{fl}_{itl}.tsv")
def het(itl):     return load(f"{OUT}/hetero_{itl}.tsv")['hetero']
def regret(all_rows):
    reg={p:0.0 for p in POLS}
    for r in all_rows:
        vals=[v for v in r if v is not None]; best=max(vals)
        for p,v in zip(POLS,r):
            if v is not None: reg[p]=max(reg[p],best-v)
    return reg
def report(title, homo_rows, het_row):
    rows=list(homo_rows.values())+[het_row]
    reg=regret(rows)
    win=min(reg,key=reg.get)
    print(f"\n### {title}")
    print("   "+"  ".join(f"{p}={reg[p]:.3f}" for p in POLS))
    print(f"   --> lowest worst-case regret: {win} ({reg[win]:.3f}); drift+VaR={reg['dpVaR']:.3f}")

print("PAPER-FAITHFUL SLO-SENSITIVITY SWEEP  (worst-case regret across all 5 regimes)")
print("baseline paper point = FLOOR 1000 / homog ITL 100 / hetero ITL 500")
report("PAPER POINT  (FLOOR=1000, homogITL=100, heteroITL=500)", homo(1000,100), het(500))
print("\n--- TTFT floor sweep (homog ITL 100, hetero ITL 500) ---")
report("TTFT floor = 450  (MLPerf interactive)", homo(450,100),  het(500))
report("TTFT floor = 1000 (paper)",             homo(1000,100), het(500))
report("TTFT floor = 2000 (MLPerf server)",     homo(2000,100), het(500))
print("\n--- homogeneous ITL sweep (TTFT floor 1000, hetero ITL 500) ---")
report("homog ITL = 40  (MLPerf interactive)", homo(1000,40),  het(500))
report("homog ITL = 100 (paper)",              homo(1000,100), het(500))
report("homog ITL = 200 (MLPerf server)",      homo(1000,200), het(500))
print("\n--- heterogeneous ITL sweep (TTFT floor 1000, homog ITL 100) ---")
report("hetero ITL = 100", homo(1000,100), het(100))
report("hetero ITL = 200", homo(1000,100), het(200))
report("hetero ITL = 500 (paper)", homo(1000,100), het(500))
PY
echo "done -> $OUT" >&2
