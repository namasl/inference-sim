#!/usr/bin/env bash
# Topology matrix, protocol v2.1 (2026-07-28) — replaces repro_topology_matrix_gp.sh
# (auto_slo floors + fixed burst horizons; see memory edpp-eval-audit-2026-07-27).
#
# Fixed 4-instance fleet (16 accelerators at TP4) provisioned as 1P3M / 2P2M / 3P1M,
# homogeneous H100. Four workload archetypes. Per (topology, archetype):
#   - baseline probe at rate 0.2 -> derived SLOs (5x ttft p99, 4x itl p99, 3x e2e p99)
#   - capacity knee measured as max(always, never) mini-sweeps (achieved/offered >= 0.95)
#   - arms at the knee, 5 seeds, steady-state n = rate x max(120 s, 10x e2e SLO)
#   - one scorer config (queue-depth:1), batch cap 16
# Usage: bash campaigns/edpp-study/repro_topology_v2.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/topo_v2
OUT="${OUT:-campaigns/edpp-study/out/topo_v2}"
SEEDS="${SEEDS:-42 7 123 5 11}"
KBETAS="${KBETAS:-0.25 0.5 1.0}"
TOPOS="${TOPOS:-1P3M 2P2M 3P1M}"
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

gp(){ python3 -c "import json;print('%.4f'%json.load(open('$1'))['per_class']['standard']['slo_attainment'])" 2>/dev/null||echo NA; }
ach(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['responses_per_sec'])" 2>/dev/null||echo 0; }
val(){ python3 -c "import json;print('%.2f'%json.load(open('$1'))['$2'])" 2>/dev/null||echo 0; }
m(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }

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

topo_flags(){ case $1 in
  1P3M) echo "--num-instances 4 --prefill-instances 1 --decode-instances 3" ;;
  2P2M) echo "--num-instances 4 --prefill-instances 2 --decode-instances 2" ;;
  3P1M) echo "--num-instances 4 --prefill-instances 3 --decode-instances 1" ;;
esac; }

# archetype: name in out knee_ladder
ARCHS="decode 256 512 |2 3 4 5 6 7
mixed 2048 128 |8 12 16 18 20 24
prefill_lean 8192 64 |8 12 16 18 20 24
prefill_bound 16000 16 |4 6 8 10 12 16"

CSV="$OUT/topo_v2.csv"; echo "topology,archetype,knee,slo_ttft,slo_itl,slo_e2e,arm,seed,goodput" > "$CSV"
emit(){ echo "$1,$2,$3,$4,$5,$6,$7,$8,$9" >> "$CSV"; }

printf "%-5s %-14s %-5s | %-7s %-7s %-7s %-8s %-7s\n" \
  "topo" "archetype" "knee" "never" "always" "kairos*" "lt-joint" "dpvar"

for TP in $TOPOS; do
  TF="$(topo_flags $TP) --decode-routing-scorers queue-depth:1 --max-num-running-reqs 16"
  while IFS='|' read -r meta ladder; do
    set -- $meta; NAME=$1; IN=$2; O=$3
    # --- baseline probe ---
    spec "$IN" "$O" 0.2 60 42 "$D/wb.yaml"
    ./blis run --model "$MODEL" --workload-spec "$D/wb.yaml" ${TF} \
      --slo-ttft standard=9999s --slo-itl standard=9999s --slo-e2e standard=9999s \
      --pd-decider always --seed 42 --metrics-path "$OUT/base_${TP}_${NAME}.json" >/dev/null 2>&1
    BT=$(val "$OUT/base_${TP}_${NAME}.json" ttft_p99_ms); BI=$(val "$OUT/base_${TP}_${NAME}.json" itl_p99_ms); BE=$(val "$OUT/base_${TP}_${NAME}.json" e2e_p99_ms)
    T=$(python3 -c "print(round(5*$BT))"); I=$(python3 -c "print(round(4*$BI))"); E=$(python3 -c "print(round(3*$BE))")
    # --- knee: max over corners, achieved/offered >= 0.95 ---
    KNEE=""
    for R in $ladder; do
      ok_any=0
      for corner in always never; do
        spec "$IN" "$O" "$R" 600 42 "$D/wk.yaml"
        ./blis run --model "$MODEL" --workload-spec "$D/wk.yaml" ${TF} \
          --slo-ttft standard=9999s --slo-itl standard=9999s --slo-e2e standard=9999s \
          --pd-decider $corner --seed 42 --metrics-path "$OUT/knee_${TP}_${NAME}_${corner}_$R.json" >/dev/null 2>&1 || true
        A=$(ach "$OUT/knee_${TP}_${NAME}_${corner}_$R.json")
        okc=$(python3 -c "print(1 if $A >= 0.95*$R else 0)"); [[ "$okc" == 1 ]] && ok_any=1
      done
      [[ "$ok_any" == 1 ]] && KNEE=$R || break
    done
    [[ -n "$KNEE" ]] || { set -- $ladder; KNEE=$1; }
    RATE=$KNEE
    NREQ=$(python3 -c "import math;print(max(int(math.ceil($RATE*max(120, 10*$E/1000.0))), 600))")
    SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
    EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
    VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")
    r(){ local tg=$1 s=$2 wf=$3; shift 3
      ./blis run --model "$MODEL" --workload-spec "$wf" ${TF} "${SLO[@]}" "$@" --seed "$s" --metrics-path "$OUT/${TP}_${NAME}_${tg}_$s.json" >/dev/null 2>&1 || true
      gp "$OUT/${TP}_${NAME}_${tg}_$s.json"; }
    declare -a NV=() A=() K=() LJ=() VV=()
    for s in $SEEDS; do
      WF="$D/w_${TP}_${NAME}_$s.yaml"; spec "$IN" "$O" "$RATE" "$NREQ" "$s" "$WF"
      nv=$(r nv "$s" "$WF" --pd-decider never);  NV+=("$nv"); emit "$TP" "$NAME" "$KNEE" "$T" "$I" "$E" never "$s" "$nv"
      a=$(r a "$s" "$WF" --pd-decider always);   A+=("$a");   emit "$TP" "$NAME" "$KNEE" "$T" "$I" "$E" always "$s" "$a"
      kb=(); for bb in $KBETAS; do kb+=("$(r k$bb "$s" "$WF" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${T}ms" --edpp-rule kairos --kairos-beta $bb)"); done
      k=$(printf '%s\n' "${kb[@]}" | sort -g | tail -1); K+=("$k"); emit "$TP" "$NAME" "$KNEE" "$T" "$I" "$E" kairos "$s" "$k"
      lj=$(r lj "$s" "$WF" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${T}ms" --edpp-rule least-ttft --edpp-joint); LJ+=("$lj"); emit "$TP" "$NAME" "$KNEE" "$T" "$I" "$E" lt-joint "$s" "$lj"
      vv=$(r vv "$s" "$WF" "${VVF[@]}"); VV+=("$vv"); emit "$TP" "$NAME" "$KNEE" "$T" "$I" "$E" dpvar "$s" "$vv"
    done
    printf "%-5s %-14s %-5s | %-7s %-7s %-7s %-8s %-7s\n" \
      "$TP" "$NAME" "$KNEE" "$(m "${NV[@]}")" "$(m "${A[@]}")" "$(m "${K[@]}")" "$(m "${LJ[@]}")" "$(m "${VV[@]}")"
  done <<< "$ARCHS"
done
echo "done -> $CSV"
