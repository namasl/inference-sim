#!/usr/bin/env bash
# TOPOLOGY & PROVISIONING matrix (structural ablation, experiment 2).
# Design: docs/superpowers/specs/2026-07-23-edpp-structural-ablations-design.md
#
# ONE run matrix, TWO paper framings sliced from it:
#   A) Topology robustness  -- worst-case regret per topology across workload
#      archetypes. Tests whether the sec:formulation:norm cancel/bind
#      normalization (one weight, no regime selection) survives a change of
#      FLEET SHAPE. Discharges the sec:eval:threats "one topology per regime".
#   B) P:D provisioning adaptivity -- fix a workload, plot goodput vs provisioned
#      P:D ratio; show drift+VaR tracks the per-provisioning best WITHOUT
#      retuning, i.e. adapts the effective split online (vs TaiChi's offline,
#      minutes-scale reconfiguration; sec:related).
#
# GPU-MATCHED: every topology uses --num-instances 4 (4 instances x TP=4 = 16
# GPUs), so 1P3D / 2P2D / 3P1D differ ONLY in the prefill:decode provisioning
# split, not in total compute. Homogeneous hardware (H100 coeffs) throughout --
# no node-pool bundle, so the first-fit placement fragility does not apply here;
# the heterogeneous ratio story is experiment 1 (repro_hetero_ratio_sweep.sh).
#
# Reuses repro_var_dominance.sh's arm set (7, deployable), auto-derived SLOs
# (idle probe x3 TTFT / x2 E2E), variable output (lognormal CV~0.42), and
# best-per-seed Kairos beta tuning.
#
# Usage:
#   bash campaigns/edpp-study/repro_topology_matrix.sh
#   TOPOS="1P3D:1:3 2P2D:2:2 3P1D:3:1" SEEDS="42 7 123" \
#     bash campaigns/edpp-study/repro_topology_matrix.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/topo_matrix
OUT="${OUT:-campaigns/edpp-study/out/topo_matrix}"
SEEDS="${SEEDS:-42 7 123}"
VAROUT="${VAROUT:-0.4}"
KBETAS="${KBETAS:-0.25 0.5 1.0}"
TOPOS="${TOPOS:-1P3D:1:3 2P2D:2:2 3P1D:3:1}"   # label:prefill:decode (num-instances = P+D = 4)
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

gp(){ python3 -c "import json;print('%.4f'%json.load(open('$1'))['per_class']['standard']['slo_attainment'])" 2>/dev/null||echo NA; }
val(){ python3 -c "import json;print('%.0f'%json.load(open('$1'))['$2'])" 2>/dev/null||echo 0; }

outdist(){ if [ "$VAROUT" = "0" ]; then echo "{type: constant, params: {value: $1}}"; else
  local mu; mu=$(python3 -c "import math;print('%.4f'%(math.log($1)-$VAROUT*$VAROUT/2))")
  local mx; mx=$(python3 -c "print(int($1*8)+16)")
  echo "{type: lognormal, params: {mu: $mu, sigma: $VAROUT, min: 4, max: $mx}}"; fi; }
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

EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl 100ms)
# topology flags for the current (P,D); num-instances = P+D.
topo_flags(){ local P="$1" Dn="$2"
  echo "--num-instances $((P+Dn)) --prefill-instances $P --decode-instances $Dn --decode-routing-scorers queue-depth:1 --max-num-running-reqs 16"
}
SLO_E2E=0; SLO_TTFT=0
auto_slo(){ local P="$1" Dn="$2" in="$3" out="$4"; hspec "$in" "$out" 0.5 42
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" $(topo_flags "$P" "$Dn") --pd-decider always \
    --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),1000))")
}
hrun(){ local tag="$1" P="$2" Dn="$3"; shift 3
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" $(topo_flags "$P" "$Dn") "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=100ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json"; }

CSV="$OUT/topo_matrix.csv"; echo "topology,prefill,decode,archetype,rate,arm,seed,goodput" > "$CSV"
emit(){ echo "$1,$2,$3,$4,$5,$6,$7,$8" >> "$CSV"; }
m(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }

echo "TOPOLOGY & PROVISIONING MATRIX  GPU-matched (4 inst)  seeds=[$SEEDS]  out-CV~0.42  topos=[$TOPOS]" >&2
# homogeneous archetypes: name in out rate (same as repro_var_dominance.sh)
ARCHS="decode:256:512:16 mixed:2048:128:16 prefill_lean:8192:64:16 prefill_bound:16000:16:8"

for topo in $TOPOS; do
  LBL="${topo%%:*}"; P="$(echo "$topo"|cut -d: -f2)"; Dn="$(echo "$topo"|cut -d: -f3)"
  echo "== topology $LBL (${P}P${Dn}D) ==" >&2
  printf "%-14s %-5s| %-8s %-8s %-8s %-8s %-8s %-9s %-8s %-8s\n" "archetype" "rate" "never" "always" "prefix16" "kairos*" "leastT" "leastT-J" "dpp" "dpVaR" >&2
  for cell in $ARCHS; do
    NAME="${cell%%:*}"; IN="$(echo "$cell"|cut -d: -f2)"; O="$(echo "$cell"|cut -d: -f3)"; R="$(echo "$cell"|cut -d: -f4)"
    auto_slo "$P" "$Dn" "$IN" "$O"
    declare -a N=() A=() PX=() K=() L=() LJ=() DP=() VV=()
    for s in $SEEDS; do
      hspec "$IN" "$O" "$R" "$s"
      n=$(hrun n "$P" "$Dn"  --pd-decider never --seed $s);                       N+=("$n");  emit "$LBL" "$P" "$Dn" "$NAME" "$R" never "$s" "$n"
      a=$(hrun a "$P" "$Dn"  --pd-decider always --seed $s);                      A+=("$a");  emit "$LBL" "$P" "$Dn" "$NAME" "$R" always "$s" "$a"
      p=$(hrun p "$P" "$Dn"  --pd-decider prefix-threshold --pd-prefix-threshold 16 --seed $s); PX+=("$p"); emit "$LBL" "$P" "$Dn" "$NAME" "$R" prefix "$s" "$p"
      kb=(); for bb in $KBETAS; do kb+=("$(hrun k "$P" "$Dn" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule kairos --kairos-beta $bb --seed $s)"); done
      k=$(printf '%s\n' "${kb[@]}"|sort -g|tail -1);                              K+=("$k");  emit "$LBL" "$P" "$Dn" "$NAME" "$R" kairos "$s" "$k"
      l=$(hrun l "$P" "$Dn"  --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft --seed $s); L+=("$l"); emit "$LBL" "$P" "$Dn" "$NAME" "$R" least-ttft "$s" "$l"
      # lt-joint: least-TTFT over the full joint action set (hardware-aware least-TTFT).
      lj=$(hrun lj "$P" "$Dn" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft --edpp-joint --seed $s); LJ+=("$lj"); emit "$LBL" "$P" "$Dn" "$NAME" "$R" lt-joint "$s" "$lj"
      dp=$(hrun dp "$P" "$Dn" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-joint --seed $s); DP+=("$dp"); emit "$LBL" "$P" "$Dn" "$NAME" "$R" dpp "$s" "$dp"
      vv=$(hrun vv "$P" "$Dn" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-var-goodput --seed $s); VV+=("$vv"); emit "$LBL" "$P" "$Dn" "$NAME" "$R" dpVaR "$s" "$vv"
    done
    printf "%-14s %-5s| %-8s %-8s %-8s %-8s %-8s %-9s %-8s %-8s\n" "$NAME" "$R" \
      "$(m "${N[@]}")" "$(m "${A[@]}")" "$(m "${PX[@]}")" "$(m "${K[@]}")" "$(m "${L[@]}")" "$(m "${LJ[@]}")" "$(m "${DP[@]}")" "$(m "${VV[@]}")" >&2
  done
done
echo "done -> $CSV" >&2