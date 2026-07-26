#!/usr/bin/env bash
# Adaptive-dominance grid: is DEPLOYABLE drift-plus-VaR never the loser?
# Tests the adaptivity story — one rule you deploy blind that matches the regime-specific best
# across decode-bound / balanced / prefill-heavy (homogeneous) AND heterogeneous hardware, on
# REALISTIC variable-output workloads (lognormal, moderate CV). Each simple baseline wins one
# regime and is catastrophic in another; the claim is dpVaR ≥ max(simple baselines) − ε in EVERY
# cell, with no per-seed craters.
#
# Arms: never | always | least-ttft (reduced) | lt-joint (least-ttft over the full joint action
#   set: --edpp-rule least-ttft --edpp-joint; the fair hardware-aware least-TTFT) | dpp (joint edpp)
#   | dpVaR (joint, deployable:
#   --edpp-rule var --edpp-var-metric util --edpp-var-congestion --edpp-var-normalize
#   --edpp-var-congestion-weight 1 --edpp-var-deployable). NO oracle flags — this is the runnable rule.
# Output length is VARIABLE: lognormal with sigma=$VAROUT (default 0.4, CV≈0.42), mean = archetype value.
#
# Usage:  bash campaigns/edpp-study/repro_var_dominance.sh        (VAROUT=0.4 SEEDS="42 7 123")
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/var_dom
OUT="${OUT:-campaigns/edpp-study/out/var_dom}"
SEEDS="${SEEDS:-42 7 123}"
VAROUT="${VAROUT:-0.4}"       # output-length lognormal sigma (0 ⇒ constant)
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['$2']['slo_attainment'])" 2>/dev/null||echo NA; }
val(){ python3 -c "import json;print('%.0f'%json.load(open('$1'))['$2'])" 2>/dev/null||echo 0; }
mean(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }
mn(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%min(xs) if xs else 'NA')" "$@"; }
mx(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%max(xs) if xs else 'NA')" "$@"; }
# Kairos is tuned IN ITS FAVOR: its TBT safety margin beta is swept and the BEST result per
# seed is taken, so the baseline is reported at its strongest rather than at an arbitrary beta.
KBETAS="${KBETAS:-0.25 0.5 1.0}"

# output_distribution spec: constant when VAROUT=0, else lognormal(mean=$1, sigma=VAROUT).
outdist(){ if [ "$VAROUT" = "0" ]; then echo "{type: constant, params: {value: $1}}"; else
  local mu; mu=$(python3 -c "import math;print('%.4f'%(math.log($1)-$VAROUT*$VAROUT/2))")
  local mx; mx=$(python3 -c "print(int($1*8)+16)")
  echo "{type: lognormal, params: {mu: $mu, sigma: $VAROUT, min: 4, max: $mx}}"; fi; }

# homogeneous spec (1P2D, one class "standard"); $1=in $2=out(mean) $3=rate $4=seed
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
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl 100ms)

# auto_slo: idle probe (variable output, always, low rate) → SLO_E2E, SLO_TTFT (globals)
auto_slo(){ hspec "$1" "$2" 0.5 42
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" --pd-decider always \
    --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),1000))")
}
hrun(){ # $1=tag $2..=policy flags ; uses SLO_E2E/SLO_TTFT ; echoes standard goodput
  local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=100ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json" standard
}

echo "ADAPTIVE-DOMINANCE GRID  variable output sigma=$VAROUT (CV≈$(python3 -c "import math;print('%.2f'%math.sqrt(math.exp($VAROUT*$VAROUT)-1))"))  seeds=[$SEEDS]" >&2
echo "cell = mean(min) over seeds. WIN test: dpVaR ≥ max(never,always,least-ttft,dpp) − 0.03 AND min-seed not a crater." >&2
printf "%-14s %-5s| %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s\n" "archetype" "rate" "never" "always" "prefix16" "kairos*" "least-ttft" "lt-joint" "dpp" "dpVaR" >&2

# homogeneous archetypes: name in out rate  (rates chosen where policies separate)
for cell in "decode 256 512 16" "mixed 2048 128 16" "prefill_lean 8192 64 16" "prefill_bound 16000 16 8"; do
  set -- $cell; NAME=$1; IN=$2; O=$3; R=$4
  auto_slo "$IN" "$O"
  declare -a N=() A=() P=() K=() L=() LJ=() DP=() VV=()
  for s in $SEEDS; do
    hspec "$IN" "$O" "$R" "$s"
    N+=("$(hrun n --pd-decider never --seed $s)")
    A+=("$(hrun a --pd-decider always --seed $s)")
    # llm-d shipped PD decider (deploy/config/pd-epp-config.yaml): disaggregate iff uncached > 16
    P+=("$(hrun p --pd-decider prefix-threshold --pd-prefix-threshold 16 --seed $s)")
    # Kairos (arXiv:2607.02043) — SOTA load-aware prefill deflection; TBT budget = the ITL SLO
    kb=(); for bb in $KBETAS; do kb+=("$(hrun k --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule kairos --kairos-beta $bb --seed $s)") ; done
    K+=("$(mx "${kb[@]}")")
    L+=("$(hrun l --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft --seed $s)")
    # lt-joint: least-TTFT scored over the full (decode, prefill) action set (hardware-aware).
    LJ+=("$(hrun lj --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft --edpp-joint --seed $s)")
    DP+=("$(hrun dp --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-joint --seed $s)")
    VV+=("$(hrun vv --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --seed $s)")
  done
  printf "%-14s %-5s| %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s\n" \
    "$NAME" "$R" "$(mean "${N[@]}")" "$(mean "${A[@]}")" "$(mean "${P[@]}")" "$(mean "${K[@]}")" \
    "$(mean "${L[@]}")" "$(mean "${LJ[@]}")" "$(mean "${DP[@]}")" "$(mean "${VV[@]}")" >&2
  printf "%-14s %-5s| %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s   (min over seeds)\n" \
    "" "" "$(mn "${N[@]}")" "$(mn "${A[@]}")" "$(mn "${P[@]}")" "$(mn "${K[@]}")" \
    "$(mn "${L[@]}")" "$(mn "${LJ[@]}")" "$(mn "${DP[@]}")" "$(mn "${VV[@]}")" >&2
done

# --- heterogeneous cell (fast H100 + crippled-A100 decode), variable output ---
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
xrun(){ local tag="$1"; shift; ./blis run --model "$MODEL" --workload-spec "$D/hetero.yaml" "${XT[@]}" "${XSLO[@]}" "$@" --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true; gp "$OUT/$tag.json" batch; }
declare -a A=() P=() K=() L=() LJ=() DP=() VV=()
for s in $SEEDS; do
  A+=("$(xrun xa --pd-decider always --seed $s)")
  P+=("$(xrun xp --pd-decider prefix-threshold --pd-prefix-threshold 16 --seed $s)")
  kb=(); for bb in $KBETAS; do kb+=("$(xrun xk --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-rule kairos --kairos-beta $bb --seed $s)") ; done
  K+=("$(mx "${kb[@]}")")
  L+=("$(xrun xl --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-rule least-ttft --seed $s)")
  # lt-joint: least-TTFT over the full joint action set — the reviewer's fair least-TTFT on
  # heterogeneous hardware (each candidate scored under its own θ_i).
  LJ+=("$(xrun xlj --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-rule least-ttft --edpp-joint --seed $s)")
  DP+=("$(xrun xdp --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-joint --seed $s)")
  VV+=("$(xrun xvv --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-tau-e2e 8s --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --seed $s)")
done
printf "%-14s %-5s| %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s\n" \
  "heterogeneous" "10" "-" "$(mean "${A[@]}")" "$(mean "${P[@]}")" "$(mean "${K[@]}")" "$(mean "${L[@]}")" "$(mean "${LJ[@]}")" "$(mean "${DP[@]}")" "$(mean "${VV[@]}")" >&2
printf "%-14s %-5s| %-9s %-9s %-9s %-9s %-11s %-11s %-9s %-9s   (min over seeds)\n" \
  "" "" "-" "$(mn "${A[@]}")" "$(mn "${P[@]}")" "$(mn "${K[@]}")" "$(mn "${L[@]}")" "$(mn "${LJ[@]}")" "$(mn "${DP[@]}")" "$(mn "${VV[@]}")" >&2
echo "done -> $OUT" >&2
