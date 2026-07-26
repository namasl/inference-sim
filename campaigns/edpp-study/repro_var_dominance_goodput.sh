#!/usr/bin/env bash
# repro_var_dominance.sh + the goodput-objective arms. Full adaptive-dominance grid with two
# extra arms alongside the deployable dpVaR "vv" baseline:
#   vg  = dpVaR + --edpp-var-goodput --edpp-oracle-output-len  (goodput objective, ORACLE upper bound)
#   vgo = dpVaR + --edpp-var-goodput                           (goodput objective, DEPLOYABLE N̂_out)
# All other arms (never/always/prefix16/kairos*/least-ttft/lt-joint/dpp/dpVaR) are unchanged from
# repro_var_dominance.sh so the regret denominator (per-cell best baseline) is identical.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/var_dom_gp
OUT="${OUT:-campaigns/edpp-study/out/var_dom_gp}"
SEEDS="${SEEDS:-42 7 123}"
VAROUT="${VAROUT:-0.4}"
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['$2']['slo_attainment'])" 2>/dev/null||echo NA; }
val(){ python3 -c "import json;print('%.0f'%json.load(open('$1'))['$2'])" 2>/dev/null||echo 0; }
mean(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }
mn(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%min(xs) if xs else 'NA')" "$@"; }
mx(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%max(xs) if xs else 'NA')" "$@"; }
KBETAS="${KBETAS:-0.25 0.5 1.0}"

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
HTOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 16)
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl 100ms)

auto_slo(){ hspec "$1" "$2" 0.5 42
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" --pd-decider always \
    --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),1000))")
}
hrun(){ local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=100ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json" standard
}
# shared dpVaR flags for the vv/vg/vgo arms
VVFLAGS=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable)

echo "ADAPTIVE-DOMINANCE + GOODPUT  variable output sigma=$VAROUT  seeds=[$SEEDS]" >&2
printf "%-14s %-5s| %-8s %-8s %-8s %-8s %-10s %-10s %-8s %-8s %-10s %-10s\n" \
  "archetype" "rate" "never" "always" "prefix16" "kairos*" "least-ttft" "lt-joint" "dpp" "dpVaR" "gp+orc" "gp-dep" >&2

for cell in "decode 256 512 16" "mixed 2048 128 16" "prefill_lean 8192 64 16" "prefill_bound 16000 16 8"; do
  set -- $cell; NAME=$1; IN=$2; O=$3; R=$4
  auto_slo "$IN" "$O"
  declare -a N=() A=() P=() K=() L=() LJ=() DP=() VV=() VG=() VGO=()
  for s in $SEEDS; do
    hspec "$IN" "$O" "$R" "$s"
    N+=("$(hrun n --pd-decider never --seed $s)")
    A+=("$(hrun a --pd-decider always --seed $s)")
    P+=("$(hrun p --pd-decider prefix-threshold --pd-prefix-threshold 16 --seed $s)")
    kb=(); for bb in $KBETAS; do kb+=("$(hrun k --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule kairos --kairos-beta $bb --seed $s)") ; done
    K+=("$(mx "${kb[@]}")")
    L+=("$(hrun l --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft --seed $s)")
    LJ+=("$(hrun lj --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft --edpp-joint --seed $s)")
    DP+=("$(hrun dp --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-joint --seed $s)")
    VV+=("$(hrun vv "${VVFLAGS[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" --seed $s)")
    VG+=("$(hrun vg "${VVFLAGS[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" --edpp-var-goodput --edpp-oracle-output-len --seed $s)")
    VGO+=("$(hrun vgo "${VVFLAGS[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" --edpp-var-goodput --seed $s)")
  done
  printf "%-14s %-5s| %-8s %-8s %-8s %-8s %-10s %-10s %-8s %-8s %-10s %-10s\n" \
    "$NAME" "$R" "$(mean "${N[@]}")" "$(mean "${A[@]}")" "$(mean "${P[@]}")" "$(mean "${K[@]}")" \
    "$(mean "${L[@]}")" "$(mean "${LJ[@]}")" "$(mean "${DP[@]}")" "$(mean "${VV[@]}")" "$(mean "${VG[@]}")" "$(mean "${VGO[@]}")" >&2
  printf "%-14s %-5s| %-8s %-8s %-8s %-8s %-10s %-10s %-8s %-8s %-10s %-10s   (min over seeds)\n" \
    "" "" "$(mn "${N[@]}")" "$(mn "${A[@]}")" "$(mn "${P[@]}")" "$(mn "${K[@]}")" \
    "$(mn "${L[@]}")" "$(mn "${LJ[@]}")" "$(mn "${DP[@]}")" "$(mn "${VV[@]}")" "$(mn "${VG[@]}")" "$(mn "${VGO[@]}")" >&2
done

# --- heterogeneous cell ---
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
declare -a A=() P=() K=() L=() LJ=() DP=() VV=() VG=() VGO=()
for s in $SEEDS; do
  A+=("$(xrun xa --pd-decider always --seed $s)")
  P+=("$(xrun xp --pd-decider prefix-threshold --pd-prefix-threshold 16 --seed $s)")
  kb=(); for bb in $KBETAS; do kb+=("$(xrun xk --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-rule kairos --kairos-beta $bb --seed $s)") ; done
  K+=("$(mx "${kb[@]}")")
  L+=("$(xrun xl --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-rule least-ttft --seed $s)")
  LJ+=("$(xrun xlj --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-rule least-ttft --edpp-joint --seed $s)")
  DP+=("$(xrun xdp --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-joint --seed $s)")
  VV+=("$(xrun xvv --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-tau-e2e 8s --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --seed $s)")
  VG+=("$(xrun xvg --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-tau-e2e 8s --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-var-goodput --edpp-oracle-output-len --seed $s)")
  VGO+=("$(xrun xvgo --pd-decider edpp "${EC[@]}" --edpp-tau-ttft 60s --edpp-tau-itl 500ms --edpp-tau-e2e 8s --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-var-goodput --seed $s)")
done
printf "%-14s %-5s| %-8s %-8s %-8s %-8s %-10s %-10s %-8s %-8s %-10s %-10s\n" \
  "heterogeneous" "10" "-" "$(mean "${A[@]}")" "$(mean "${P[@]}")" "$(mean "${K[@]}")" "$(mean "${L[@]}")" "$(mean "${LJ[@]}")" "$(mean "${DP[@]}")" "$(mean "${VV[@]}")" "$(mean "${VG[@]}")" "$(mean "${VGO[@]}")" >&2
printf "%-14s %-5s| %-8s %-8s %-8s %-8s %-10s %-10s %-8s %-8s %-10s %-10s   (min over seeds)\n" \
  "" "" "-" "$(mn "${A[@]}")" "$(mn "${P[@]}")" "$(mn "${K[@]}")" "$(mn "${L[@]}")" "$(mn "${LJ[@]}")" "$(mn "${DP[@]}")" "$(mn "${VV[@]}")" "$(mn "${VG[@]}")" "$(mn "${VGO[@]}")" >&2
echo "done -> $OUT" >&2
