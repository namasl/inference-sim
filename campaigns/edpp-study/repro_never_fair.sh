#!/usr/bin/env bash
# Fairness control for the "never" policy.
#
# In repro_var_dominance_goodput.sh every policy runs on a 1P2D fleet
# (--prefill-instances 1 --decode-instances 2). "never" always collocates, so it
# routes only to the decode pool and leaves the dedicated prefill instance idle
# (verified: instance_0 injected=0). It therefore runs on 2 of 3 instances while
# the disaggregating policies use all 3.
#
# This script reruns "never" on an equal-hardware ALL-MIXED fleet
# (--prefill-decode-instances 3: three shared-role instances, each prefill+decode
# capable, no dedicated prefill node to strand). Same model, coeffs, workload,
# seeds, output-length variance, and SLO bars as the main campaign, so the goodput
# numbers drop straight into tab:grid.
#
# Per cell it reports:
#   never-1P2D    "never" on the paper's fleet (reproduces the tab:grid never column)
#   never-3mix-qd "never" on 3 shared instances, queue-depth scorer (isolates topology)
#   never-3mix-la "never" on 3 shared instances, load-aware scorer
#   always-1P2D   disaggregating reference on the paper's fleet
#   gpdep-1P2D    deployable drift-plus-VaR champion on the paper's fleet
#
# The SLO bar is derived exactly as the main campaign (auto_slo on a 1P2D "always"
# low-load run), so every arm here is scored against the identical bar.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/never_fair
OUT="${OUT:-campaigns/edpp-study/out/never_fair}"
SEEDS="${SEEDS:-42 7 123}"
VAROUT="${VAROUT:-0.4}"
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['$2']['slo_attainment'])" 2>/dev/null||echo NA; }
val(){ python3 -c "import json;print('%.0f'%json.load(open('$1'))['$2'])" 2>/dev/null||echo 0; }
mean(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }
mn(){   python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%min(xs) if xs else 'NA')" "$@"; }

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

# Paper fleet (1 dedicated prefill + 2 decode) and the equal-hardware all-mixed fleet.
# Space-split strings (each token is space-free) so this runs under macOS bash 3.2 (no namerefs).
TOPO_1P2D="--num-instances 3 --prefill-instances 1 --decode-instances 2 --max-num-running-reqs 16"
TOPO_3MIX="--num-instances 3 --prefill-decode-instances 3 --max-num-running-reqs 16"
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl 100ms)
VVFLAGS=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable)

# auto_slo: derive the SLO bar exactly as the main campaign — a low-load 1P2D "always" run.
auto_slo(){ hspec "$1" "$2" 0.5 42
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" $TOPO_1P2D --decode-routing-scorers "queue-depth:1" --pd-decider always \
    --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),1000))")
}

# run <tag> <topo-flags-string> <scorer> <decider-args...>
run(){ local tag="$1" topo="$2" scorer="$3"; shift 3
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" $topo --decode-routing-scorers "$scorer" "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=100ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json" standard
}

echo "NEVER FAIRNESS CONTROL  variable output sigma=$VAROUT  seeds=[$SEEDS]" >&2
printf "%-14s %-5s| %-12s %-14s %-14s %-12s %-12s\n" \
  "archetype" "rate" "never-1P2D" "never-3mix-qd" "never-3mix-la" "always-1P2D" "gpdep-1P2D" >&2

for cell in "decode 256 512 16" "mixed 2048 128 16" "prefill_lean 8192 64 16" "prefill_bound 16000 16 8"; do
  set -- $cell; NAME=$1; IN=$2; O=$3; R=$4
  auto_slo "$IN" "$O"
  declare -a N1=() N3Q=() N3L=() A1=() G1=()
  for s in $SEEDS; do
    hspec "$IN" "$O" "$R" "$s"
    N1+=( "$(run  n1  "$TOPO_1P2D" "queue-depth:1" --pd-decider never  --seed $s)" )
    N3Q+=("$(run  n3q "$TOPO_3MIX" "queue-depth:1" --pd-decider never  --seed $s)" )
    N3L+=("$(run  n3l "$TOPO_3MIX" "load-aware:1"  --pd-decider never  --seed $s)" )
    A1+=( "$(run  a1  "$TOPO_1P2D" "queue-depth:1" --pd-decider always --seed $s)" )
    G1+=( "$(run  g1  "$TOPO_1P2D" "queue-depth:1" "${VVFLAGS[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" --edpp-var-goodput --seed $s)" )
  done
  printf "%-14s %-5s| %-12s %-14s %-14s %-12s %-12s\n" \
    "$NAME" "$R" "$(mean "${N1[@]}")" "$(mean "${N3Q[@]}")" "$(mean "${N3L[@]}")" "$(mean "${A1[@]}")" "$(mean "${G1[@]}")" >&2
  printf "%-14s %-5s| %-12s %-14s %-14s %-12s %-12s   (min over seeds)\n" \
    "" "" "$(mn "${N1[@]}")" "$(mn "${N3Q[@]}")" "$(mn "${N3L[@]}")" "$(mn "${A1[@]}")" "$(mn "${G1[@]}")" >&2
done
echo "done -> $OUT" >&2
