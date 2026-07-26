#!/usr/bin/env bash
# MERIT SMOKE TEST for the goodput-objective reframing (--edpp-var-goodput).
#
# Question: does reframing the executed rule from "minimize transfer cost" to "maximize goodput"
# (charge VaR − good_r, drop the standalone transfer penalty) beat the current transfer-cost
# objective — GIVEN PERFECT output-length knowledge (--edpp-oracle-output-len)? This is the UPPER
# BOUND. If even the oracle-fed goodput arm does not beat the plain deployable dpVaR, the reframe
# is narrative-only and we stop (fall back to Option A). If it helps, it justifies building the
# INV-9-safe censored good_r and measuring the oracle→deployable gap.
#
# Arms (same auto-derived SLOs, same topology, same seeds):
#   B  = deployable dpVaR (the headline rule, exactly the repro_var_dominance.sh "vv" arm)
#   G  = B + --edpp-var-goodput --edpp-oracle-output-len   (goodput objective, oracle upper bound)
#   Go = B + --edpp-var-goodput                            (goodput objective, DEPLOYABLE N̂_out)
#
# Metric: standard-class SLO attainment (goodput). Higher is better. We report mean and min over
# seeds per archetype so a per-seed crater is visible.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/goodput_merit
OUT="${OUT:-campaigns/edpp-study/out/goodput_merit}"
SEEDS="${SEEDS:-42 7}"
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
HTOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 16)
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl 100ms)

auto_slo(){ hspec "$1" "$2" 0.5 42
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" --pd-decider always \
    --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),1000))")
}
# dpVaR base flags (deployable, exactly the "vv" arm); extra goodput flags passed by caller
vvrun(){ local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$D/w.yaml" "${HTOPO[@]}" \
    --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-e2e "${SLO_E2E}ms" \
    --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize \
    --edpp-var-congestion-weight 1 --edpp-var-deployable \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=100ms" --slo-e2e "standard=${SLO_E2E}ms" \
    "$@" --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json" standard
}

echo "GOODPUT-MERIT SMOKE  variable output sigma=$VAROUT  seeds=[$SEEDS]" >&2
echo "B=deployable dpVaR | G=+goodput+oracle (upper bound) | Go=+goodput deployable" >&2
printf "%-14s %-5s| %-14s %-14s %-14s\n" "archetype" "rate" "B (dpVaR)" "G (goodput+orc)" "Go (goodput dep)" >&2

for cell in "decode 256 512 16" "mixed 2048 128 16" "prefill_lean 8192 64 16" "prefill_bound 16000 16 8"; do
  set -- $cell; NAME=$1; IN=$2; O=$3; R=$4
  auto_slo "$IN" "$O"
  declare -a B=() G=() GO=()
  for s in $SEEDS; do
    hspec "$IN" "$O" "$R" "$s"
    B+=("$(vvrun b --seed $s)")
    G+=("$(vvrun g --seed $s --edpp-var-goodput --edpp-oracle-output-len)")
    GO+=("$(vvrun go --seed $s --edpp-var-goodput)")
  done
  printf "%-14s %-5s| %-6s(%-6s) %-6s(%-6s) %-6s(%-6s)\n" \
    "$NAME" "$R" "$(mean "${B[@]}")" "$(mn "${B[@]}")" "$(mean "${G[@]}")" "$(mn "${G[@]}")" "$(mean "${GO[@]}")" "$(mn "${GO[@]}")" >&2
done
echo "done -> $OUT   (format: mean(min) over seeds)" >&2
