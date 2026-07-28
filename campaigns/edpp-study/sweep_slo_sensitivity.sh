#!/usr/bin/env bash
# SLO-SENSITIVITY SWEEP. Reviewer-defense check for the eval grid: does the
# worst-case-regret RANKING (drift-plus-VaR lowest, fixed corners / least-ttft
# high) survive when the SLO knobs move to the MLPerf endpoints?
#
#   TTFT floor  in {450ms (MLPerf interactive), 1s (our value), 2s (MLPerf server)}
#   ITL target  in {40ms  (MLPerf interactive), 100ms (our value), 200ms (server)}
#
# The E2E deadline stays probe-derived (2x idle e2e_p99) throughout, because E2E
# is the length-dependent target. TTFT stays SLO-scaled (3x idle) but the FLOOR
# is the swept knob. ITL is a fixed absolute (the swept knob). This isolates the
# two absolutes the paper now grounds in MLPerf and asks whether the policy
# ranking depends on where inside the MLPerf bracket we land.
#
# For each (TTFT_FLOOR, ITL) cell it runs the 4-archetype x 5-policy 1P2D grid at
# one rate and prints goodput per archetype PLUS the worst-case regret each policy
# carries across the four archetypes. PASS = edpp holds the lowest worst-case
# regret and the corners/least-ttft stay high, in every cell.
#
# Usage (from repo root):  bash campaigns/edpp-study/sweep_slo_sensitivity.sh
#   RATE=8 bash ...                 # different load
#   FLOORS="1000" ITLS="40 100" ... # narrow the grid
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
SPECDIR="campaigns/edpp-study/specs/slosweep"
OUT="${OUT:-campaigns/edpp-study/out/slosweep}"
SEED="${SEED:-42}"
RATE="${RATE:-16}"
CAP="${CAP:-16}"
FLOORS="${FLOORS:-450 1000 2000}"
ITLS="${ITLS:-40 100 200}"
mkdir -p "$SPECDIR" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --max-num-running-reqs "$CAP")
DEC=(--decode-routing-scorers "queue-depth:1")   # balanced, as in the paper grid

arch_dims(){
  case "$1" in
    decode)        echo "256 512"   ;;
    mixed)         echo "2048 128"  ;;
    prefill_lean)  echo "8192 64"   ;;
    prefill_bound) echo "16000 16"  ;;
    *) echo "unknown archetype: $1" >&2; exit 1 ;;
  esac
}
ARCH_ORDER="decode mixed prefill_lean prefill_bound"

spec(){ # $1=in $2=out $3=rate $4=seed
  cat > "$SPECDIR/w.yaml" <<YAML
version: "2"
seed: $4
category: language
aggregate_rate: $3
num_requests: 240
clients:
  - {id: w, tenant_id: t, slo_class: standard, rate_fraction: 1.0, streaming: false, arrival: {process: poisson}, input_distribution: {type: constant, params: {value: $1}}, output_distribution: {type: constant, params: {value: $2}}, prefix_group: g, prefix_length: 0}
YAML
}
gp(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['per_class']['standard']['slo_attainment'])" 2>/dev/null || echo NA; }
val(){ python3 -c "import json;print('%.0f'%json.load(open('$1'))['$2'])" 2>/dev/null || echo 0; }

# auto_slo: probe idle tails. TTFT target = max(3x idle_ttft_p99, FLOOR); E2E = max(2x idle_e2e_p99, 1000).
auto_slo(){ # $1=in $2=out $3=floor ; sets SLO_E2E SLO_TTFT
  spec "$1" "$2" 0.5 42
  ./blis run --model "$MODEL" --workload-spec "$SPECDIR/w.yaml" "${TOPO[@]}" "${DEC[@]}" \
    --pd-decider always --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" \
    --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))")
  SLO_TTFT=$(python3 -c "print(max(int($it*3),$3))")
}

# run_policy: run one arm at the current SLO_TTFT/SLO_E2E/ITL_MS, echo goodput.
run_policy(){ # $1=tag $2..=flags
  local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$SPECDIR/w.yaml" "${TOPO[@]}" "${DEC[@]}" "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=${ITL_MS}ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json"
}

echo "SLO SENSITIVITY  1P2D cap=$CAP rate=$RATE seed=$SEED  (goodput per archetype; worst-case regret per policy)" >&2
echo "E2E fixed at probe 2x; sweeping TTFT floor x ITL absolute over the MLPerf bracket." >&2

for FL in $FLOORS; do
  for ITL_MS in $ITLS; do
    echo "" >&2
    echo "########## TTFT floor=${FL}ms   ITL=${ITL_MS}ms ##########" >&2
    printf "   %-14s| %-8s %-8s %-8s %-10s %-8s\n" "archetype" "never" "always" "prefix16" "least-ttft" "edpp" >&2
    # accumulate per-policy goodput across archetypes for the regret computation
    : > "$OUT/grid_${FL}_${ITL_MS}.tsv"
    for name in $ARCH_ORDER; do
      set -- $(arch_dims "$name"); IN=$1; O=$2
      auto_slo "$IN" "$O" "$FL"
      spec "$IN" "$O" "$RATE" "$SEED"
      EC=(--edpp-coeffs "$COEFFS" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-itl "${ITL_MS}ms" --edpp-tadm-estimator rollforward)
      N=$(run_policy n --pd-decider never)
      A=$(run_policy a --pd-decider always)
      P=$(run_policy p --pd-decider prefix-threshold --pd-prefix-threshold 16)
      L=$(run_policy l --pd-decider edpp "${EC[@]}" --edpp-rule least-ttft)
      E=$(run_policy e --pd-decider edpp "${EC[@]}")
      printf "   %-14s| %-8s %-8s %-8s %-10s %-8s  (slo ttft=%sms e2e=%sms)\n" \
        "$name" "$N" "$A" "$P" "$L" "$E" "$SLO_TTFT" "$SLO_E2E" >&2
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$N" "$A" "$P" "$L" "$E" >> "$OUT/grid_${FL}_${ITL_MS}.tsv"
    done
    # worst-case regret per policy = max over archetypes of (row-best - policy)
    python3 -c "
import sys
pols=['never','always','prefix16','least-ttft','edpp']
rows=[l.split('\t') for l in open('$OUT/grid_${FL}_${ITL_MS}.tsv') if l.strip()]
reg={p:0.0 for p in pols}
for r in rows:
    g=[float(x) for x in r[1:6]]
    best=max(g)
    for p,v in zip(pols,g): reg[p]=max(reg[p],best-v)
print('   worst-case regret : '+'  '.join('%s=%.3f'%(p,reg[p]) for p in pols))
winner=min(reg,key=reg.get)
print('   lowest-regret policy: %s (%.3f)'%(winner,reg[winner]))
" >&2
  done
done
echo "" >&2
echo "done -> $OUT" >&2
