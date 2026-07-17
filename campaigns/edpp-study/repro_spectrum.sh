#!/usr/bin/env bash
# Workload-spectrum policy comparison on a FIXED 1P2D topology.
# Backs every number in campaigns/edpp-study/STUDY_REPORT.md. Self-contained:
# emits its own workload specs (campaigns/edpp-study/specs/ is git-ignored scratch).
#
# THE QUESTION: on a fixed disaggregated topology, does a "smart" policy (EDPP)
# beat naive heuristics on goodput, and where?
#
# METHOD (why it is built this way):
#  - Fixed 1P2D (instance_0 = prefill, instance_1/2 = decode). The operator picks the
#    topology; the policy only decides per-request routing. Comparing to a different
#    topology would answer a provisioning question, not a policy question.
#  - 4 workload archetypes spanning decode-bound -> prefill-bound, because which pool is
#    the bottleneck is the thing that should drive the disaggregation decision.
#  - SLO is auto-set per archetype from an IDLE probe (2x idle e2e_p99, 3x idle ttft_p99).
#    Each archetype has a different intrinsic latency, so a single fixed SLO would measure
#    "which archetype is fast" instead of "which policy is good". Auto-scaling makes
#    goodput comparable ACROSS archetypes.
#  - --max-num-running-reqs caps concurrency so the system actually saturates at reachable
#    rates. With the default cap (256) + abundant KV these workloads never queue and every
#    policy scores 1.000 (no signal).
#  - SCORER: which DECODE INSTANCE a request lands on is chosen by the routing scorer, NOT
#    by the reduced EDPP rule. We report BOTH setups:
#      SCORER=llmd     -> the llm-d shipped PD profile (precise-prefix-cache:2,queue-depth:1)
#      SCORER=balanced -> queue-depth:1 (pure load balancing)
#    They differ enormously (see MODE=scorer). Holding it fixed at `balanced` isolates the
#    disaggregation decision from a decode load-balancing artifact.
#
# POLICIES COMPARED (all reduced path; none of them choose the decode instance):
#   never            - never disaggregate (prefill runs on the decode instance)
#   always           - always disaggregate (prefill on the prefill instance)
#   prefix-threshold - disaggregate iff uncached prompt tokens > N (default 16 = llm-d shipped)
#   least-ttft       - disaggregate iff predicted TTFT_disagg < predicted TTFT_local
#                      (EDPP's estimator, WITHOUT its drift/z/V machinery)
#   edpp             - full drift-plus-penalty rule (penalty = V*c_xfer transfer cost)
#
# MODES:
#   MODE=spectrum (default) - archetypes x policies x load, for one SCORER setting
#   MODE=scorer             - the llm-d-default vs balanced decode-scorer comparison
#   MODE=oracle             - static disaggregation-fraction sweep (the fixed-plan yardstick)
#
# Usage (from repo root):
#   bash campaigns/edpp-study/repro_spectrum.sh                       # spectrum, balanced
#   SCORER=llmd bash campaigns/edpp-study/repro_spectrum.sh           # spectrum, llm-d default
#   MODE=scorer bash campaigns/edpp-study/repro_spectrum.sh           # scorer comparison
#   MODE=oracle ARCH=prefill_bound RATE=8 bash campaigns/edpp-study/repro_spectrum.sh
#   MODE=ablate bash campaigns/edpp-study/repro_spectrum.sh              # term ablation
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
SPECDIR="campaigns/edpp-study/specs/spectrum"
OUT="${OUT:-campaigns/edpp-study/out/spectrum}"
MODE="${MODE:-spectrum}"
SCORER="${SCORER:-balanced}"
CAP="${CAP:-16}"
SEED="${SEED:-42}"
RATES="${RATES:-4 8 16}"
mkdir -p "$SPECDIR" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

# 1P2D: instance_0 = prefill, instance_1 + instance_2 = decode.
TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --max-num-running-reqs "$CAP")
case "$SCORER" in
  llmd)     DEC=() ;;                                            # unset => llm-d PD profile precise-prefix-cache:2,queue-depth:1
  balanced) DEC=(--decode-routing-scorers "queue-depth:1") ;;    # pure load balancing
  *) echo "SCORER must be llmd|balanced" >&2; exit 1 ;;
esac

# archetype name -> "input_tokens output_tokens"; spans decode-bound -> prefill-bound.
# (Plain function, not an associative array: macOS ships bash 3.2, which has no `declare -A`.)
arch_dims(){
  case "$1" in
    decode)        echo "256 512"   ;;  # short prompt, long generation  -> decode-bound
    mixed)         echo "2048 128"  ;;  # balanced
    prefill_lean)  echo "8192 64"   ;;  # long prompt, short generation
    prefill_bound) echo "16000 16"  ;;  # very long prompt, tiny generation -> prefill-bound
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
# split(): realized decode allocation. Every completed request decodes exactly once, so
# instance_1.completed / instance_2.completed IS where decode work went.
split(){ python3 -c "
import re,sys
try: t=open('$1').read()
except OSError: print('NA'); sys.exit()
m={i:int(c) for i,c in re.findall(r'\"instance_id\":\s*\"(instance_\d+)\".*?\"completed_requests\":\s*(\d+)',t,re.S)}
f=m.get('instance_1',0); s=m.get('instance_2',0)
print('i1=%d i2=%d'%(f,s))" 2>/dev/null || echo NA; }

# auto_slo(): idle probe -> per-archetype SLO so goodput is comparable across archetypes.
# The probe uses a FIXED seed (SLO_PROBE_SEED), never $SEED: the SLO is a fixed target, so it
# must not move when you sweep seeds, or each seed would be graded against different goalposts.
auto_slo(){ # $1=in $2=out ; sets globals SLO_E2E SLO_TTFT
  spec "$1" "$2" 0.5 "${SLO_PROBE_SEED:-42}"
  ./blis run --model "$MODEL" --workload-spec "$SPECDIR/w.yaml" "${TOPO[@]}" ${DEC[@]+"${DEC[@]}"} \
    --pd-decider always --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" \
    --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),1000))")
  IDLE_E2E=$ie; IDLE_TTFT=$it
}

run_policy(){ # $1=tag $2..=extra flags ; echoes "goodput  i1=..,i2=.."
  local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$SPECDIR/w.yaml" "${TOPO[@]}" ${DEC[@]+"${DEC[@]}"} "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=100ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >"$OUT/$tag.out" 2>/dev/null || true
  printf '%s  %s' "$(gp "$OUT/$tag.json")" "$(split "$OUT/$tag.out")"
}

if [[ "$MODE" == "spectrum" ]]; then
  echo "SPECTRUM  topology=1P2D cap=$CAP seed=$SEED scorer=$SCORER  (goodput; higher is better)" >&2
  for name in $ARCH_ORDER; do
    set -- $(arch_dims "$name"); IN=$1; O=$2
    auto_slo "$IN" "$O"
    echo "== $name (in=$IN out=$O) idle e2e_p99=${IDLE_E2E}ms ttft_p99=${IDLE_TTFT}ms -> SLO e2e=${SLO_E2E}ms ttft=${SLO_TTFT}ms ==" >&2
    printf "   %-5s| %-8s %-8s %-8s %-10s %-8s\n" "rate" "never" "always" "prefix16" "least-ttft" "edpp" >&2
    for r in $RATES; do
      spec "$IN" "$O" "$r" "$SEED"
      EC=(--edpp-coeffs "$COEFFS" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-itl 100ms --edpp-tadm-estimator rollforward)
      N=$(run_policy n --pd-decider never);                                          N=${N%% *}
      A=$(run_policy a --pd-decider always);                                         A=${A%% *}
      P=$(run_policy p --pd-decider prefix-threshold --pd-prefix-threshold 16);      P=${P%% *}
      L=$(run_policy l --pd-decider edpp "${EC[@]}" --edpp-rule least-ttft);         L=${L%% *}
      E=$(run_policy e --pd-decider edpp "${EC[@]}");                                E=${E%% *}
      printf "   %-5s| %-8s %-8s %-8s %-10s %-8s\n" "$r" "$N" "$A" "$P" "$L" "$E" >&2
    done
  done

elif [[ "$MODE" == "scorer" ]]; then
  # The decode-instance scorer is chosen by the ROUTER, not by the reduced EDPP rule.
  # llm-d's shipped PD profile weights precise-prefix-cache 2x; with one shared prefix
  # group it pins ALL decode onto a single instance. Compare against pure load balancing.
  set -- $(arch_dims mixed); IN=$1; O=$2; R="${RATE:-16}"
  echo "SCORER COMPARISON  archetype=mixed (in=$IN out=$O) rate=$R cap=$CAP  (goodput + realized decode split)" >&2
  for s in llmd balanced; do
    case "$s" in llmd) DEC=() ;; balanced) DEC=(--decode-routing-scorers "queue-depth:1") ;; esac
    auto_slo "$IN" "$O"; spec "$IN" "$O" "$R" "$SEED"
    echo " -- scorer=$s --" >&2
    for pol in never always; do
      out=$(run_policy "sc_${s}_${pol}" --pd-decider "$pol")
      printf "    %-7s %s\n" "$pol" "$out" >&2
    done
  done

elif [[ "$MODE" == "oracle" ]]; then
  # Static disaggregation-fraction yardstick via --pd-plan (a forced per-request plan).
  # For fraction f: the first f of every 100 requests get prefill_instance=instance_0
  # (disaggregated); the rest get "" (local). Decode alternates instance_1/instance_2 so
  # decode is perfectly balanced and cannot confound the comparison.
  # f=0 == never, f=100 == always. An INTERIOR maximum means neither corner is optimal.
  name="${ARCH:-prefill_bound}"; set -- $(arch_dims "$name"); IN=$1; O=$2; R="${RATE:-8}"
  auto_slo "$IN" "$O"; spec "$IN" "$O" "$R" "$SEED"
  echo "STATIC DISAGG-FRACTION ORACLE  archetype=$name (in=$IN out=$O) rate=$R cap=$CAP seed=$SEED" >&2
  echo "  SLO e2e=${SLO_E2E}ms ttft=${SLO_TTFT}ms   (f=0 is 'never', f=100 is 'always')" >&2
  for f in ${FRACS:-0 20 25 30 35 40 45 50 60 80 100}; do
    python3 -c "
import csv
f=$f
w=csv.DictWriter(open('$OUT/plan.csv','w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for i in range(240):
    w.writerow({'request_id':f'request_{i}','decode_instance':('instance_1' if i%2==0 else 'instance_2'),'prefill_instance':('instance_0' if (i%100)<f else '')})"
    g=$(run_policy "f$f" --pd-plan "$OUT/plan.csv"); g=${g%% *}
    printf "   f=%-3s disagg%%  goodput=%s\n" "$f" "$g" >&2
  done
  EC=(--edpp-coeffs "$COEFFS" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-tau-itl 100ms --edpp-tadm-estimator rollforward)
  echo "   ---- dynamic policies on the same cell ----" >&2
  printf "   %-12s goodput=%s\n" "least-ttft" "$(run_policy o_l --pd-decider edpp "${EC[@]}" --edpp-rule least-ttft)" >&2
  printf "   %-12s goodput=%s\n" "edpp"       "$(run_policy o_e --pd-decider edpp "${EC[@]}")" >&2
elif [[ "$MODE" == "ablate" ]]; then
  # TERM ABLATION of the reduced rule:  disagg iff  lhs > rhs
  #   lhs = balanceTermD - balanceTermP        (congestion drift, the Q_i work backlogs)
  #   rhs = transferTerm + ttftTerm + itlTerm  (V*c_xfer  +  the z SLO virtual queues)
  # Reachable from config alone — no code change:
  #   --edpp-v 0            => transferTerm = 0
  #   --edpp-tau-ttft 999s  => TTFT unviolatable => z_ttft == 0 => ttftTerm = 0
  # tau_itl is HELD at its real value in every arm, for two reasons: the normalizer
  # mu_D = 1 - alpha_D/tau_itl must not move between arms, and z_itl is inert anyway
  # (measured ITL never approaches 100ms). With rhs=0 the surviving test `lhs > 0` is
  # invariant to the tau_ttft scaling of W*, so the ablation removes z without rescaling drift.
  # ARMS:  least-ttft = no drift, no z, no V   |  drift only = lhs > 0
  #        drift + z  = no V                   |  full       = everything
  echo "TERM ABLATION  topology=1P2D cap=$CAP scorer=$SCORER seeds=[${SEEDS:-42 7 123}]" >&2
  echo "(goodput; which TERM is load-bearing?  z_itl is inert throughout — this ablates z_ttft)" >&2
  for name in $ARCH_ORDER; do
    set -- $(arch_dims "$name"); IN=$1; O=$2
    auto_slo "$IN" "$O"
    echo "== $name (in=$IN out=$O)  SLO e2e=${SLO_E2E}ms ttft=${SLO_TTFT}ms ==" >&2
    printf "   %-5s %-5s| %-11s %-11s %-11s %-11s\n" "rate" "seed" "least-ttft" "drift-only" "drift+z" "full" >&2
    for r in $RATES; do
      for s in ${SEEDS:-42 7 123}; do
        spec "$IN" "$O" "$r" "$s"
        BB=(--edpp-coeffs "$COEFFS" --edpp-tadm-estimator rollforward --edpp-tau-itl 100ms)
        L=$(run_policy ab_l --pd-decider edpp "${BB[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-rule least-ttft); L=${L%% *}
        D=$(run_policy ab_d --pd-decider edpp "${BB[@]}" --edpp-tau-ttft 999s --edpp-v 0);                       D=${D%% *}
        Z=$(run_policy ab_z --pd-decider edpp "${BB[@]}" --edpp-tau-ttft "${SLO_TTFT}ms" --edpp-v 0);            Z=${Z%% *}
        F=$(run_policy ab_f --pd-decider edpp "${BB[@]}" --edpp-tau-ttft "${SLO_TTFT}ms");                       F=${F%% *}
        printf "   %-5s %-5s| %-11s %-11s %-11s %-11s\n" "$r" "$s" "$L" "$D" "$Z" "$F" >&2
      done
    done
  done
else
  echo "MODE must be spectrum|scorer|oracle|ablate" >&2; exit 1
fi
echo "done -> $OUT" >&2
