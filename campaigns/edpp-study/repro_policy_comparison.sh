#!/usr/bin/env bash
# EDPP policy comparison — goodput/SLO + one-step counterfactual regret for the five
# policies in the joint-routing formulation's baseline table
# (docs/design/2026-06-30-pd-joint-routing-problem-formulation.md, section 5), across
# topology×workload cells. Complements repro_joint.sh (which is reduced-vs-joint only)
# by adding the static/heuristic baselines (never / always / prefix-threshold).
#
# Policies (all with the occupancy-aware rollforward estimator where EDPP applies):
#   never            — force all-local (aggregate)              (--pd-decider never)
#   always           — force all-disaggregate                  (--pd-decider always)
#   prefix-threshold — disagg iff uncached tokens > 16 (llm-d)  (--pd-decider prefix-threshold)
#   reduced          — EDPP, scorer picks decode d              (--pd-decider edpp)
#   joint            — EDPP joint argmin over (d,p)             (--pd-decider edpp --edpp-joint)
#
# Per cell × policy: run once (goodput = slo_attainment), then the counterfactual
# one-step-deviation regret (reuse counterfactual_regret.py): capture the realized
# (d,p) plan, re-run K sampled requests × each alternative action, total_regret.
# Regret is a LOCAL diagnostic (how much a single better choice would recover), NOT
# the global optimum. For never/always it is a fixed-policy diagnostic; goodput is
# the primary "how good".
#
# Plan capture: joint overrides the scorer's LOCAL decode, so its plan comes from the
# --edpp-joint-trace (records joint_d/joint_p for every request); all other policies
# never override the scorer's local decode, so --pd-outcome-trace capture is faithful.
#
# Cost: per policy-cell = 1 + K*(|A|-1) sims (|A|=4 at 1P2D, 6 at 2P2D). Small K default.
# Usage:  bash campaigns/edpp-study/repro_policy_comparison.sh   [K=.. overridable]
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
ESTIMATOR="${ESTIMATOR:-rollforward}"
PREFIX_THRESHOLD="${PREFIX_THRESHOLD:-16}"
K="${K:-4}"
OUT="${OUT:-campaigns/edpp-study/out/policy_cmp}"
REGRET_DRIVER="campaigns/edpp-study/analyze/counterfactual_regret.py"
POLICIES="never always prefix-threshold reduced joint"
mkdir -p "$OUT"

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi
if [[ ! -f campaigns/edpp-study/specs/synth_cf.yaml || ! -f campaigns/edpp-study/specs/synth_asym.yaml ]]; then
  echo "generating specs (make_specs.py)..." >&2
  python3 campaigns/edpp-study/make_specs.py >/dev/null
fi

SLO=(--slo-ttft "batch=2s" --slo-itl "batch=150ms")
goodput() { python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(m['slo_attainment'] if isinstance(m,dict) else sum(x['slo_attainment'] for x in m)/len(m))" "$1"; }

# decider_flags <policy> — echo the per-policy decider flag set.
decider_flags() {
  case "$1" in
    never)            echo "--pd-decider never" ;;
    always)           echo "--pd-decider always" ;;
    prefix-threshold) echo "--pd-decider prefix-threshold --pd-prefix-threshold $PREFIX_THRESHOLD" ;;
    reduced)          echo "--pd-decider edpp --edpp-coeffs $COEFFS --edpp-tau-ttft 2s --edpp-tau-itl 150ms --edpp-tadm-estimator $ESTIMATOR" ;;
    joint)            echo "--pd-decider edpp --edpp-coeffs $COEFFS --edpp-tau-ttft 2s --edpp-tau-itl 150ms --edpp-tadm-estimator $ESTIMATOR --edpp-joint" ;;
  esac
}

# joint plan from its trace (joint overrides local decode; outcome trace can't capture that)
plan_from_joint_trace() {
  python3 -c "import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
w=csv.DictWriter(open(sys.argv[2],'w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for r in rows:
    disagg=r['disaggregate'].strip().lower()=='true'
    w.writerow({'request_id':r['request_id'],'decode_instance':r['joint_d'],'prefill_instance':(r['joint_p'] if disagg and r['joint_p'] else 'local')})" "$1" "$2"
}

# regret_for <policy> <cell-out-dir>  — needs CELL_COMMON, DECODES, PREFILLS globals.
# Returns 0 and writes $rout/regret.json on success; returns 1 (regret unavailable)
# when the policy's realized (d,p) plan cannot be captured — notably `never`, whose
# all-local run produces NO --pd-outcome-trace rows ("need PD/disaggregation enabled"),
# so there is no plan file. (Known deferred gap: --pd-outcome-trace omits decode_instance
# for local requests; `never` is entirely local.) Goodput is still reported for every policy.
regret_for() {
  local pol="$1" cdir="$2"
  local rout="$cdir/regret_$pol"
  mkdir -p "$rout"; rm -f "$rout"/dev_*.json "$rout"/dev_*.plan.csv "$rout/regret.json"
  cp "$cdir/${pol}.json" "$rout/baseline.json"
  if [[ "$pol" == "joint" ]]; then
    plan_from_joint_trace "$cdir/joint_trace.csv" "$rout/plan.csv"
  else
    if [[ ! -s "$cdir/${pol}_outcome.csv" ]]; then
      echo "    NOTE[$pol]: no outcome-trace rows (all-local / no disagg) — regret unavailable." >&2
      return 1
    fi
    python3 "$REGRET_DRIVER" capture-plan --outcome "$cdir/${pol}_outcome.csv" --out "$rout/plan.csv"
  fi
  # self-consistency: the captured plan must replay to the policy's own goodput
  ./blis run "${CELL_COMMON[@]}" --pd-plan "$rout/plan.csv" --metrics-path "$rout/replay.json" >/dev/null 2>&1
  local bg rg; bg="$(goodput "$rout/baseline.json")"; rg="$(goodput "$rout/replay.json")"
  if ! python3 -c "import sys;sys.exit(0 if abs($bg-$rg)<1e-9 else 1)"; then
    echo "    WARN[$pol]: plan replay ($rg) != baseline ($bg); regret suspect." >&2; fi
  local SAMPLE=()
  while IFS= read -r line; do SAMPLE+=("$line"); done < <(python3 - "$rout/plan.csv" "$K" <<'PY'
import csv,random,sys
rows=list(csv.DictReader(open(sys.argv[1]))); K=min(int(sys.argv[2]),len(rows)); random.seed(1234)
for r in random.sample(rows,K): print(r["request_id"])
PY
)
  for rid in "${SAMPLE[@]}"; do
    local cur cur_dec cur_pre
    cur="$(python3 -c "import csv,sys;[print(r['decode_instance']+'|'+r['prefill_instance']) for r in csv.DictReader(open(sys.argv[1])) if r['request_id']==sys.argv[2]]" "$rout/plan.csv" "$rid")"
    cur_dec="${cur%%|*}"; cur_pre="${cur##*|}"; [[ "$cur_pre" == "" ]] && cur_pre="local"
    for dec in $DECODES; do
      for pre in local $PREFILLS; do
        [[ "$dec" == "$cur_dec" && "$pre" == "$cur_pre" ]] && continue
        local dec_tok pre_tok action devplan
        dec_tok="${dec//_/}"; pre_tok="local"; [[ "$pre" != "local" ]] && pre_tok="${pre//_/}"
        action="${dec_tok}-${pre_tok}"; devplan="$rout/dev_${rid}_${action}.plan.csv"
        python3 -c "import csv,sys
rid,dec,pre=sys.argv[2],sys.argv[3],sys.argv[4]
rows=list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    if r['request_id']==rid: r['decode_instance']=dec; r['prefill_instance']=('' if pre=='local' else pre)
w=csv.DictWriter(open(sys.argv[5],'w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader();w.writerows(rows)" \
          "$rout/plan.csv" "$rid" "$dec" "$pre" "$devplan"
        ./blis run "${CELL_COMMON[@]}" --pd-plan "$devplan" --metrics-path "$rout/dev_${rid}_${action}.json" >/dev/null 2>&1
      done
    done
  done
  python3 "$REGRET_DRIVER" regret --sweep-dir "$rout" --out "$rout/regret.json" >/dev/null
}

echo "policy comparison — K=$K, estimator=$ESTIMATOR, prefix-threshold=$PREFIX_THRESHOLD" >&2
printf '%-8s %-10s' cell workload
for p in $POLICIES; do printf ' %-18s' "$p(g/reg)"; done; printf '\n'

for topo in ${TOPOS:-1P2D 2P2D}; do
  if [[ "$topo" == "1P2D" ]]; then
    TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2); DECODES="instance_1 instance_2"; PREFILLS="instance_0"
  else
    TOPO=(--num-instances 4 --prefill-instances 2 --decode-instances 2); DECODES="instance_2 instance_3"; PREFILLS="instance_0 instance_1"
  fi
  for wl in ${WORKLOADS:-synth_cf synth_asym}; do
    SPEC="campaigns/edpp-study/specs/${wl}.yaml"
    cdir="$OUT/${topo}_${wl}"; mkdir -p "$cdir"
    CELL_COMMON=(--model "$MODEL" --workload-spec "$SPEC" "${TOPO[@]}" "${SLO[@]}")
    printf '%-8s %-10s' "$topo" "$wl"
    for pol in $POLICIES; do
      read -r -a DF <<< "$(decider_flags "$pol")"
      if [[ "$pol" == "joint" ]]; then
        ./blis run "${CELL_COMMON[@]}" "${DF[@]}" --trace-level decisions \
          --edpp-joint-trace "$cdir/joint_trace.csv" \
          --pd-outcome-trace "$cdir/${pol}_outcome.csv" --metrics-path "$cdir/${pol}.json" >/dev/null 2>&1
      else
        ./blis run "${CELL_COMMON[@]}" "${DF[@]}" \
          --pd-outcome-trace "$cdir/${pol}_outcome.csv" --metrics-path "$cdir/${pol}.json" >/dev/null 2>&1
      fi
      local_g="$(goodput "$cdir/${pol}.json")"   # goodput always available
      local_r="NA"
      if regret_for "$pol" "$cdir"; then           # regret best-effort (|| true so set -e won't abort)
        local_r="$(python3 -c "import json;print('%.4f'%json.load(open('$cdir/regret_$pol/regret.json'))['total_regret'])")"
      fi || true
      printf ' %-18s' "$(printf '%.3f/%s' "$local_g" "$local_r")"
    done
    printf '\n'
  done
done
echo "Done. Artifacts per cell in $OUT/<topo>_<workload>/ (per-policy .json + regret_<policy>/regret.json)." >&2
