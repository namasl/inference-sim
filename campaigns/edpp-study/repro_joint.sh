#!/usr/bin/env bash
# EDPP joint P/D mechanism: correctness gates + exploratory topology×workload sweep.
#
# Compares the REDUCED EDPP decider (fixed-decode local-vs-disagg rule) against the
# JOINT decider (--edpp-joint: enumerate all (decode,prefill) candidates, pick the
# drift-plus-penalty argmin). Both run with the occupancy-aware rollforward
# admission-delay estimator. This is a LOCAL diagnostic on HOMOGENEOUS hardware —
# the joint objective's active levers here are cache warmth and per-instance
# occupancy; per-instance hardware heterogeneity (θ_i) is deferred.
#
# STAGE 1 — CORRECTNESS GATES (pass/fail, run FIRST; any failure => exit 1):
#   (a) byte-identical: two reduced-EDPP runs (--edpp-joint OFF) must produce
#       byte-identical metrics + decision trace (INV-6; confirms the joint plumbing
#       does not perturb the reduced default path).
#   (b) §5.5 reduction unit test: joint J restricted to the scorer's single decode
#       reproduces the reduced decision (go test TestJoint_ReducesToScorer...).
#   (c) counterfactual self-consistency on the JOINT plan: capture joint's realized
#       (d,p) plan, replay it via --pd-plan, assert slo_attainment == joint baseline
#       (INV-6/INV-13). If the joint plan does not replay faithfully, all downstream
#       regret is meaningless => STOP.
#
# STAGE 2 — EXPLORATORY SWEEP (observe, don't predict):
#   cells = {1P2D, 2P2D} × {synth_cf (cache-uniform), synth_asym (cache-asymmetric)}
#   per cell, per policy in {reduced, joint}:
#     - run with --pd-outcome-trace + --metrics-path (+ --edpp-joint-trace for joint)
#     - counterfactual_regret.py: capture plan -> K sampled one-step deviations -> regret
#     - joint_divergence.py (joint only): d/p divergence rate + direction
#   echo a per-cell summary line: goodput + regret (reduced vs joint) + divergence.
#
# K is SMALL to bound runtime: cost per policy-cell = K*(|A|-1) sims
#   (|A|=4 for 1P2D => 3 devs/req; |A|=6 for 2P2D => 5 devs/req). Override with K=..
#
# Usage (from repo root):
#   bash campaigns/edpp-study/repro_joint.sh
#   K=4 bash campaigns/edpp-study/repro_joint.sh
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
ESTIMATOR="${ESTIMATOR:-rollforward}"
K="${K:-6}"
OUT="${OUT:-campaigns/edpp-study/out/joint}"
REGRET_DRIVER="campaigns/edpp-study/analyze/counterfactual_regret.py"
DIV_DRIVER="campaigns/edpp-study/analyze/joint_divergence.py"
mkdir -p "$OUT"

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi

# specs/ is gitignored (reproducible artifacts); regenerate from make_specs.py if
# the cache-uniform (synth_cf) or cache-asymmetric (synth_asym) spec is missing.
if [[ ! -f campaigns/edpp-study/specs/synth_cf.yaml || ! -f campaigns/edpp-study/specs/synth_asym.yaml ]]; then
  echo "generating specs (make_specs.py)..." >&2
  python3 campaigns/edpp-study/make_specs.py >/dev/null
fi

SLO=(--slo-ttft "batch=2s" --slo-itl "batch=150ms")
EDPP=(--pd-decider edpp --edpp-coeffs "$COEFFS" --edpp-tau-ttft 2s --edpp-tau-itl 150ms --edpp-tadm-estimator "$ESTIMATOR")

goodput() { python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(m['slo_attainment'] if isinstance(m,dict) else sum(x['slo_attainment'] for x in m)/len(m))" "$1"; }

# plan_from_joint_trace <joint-trace.csv> <plan.csv>
# The --pd-outcome-trace leaves decode_instance EMPTY for non-disaggregated (local)
# requests; on replay an empty decode falls back to the scorer's decode. That is
# faithful for REDUCED (which never overrides the scorer's decode) but NOT for JOINT,
# whose whole point is to override the local decode away from the scorer's pick. So
# the faithful source for a joint plan is the joint trace, which records joint_d /
# joint_p for EVERY request. Verified: replaying this plan reproduces joint's goodput.
plan_from_joint_trace() {
  python3 -c "import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
w=csv.DictWriter(open(sys.argv[2],'w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for r in rows:
    disagg=r['disaggregate'].strip().lower()=='true'
    w.writerow({'request_id':r['request_id'],'decode_instance':r['joint_d'],'prefill_instance':(r['joint_p'] if disagg and r['joint_p'] else 'local')})" "$1" "$2"
}

echo "########################################################################" >&2
echo "# STAGE 1: CORRECTNESS GATES" >&2
echo "########################################################################" >&2

# Gate (a): byte-identical reduced runs (INV-6). Use 1P2D synth_cf as the reference cell.
G_TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2)
G_COMMON=(--model "$MODEL" --workload-spec campaigns/edpp-study/specs/synth_cf.yaml "${G_TOPO[@]}" "${SLO[@]}")
echo "[gate a] byte-identical reduced-EDPP (--edpp-joint OFF), two runs" >&2
for i in 1 2; do
  ./blis run "${G_COMMON[@]}" "${EDPP[@]}" --trace-level decisions \
    --edpp-decision-trace "$OUT/gateA_dec_$i.csv" --metrics-path "$OUT/gateA_$i.json" >/dev/null 2>&1
done
if ! diff -q "$OUT/gateA_1.json" "$OUT/gateA_2.json" >/dev/null; then
  echo "GATE A FAILED: reduced-EDPP metrics differ between runs (INV-6 determinism)." >&2; exit 1; fi
if ! diff -q "$OUT/gateA_dec_1.csv" "$OUT/gateA_dec_2.csv" >/dev/null; then
  echo "GATE A FAILED: reduced-EDPP decision trace differs between runs." >&2; exit 1; fi
echo "    OK: reduced metrics + decision trace byte-identical." >&2

# Gate (b): §5.5 reduction unit test.
echo "[gate b] §5.5 reduction unit test (joint|scorer-d == reduced)" >&2
if ! go test ./sim/ -run 'TestJoint_ReducesToScorerSliceMatchesReduced' -count=1 >/dev/null 2>&1; then
  echo "GATE B FAILED: §5.5 reduction unit test failed." >&2; exit 1; fi
echo "    OK: reduction unit test passes." >&2

# Gate (c): counterfactual self-consistency on the JOINT plan (1P2D synth_cf).
echo "[gate c] counterfactual self-consistency on the JOINT plan (--pd-plan replay)" >&2
./blis run "${G_COMMON[@]}" "${EDPP[@]}" --edpp-joint --trace-level decisions \
  --edpp-joint-trace "$OUT/gateC_trace.csv" \
  --pd-outcome-trace "$OUT/gateC_outcome.csv" --metrics-path "$OUT/gateC_base.json" >/dev/null 2>&1
plan_from_joint_trace "$OUT/gateC_trace.csv" "$OUT/gateC_plan.csv"
./blis run "${G_COMMON[@]}" --pd-plan "$OUT/gateC_plan.csv" --metrics-path "$OUT/gateC_replay.json" >/dev/null 2>&1
CB="$(goodput "$OUT/gateC_base.json")"; CR="$(goodput "$OUT/gateC_replay.json")"
if ! python3 -c "import sys;sys.exit(0 if abs($CB-$CR)<1e-9 else 1)"; then
  echo "GATE C FAILED: joint plan replay ($CR) != joint baseline ($CB). Plan not faithfully replayed (INV-6/INV-13). STOP." >&2
  exit 1; fi
echo "    OK: joint baseline=$CB replay=$CR (self-consistent)." >&2
echo "ALL CORRECTNESS GATES PASSED." >&2

# ---------------------------------------------------------------------------
# regret_for: capture a policy's plan on a cell, sweep K one-step deviations, print regret json path.
# Globals expected: CELL_COMMON (array), CELL_EDPP (array), DECODES (space list), PREFILLS (space list)
# Args: <policy-tag> <cell-out-dir>
# ---------------------------------------------------------------------------
regret_for() {
  local tag="$1" cdir="$2"
  local rout="$cdir/regret_$tag"
  mkdir -p "$rout"; rm -f "$rout"/dev_*.json "$rout"/dev_*.plan.csv
  cp "$cdir/${tag}.json" "$rout/baseline.json"
  if [[ "$tag" == "joint" ]]; then
    # joint decode overrides the scorer for local requests => build from the joint trace.
    plan_from_joint_trace "$cdir/joint_trace.csv" "$rout/plan.csv"
  else
    cp "$cdir/${tag}_outcome.csv" "$rout/outcome.csv"
    python3 "$REGRET_DRIVER" capture-plan --outcome "$rout/outcome.csv" --out "$rout/plan.csv"
  fi
  # self-consistency for this plan
  ./blis run "${CELL_COMMON[@]}" --pd-plan "$rout/plan.csv" --metrics-path "$rout/replay.json" >/dev/null 2>&1
  local bg rg; bg="$(goodput "$rout/baseline.json")"; rg="$(goodput "$rout/replay.json")"
  if ! python3 -c "import sys;sys.exit(0 if abs($bg-$rg)<1e-9 else 1)"; then
    echo "    WARN[$tag]: plan replay ($rg) != baseline ($bg); regret suspect." >&2; fi
  # sample K request ids
  local SAMPLE=()
  while IFS= read -r line; do SAMPLE+=("$line"); done < <(python3 - "$rout/plan.csv" "$K" <<'PY'
import csv,random,sys
rows=list(csv.DictReader(open(sys.argv[1])))
K=min(int(sys.argv[2]),len(rows))
random.seed(1234)
for r in random.sample(rows,K): print(r["request_id"])
PY
)
  for rid in "${SAMPLE[@]}"; do
    local cur cur_dec cur_pre
    cur="$(python3 -c "import csv,sys;[print(r['decode_instance']+'|'+r['prefill_instance']) for r in csv.DictReader(open(sys.argv[1])) if r['request_id']==sys.argv[2]]" "$rout/plan.csv" "$rid")"
    cur_dec="${cur%%|*}"; cur_pre="${cur##*|}"; [[ "$cur_pre" == "" ]] && cur_pre="local"
    for dec in $DECODES; do
      for pre in local $PREFILLS; do
        if [[ "$dec" == "$cur_dec" && "$pre" == "$cur_pre" ]]; then continue; fi
        local dec_tok pre_tok action devplan
        dec_tok="${dec//_/}"; pre_tok="local"; [[ "$pre" != "local" ]] && pre_tok="${pre//_/}"
        action="${dec_tok}-${pre_tok}"
        devplan="$rout/dev_${rid}_${action}.plan.csv"
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

echo "" >&2
echo "########################################################################" >&2
echo "# STAGE 2: EXPLORATORY SWEEP (K=$K, estimator=$ESTIMATOR)" >&2
echo "########################################################################" >&2
printf '%-8s %-10s %-9s %-9s %-9s %-9s %-10s %-10s\n' cell workload g_red g_joint reg_red reg_joint d_div dir_lowerJ

for topo in 1P2D 2P2D; do
  if [[ "$topo" == "1P2D" ]]; then
    TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2); DECODES="instance_1 instance_2"; PREFILLS="instance_0"
  else
    TOPO=(--num-instances 4 --prefill-instances 2 --decode-instances 2); DECODES="instance_2 instance_3"; PREFILLS="instance_0 instance_1"
  fi
  for wl in synth_cf synth_asym; do
    SPEC="campaigns/edpp-study/specs/${wl}.yaml"
    cdir="$OUT/${topo}_${wl}"; mkdir -p "$cdir"
    CELL_COMMON=(--model "$MODEL" --workload-spec "$SPEC" "${TOPO[@]}" "${SLO[@]}")
    CELL_EDPP=("${EDPP[@]}")

    # reduced
    ./blis run "${CELL_COMMON[@]}" "${CELL_EDPP[@]}" \
      --pd-outcome-trace "$cdir/reduced_outcome.csv" --metrics-path "$cdir/reduced.json" >/dev/null 2>&1
    # joint
    ./blis run "${CELL_COMMON[@]}" "${CELL_EDPP[@]}" --edpp-joint --trace-level decisions \
      --edpp-joint-trace "$cdir/joint_trace.csv" \
      --pd-outcome-trace "$cdir/joint_outcome.csv" --metrics-path "$cdir/joint.json" >/dev/null 2>&1

    G_RED="$(goodput "$cdir/reduced.json")"; G_JOINT="$(goodput "$cdir/joint.json")"

    regret_for reduced "$cdir"
    regret_for joint "$cdir"
    REG_RED="$(python3 -c "import json;print('%.4f'%json.load(open('$cdir/regret_reduced/regret.json'))['total_regret'])")"
    REG_JOINT="$(python3 -c "import json;print('%.4f'%json.load(open('$cdir/regret_joint/regret.json'))['total_regret'])")"

    python3 "$DIV_DRIVER" summary --trace "$cdir/joint_trace.csv" --out "$cdir/divergence.json" >/dev/null 2>&1
    D_DIV="$(python3 -c "import json;print('%.3f'%json.load(open('$cdir/divergence.json'))['any_divergence_rate'])")"
    DIR_LO="$(python3 -c "import json;print('%.3f'%json.load(open('$cdir/divergence.json'))['direction_on_divergent']['dir_lower_J'])")"

    printf '%-8s %-10s %-9s %-9s %-9s %-9s %-10s %-10s\n' "$topo" "$wl" "$G_RED" "$G_JOINT" "$REG_RED" "$REG_JOINT" "$D_DIV" "$DIR_LO"
  done
done
echo "" >&2
echo "Done. Per-cell artifacts in $OUT/<topo>_<workload>/ (regret_*/regret.json, divergence.json)." >&2
