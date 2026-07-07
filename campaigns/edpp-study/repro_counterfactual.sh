#!/usr/bin/env bash
# Per-decision counterfactual regret for reduced-EDPP (see FINDINGS.md
# "Counterfactual regret"). Exact one-step-deviation hindsight regret: capture the
# policy's realized (decode,prefill) plan, then for each sampled request replay the
# plan with only THAT request's action changed and measure the aggregate-goodput
# delta. A positive regret means some single alternative decision would have
# improved total goodput in hindsight, i.e. the decider left goodput on the table.
#
# Pipeline:
#   1. Run TARGET_POLICY on a saturating synth trace at 1P2D with
#      --pd-outcome-trace + --metrics-path  -> baseline plan + baseline goodput.
#   2. capture-plan: outcome.csv -> plan.csv (fixed per-request routing).
#   3. Self-consistency gate (INV-6/INV-13): replay plan.csv via --pd-plan and
#      assert slo_attainment == baseline. If not, the decider is not replaying the
#      plan faithfully -> STOP (all downstream regret would be meaningless).
#   4. Sweep: sample K request IDs; for each and each action a in A\{plan(r)}
#      (decode instances x {local, each prefill}) write a deviated plan and run it.
#   5. regret: aggregate dev_*.json + baseline.json -> regret.json; echo headline.
#
# |A| for 1P2D = 2 decode x (1 prefill + local) = 4  =>  3 deviations per request.
# Cost: K*(|A|-1) simulations.  Diagnostic (local one-step deviation), NOT the
# global optimum. See FINDINGS.md and the hand-case checks at the bottom.
#
# Usage (from repo root):
#   bash campaigns/edpp-study/repro_counterfactual.sh
#   K=8 TARGET_POLICY=edpp bash campaigns/edpp-study/repro_counterfactual.sh
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
SPEC="${SPEC:-campaigns/edpp-study/specs/synth_cf.yaml}"
TARGET_POLICY="${TARGET_POLICY:-edpp}"
K="${K:-10}"
OUT="${OUT:-campaigns/edpp-study/out/counterfactual}"
DRIVER="campaigns/edpp-study/analyze/counterfactual_regret.py"
DECODES=(instance_1 instance_2)   # 1P2D: prefill=instance_0, decode=instance_1,2
PREFILL="instance_0"
mkdir -p "$OUT"
rm -f "$OUT"/dev_*.json

if [[ ! -x ./blis ]]; then echo "building blis..." >&2; go build -o blis main.go; fi

# SLO knobs: slo_attainment is only populated when --slo-ttft/--slo-itl are set.
SLO=(--slo-ttft "batch=2s" --slo-itl "batch=150ms")
# EDPP coeffs/tau flags only apply to the edpp decider; never/always (used by the
# hand-case sanity checks) reject them.
EDPP=(--pd-decider "$TARGET_POLICY")
if [[ "$TARGET_POLICY" == "edpp" ]]; then
  EDPP+=(--edpp-coeffs "$COEFFS" --edpp-tau-ttft 2s --edpp-tau-itl 150ms)
fi
TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2)
COMMON=(--model "$MODEL" --workload-spec "$SPEC" "${TOPO[@]}" "${SLO[@]}")

goodput() { python3 -c "import json,sys;m=json.load(open(sys.argv[1]));print(m['slo_attainment'] if isinstance(m,dict) else sum(x['slo_attainment'] for x in m)/len(m))" "$1"; }

echo "[1] baseline run (policy=$TARGET_POLICY, K=$K, spec=$SPEC)" >&2
./blis run "${COMMON[@]}" "${EDPP[@]}" \
  --pd-outcome-trace "$OUT/outcome.csv" --metrics-path "$OUT/baseline.json" >/dev/null 2>&1
BASE_G="$(goodput "$OUT/baseline.json")"
echo "    baseline slo_attainment = $BASE_G" >&2

echo "[2] capture-plan -> $OUT/plan.csv" >&2
python3 "$DRIVER" capture-plan --outcome "$OUT/outcome.csv" --out "$OUT/plan.csv"

echo "[3] self-consistency gate: replay plan.csv via --pd-plan" >&2
./blis run "${COMMON[@]}" --pd-plan "$OUT/plan.csv" --metrics-path "$OUT/replay.json" >/dev/null 2>&1
REPLAY_G="$(goodput "$OUT/replay.json")"
echo "    replay slo_attainment   = $REPLAY_G" >&2
if ! python3 -c "import sys;sys.exit(0 if abs(float('$BASE_G')-float('$REPLAY_G'))<1e-9 else 1)"; then
  echo "SELF-CONSISTENCY GATE FAILED: replay ($REPLAY_G) != baseline ($BASE_G)." >&2
  echo "The fixed-plan decider is NOT replaying the captured plan faithfully (INV-6/INV-13). STOP." >&2
  exit 1
fi
echo "    OK: replay reproduces baseline goodput." >&2

echo "[4] deviation sweep: K=$K requests x $(( ${#DECODES[@]} * 2 - 1 )) deviations each" >&2
# Sample K request IDs deterministically (seeded) from the plan.
SAMPLE=()
while IFS= read -r line; do SAMPLE+=("$line"); done < <(python3 - "$OUT/plan.csv" "$K" <<'PY'
import csv,random,sys
rows=list(csv.DictReader(open(sys.argv[1])))
K=min(int(sys.argv[2]),len(rows))
random.seed(1234)
for r in random.sample(rows,K): print(r["request_id"])
PY
)
echo "    sampled: ${SAMPLE[*]}" >&2

n_runs=0
for rid in "${SAMPLE[@]}"; do
  # current plan action for rid
  cur="$(python3 -c "import csv,sys;[print(r['decode_instance']+'|'+r['prefill_instance']) for r in csv.DictReader(open(sys.argv[1])) if r['request_id']==sys.argv[2]]" "$OUT/plan.csv" "$rid")"
  for dec in "${DECODES[@]}"; do
    for pre in local "$PREFILL"; do
      # skip the baseline action itself (normalize local prefill token)
      cur_dec="${cur%%|*}"; cur_pre="${cur##*|}"
      [[ "$cur_pre" == "" ]] && cur_pre="local"
      if [[ "$dec" == "$cur_dec" && "$pre" == "$cur_pre" ]]; then continue; fi
      # The driver regex dev_(.+)_([^_]+)\.json splits on the LAST underscore, so the
      # action token must be underscore-free. Strip underscores from instance names
      # for the filename token only; the plan CSV keeps the real instance names.
      dec_tok="${dec//_/}"; pre_tok="local"; [[ "$pre" != "local" ]] && pre_tok="${pre//_/}"
      action="${dec_tok}-${pre_tok}"               # dev_<reqid>_<decode>-<prefill|local>.json
      devplan="$OUT/dev_${rid}_${action}.plan.csv"
      python3 -c "import csv,sys
rid,dec,pre=sys.argv[2],sys.argv[3],sys.argv[4]
rows=list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    if r['request_id']==rid: r['decode_instance']=dec; r['prefill_instance']=('' if pre=='local' else pre)
w=csv.DictWriter(open(sys.argv[5],'w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader();w.writerows(rows)" \
        "$OUT/plan.csv" "$rid" "$dec" "$pre" "$devplan"
      ./blis run "${COMMON[@]}" --pd-plan "$devplan" --metrics-path "$OUT/dev_${rid}_${action}.json" >/dev/null 2>&1
      n_runs=$((n_runs+1))
    done
  done
done
echo "    completed $n_runs deviation runs" >&2

echo "[5] regret aggregation -> $OUT/regret.json" >&2
python3 "$DRIVER" regret --sweep-dir "$OUT" --out "$OUT/regret.json" >/dev/null
python3 -c "import json;r=json.load(open('$OUT/regret.json'));print('HEADLINE  baseline_goodput=%.4f  n=%d  total_regret=%.4f  mean_regret=%.4f  frac_positive=%.3f'%(r['baseline_goodput'],r['n_requests'],r['total_regret'],r['mean_regret'],r['frac_positive_regret']))"

# ---------------------------------------------------------------------------
# HAND-CASE SANITY CHECKS (Step 3) — confirm the harness reproduces known
# answers on a tiny 2-request idle 1P2D trace. Run manually; both verified
# 2026-07 (blis @ this commit). These do NOT go through the edpp capture path
# (the `never` decider emits no --pd-outcome-trace), so the plans are built by
# hand and fed to `counterfactual_regret.py regret` directly.
#
# Trace: 2 constant requests (isl=256, osl=128, prefix=0), aggregate_rate 0.02
#        => effectively idle, no contention.
#
# CASE A — baseline ALL-LOCAL, loose ttft SLO (batch=2s):
#   baseline goodput = 1.000; every hindsight-best = baseline; total_regret = 0.
#   => On an idle cluster local is already optimal; the harness invents no gains.
#
# CASE B — baseline ALL-DISAGG (decode=instance_1, prefill=instance_0),
#          tight ttft SLO (batch=40ms) so the KV-transfer hop pushes disagg
#          TTFT (~51ms) over the deadline while local (~<40ms) meets it:
#   baseline goodput = 0.000; hindsight-best for the deviated request = a LOCAL
#   action; total_regret > 0 (0.5 for one sampled request of the 2-req trace).
#   => The harness detects that, with no contention, local would have been
#      better than the transfer-penalised disaggregated route (known answer).
# ---------------------------------------------------------------------------
