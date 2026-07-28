#!/usr/bin/env bash
# Heterogeneity ratio sweep, protocol v2.1 (2026-07-28) — replaces
# repro_hetero_ratio_sweep_gp.sh, which ran at an asserted operating point
# (rate 10, ttft 60s / e2e 8s / itl 500ms) where the audit showed no signal was
# live (memory: edpp-eval-audit-2026-07-27).
#
# v2.1 protocol per ratio N in {1.0 1.2 1.5 2.0 3.0 5.0}:
#   - slow decode instance = uniform Nx slowdown coeffs (existing ratio ladder files)
#   - SLOs DERIVED from the fast-fleet baseline (same for all N): ttft 259ms (5x),
#     itl 67ms (4x), e2e 6208ms (3x) — the hetero grid-v2.1 cell's targets
#   - per-N capacity knee measured with an `always` mini-sweep (achieved/offered >= 0.95)
#   - arms run AT the knee (1.0x), 5 seeds, steady-state n = rate x 120s
#   - one scorer config (queue-depth:1), batch cap 16
#   - reference: best static fast/slow split (--pd-plan), fraction grid-searched
#     on seed 42 at the same operating point, evaluated over all seeds
# Usage: bash campaigns/edpp-study/repro_ratio_sweep_v2.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /Users/vishakha/git-repos/llm-git-repos/edpp-fresh/inference-sim)"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEF="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
D=campaigns/edpp-study/specs/ratio_v2
OUT="${OUT:-campaigns/edpp-study/out/ratio_v2}"
SEEDS="${SEEDS:-42 7 123 5 11}"
RATIOS="${RATIOS:-1.0 1.2 1.5 2.0 3.0 5.0}"
KBETAS="${KBETAS:-0.25 0.5 1.0}"
T=259; I=67; E=6208
mkdir -p "$D" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

tag(){ printf 'ratio%s' "$(echo "$1" | tr '.' 'p')"; }
gp(){ python3 -c "import json;print('%.4f'%json.load(open('$1'))['per_class']['standard']['slo_attainment'])" 2>/dev/null||echo NA; }
ach(){ python3 -c "import json;print('%.3f'%json.load(open('$1'))['responses_per_sec'])" 2>/dev/null||echo 0; }
m(){ python3 -c "import sys;xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')];print('%.3f'%(sum(xs)/len(xs)) if xs else 'NA')" "$@"; }

write_bundle(){ local N="$1"
  local t; t="$(tag "$N")"
  local SLOWC="scripts/calibration/coeffs-llama70b-$t-tp4.json"
  [[ -f "$SLOWC" ]] || { echo "missing $SLOWC — generate with repro_theta_by_gpu.sh" >&2; exit 1; }
  local TF BW; TF=$(python3 -c "print(f'{1979.0/$N:.4f}')"); BW=$(python3 -c "print(f'{3.35/$N:.5f}')")
  cat > "$D/bundle_$t.yaml" <<YAML
node_pools:
  - {name: fast, gpu_type: H100, gpus_per_node: 8, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
  - {name: slow, gpu_type: A100, gpus_per_node: 4, gpu_memory_gib: 80.0, initial_nodes: 1, min_nodes: 1, max_nodes: 1, cost_per_hour: 0.0, provisioning_delay: {mean: 0.0, stddev: 0.0}}
hw_config_by_gpu:
  H100: {tflops_peak: 1979.0, bw_peak_tbs: 3.35, mfu_prefill: 0.5, mfu_decode: 0.5}
  A100: {tflops_peak: $TF, bw_peak_tbs: $BW, mfu_prefill: 0.5, mfu_decode: 0.5}
coeffs_by_gpu:
  H100: $COEF
  A100: $SLOWC
YAML
  echo "$D/bundle_$t.yaml"
}

spec(){ # rate n seed file (workload fixed: 256 in / lognormal mean-64 out)
  python3 - "$1" "$2" "$3" > "$4" <<'PY'
import sys, math
rate,n,seed = sys.argv[1:4]
mu = math.log(64) - 0.08
print(f"""version: "2"
seed: {seed}
category: language
aggregate_rate: {rate}
num_requests: {n}
clients:
  - {{id: w, tenant_id: t, slo_class: standard, rate_fraction: 1.0, streaming: false, arrival: {{process: poisson}}, input_distribution: {{type: constant, params: {{value: 256}}}}, output_distribution: {{type: lognormal, params: {{mu: {mu:.4f}, sigma: 0.4, min: 4, max: 528}}}}, prefix_group: g, prefix_length: 0}}""")
PY
}

TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --decode-routing-scorers "queue-depth:1" --max-num-running-reqs 16)
SLO=(--slo-ttft "standard=${T}ms" --slo-itl "standard=${I}ms" --slo-e2e "standard=${E}ms")
EC=(--edpp-coeffs "$COEF" --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware --edpp-tau-itl "${I}ms")
VVF=(--pd-decider edpp "${EC[@]}" --edpp-rule var --edpp-var-metric util --edpp-joint --edpp-var-congestion --edpp-var-normalize --edpp-var-congestion-weight 1 --edpp-var-deployable --edpp-var-goodput --edpp-tau-ttft "${T}ms" --edpp-tau-e2e "${E}ms")

run(){ local tg="$1" bundle="$2" seed="$3" wf="$4"; shift 4
  ./blis run --model "$MODEL" --workload-spec "$wf" "${TOPO[@]}" --policy-config "$bundle" "${SLO[@]}" \
    --seed "$seed" "$@" --metrics-path "$OUT/$tg.json" >/dev/null 2>&1 || true
  gp "$OUT/$tg.json"; }

write_plan(){ local frac="$1" n="$2"
  python3 -c "
import csv
w=csv.DictWriter(open('$OUT/plan.csv','w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for i in range($n): w.writerow({'request_id':f'request_{i}','decode_instance':('instance_1' if i%100<$frac else 'instance_2'),'prefill_instance':'instance_0'})"
}

CSV="$OUT/ratio_v2.csv"; echo "ratio,knee,arm,seed,goodput" > "$CSV"
emit(){ echo "$1,$2,$3,$4,$5" >> "$CSV"; }
printf "%-5s %-6s| %-7s %-7s %-7s %-8s %-7s | %-7s\n" \
  "N" "knee" "never" "always" "kairos*" "lt-joint" "dpvar" "optimum"

for N in $RATIOS; do
  BUNDLE="$(write_bundle "$N")"
  # --- per-N knee: always mini-sweep, achieved/offered >= 0.95 ---
  KNEE=""
  for R in 8 10 12 14 16 18 20 24; do
    spec "$R" 600 42 "$D/wk.yaml"
    ./blis run --model "$MODEL" --workload-spec "$D/wk.yaml" "${TOPO[@]}" --policy-config "$BUNDLE" \
      --slo-ttft standard=9999s --slo-itl standard=9999s --slo-e2e standard=9999s \
      --pd-decider always --seed 42 --metrics-path "$OUT/knee_${N}_$R.json" >/dev/null 2>&1 || true
    A=$(ach "$OUT/knee_${N}_$R.json")
    ok=$(python3 -c "print(1 if $A >= 0.95*$R else 0)")
    [[ "$ok" == 1 ]] && KNEE=$R || break
  done
  [[ -n "$KNEE" ]] || KNEE=8
  RATE=$KNEE
  NREQ=$(python3 -c "import math;print(max(int(math.ceil($RATE*120)),600))")

  # --- best static split: grid search on seed 42 at the knee ---
  spec "$RATE" "$NREQ" 42 "$D/w42.yaml"
  best_frac=0; best_gp=-1
  for frac in 0 10 20 30 40 50 60 70 80 90 100; do
    write_plan "$frac" "$NREQ"
    g=$(run "o_${N}_${frac}" "$BUNDLE" 42 "$D/w42.yaml" --pd-plan "$OUT/plan.csv")
    awk "BEGIN{exit !($g > $best_gp)}" 2>/dev/null && { best_gp=$g; best_frac=$frac; }
  done
  write_plan "$best_frac" "$NREQ"

  declare -a NV=() A=() K=() LJ=() VV=() O=()
  for s in $SEEDS; do
    WF="$D/w_${N}_$s.yaml"; spec "$RATE" "$NREQ" "$s" "$WF"
    nv=$(run "nv_${N}_$s" "$BUNDLE" "$s" "$WF" --pd-decider never);  NV+=("$nv"); emit "$N" "$KNEE" never "$s" "$nv"
    a=$(run "a_${N}_$s" "$BUNDLE" "$s" "$WF" --pd-decider always);   A+=("$a");   emit "$N" "$KNEE" always "$s" "$a"
    kb=(); for bb in $KBETAS; do kb+=("$(run "k_${N}_${s}_$bb" "$BUNDLE" "$s" "$WF" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${T}ms" --edpp-rule kairos --kairos-beta "$bb")"); done
    k=$(printf '%s\n' "${kb[@]}" | sort -g | tail -1);               K+=("$k");   emit "$N" "$KNEE" kairos "$s" "$k"
    lj=$(run "lj_${N}_$s" "$BUNDLE" "$s" "$WF" --pd-decider edpp "${EC[@]}" --edpp-tau-ttft "${T}ms" --edpp-rule least-ttft --edpp-joint); LJ+=("$lj"); emit "$N" "$KNEE" lt-joint "$s" "$lj"
    vv=$(run "vv_${N}_$s" "$BUNDLE" "$s" "$WF" "${VVF[@]}");         VV+=("$vv"); emit "$N" "$KNEE" dpvar "$s" "$vv"
    o=$(run "oe_${N}_$s" "$BUNDLE" "$s" "$WF" --pd-plan "$OUT/plan.csv"); O+=("$o"); emit "$N" "$KNEE" optimum "$s" "$o"
  done
  printf "%-5s %-6s| %-7s %-7s %-7s %-8s %-7s | %-7s (opt=%s%% fast)\n" \
    "$N" "$KNEE" "$(m "${NV[@]}")" "$(m "${A[@]}")" "$(m "${K[@]}")" "$(m "${LJ[@]}")" "$(m "${VV[@]}")" "$(m "${O[@]}")" "$best_frac"
done
echo "done -> $CSV"
