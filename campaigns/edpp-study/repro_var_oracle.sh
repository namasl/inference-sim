#!/usr/bin/env bash
# Value-at-risk (VaR) drift ORACLE vs the least-ttft one-liner, on a FIXED 1P2D topology.
# Design: docs/superpowers/specs/2026-07-21-edpp-var-oracle-design.md.
# Companion to repro_spectrum.sh; same topology, archetypes, per-archetype auto-SLO, and
# balanced decode scorer, so the numbers are directly comparable to the spectrum table.
#
# THE QUESTION (the bar, §1): does a PERFECT-INFORMATION VaR rule — the drift term re-priced
# in value-at-risk (goodput destroyed among co-residents) instead of work (µs) — CLEARLY beat
# the one-line `least-ttft` baseline on the archetypes where least-ttft currently ties/wins?
#   - If the oracle cannot beat the one-liner, the value-currency idea is dead (a clean stop).
#   - If it clears the bar, a deployable approximation becomes worth building (deferred, §7).
#
# WHY IT IS AN ORACLE: VaR reads each co-resident's TRUE remaining decode steps (un-censored
# TrueRemaining) to project its completion. That is a deliberate, gated INV-9 violation
# (--edpp-rule var emits a loud UPPER-BOUND warning). By default we ALSO add
# --edpp-oracle-output-len (VAROR=1) so the deciding request gets its true o_r too — a fully
# clean ceiling. Set VAROR=0 to drop it (R's own work stays estimated).
#
# THREE KERNELS (§2, all reported):
#   var:flip   - A: binary composite-good flip count (the hyperparameter-free ceiling number)
#   var:util   - B: saturating slack utility (expected to reproduce the neglect trap — measured)
#   var:hazard - C: deadline-slack hazard × delay (the smoothed deployable-target shape)
# REFERENCE COLUMNS: never / always / least-ttft / edpp(dpp), exactly as in repro_spectrum.sh.
#
# JOINT=1 adds --edpp-joint to the var + edpp arms (the joint (decode,prefill) argmin path);
# least-ttft is reduced-only and shown as n/a then. never/always are unaffected.
#
# Usage (from repo root):
#   bash campaigns/edpp-study/repro_var_oracle.sh                 # reduced, balanced, VAROR=1
#   JOINT=1 bash campaigns/edpp-study/repro_var_oracle.sh         # joint argmin path
#   SEEDS="42 7 123" RATES="4 8 16" bash campaigns/edpp-study/repro_var_oracle.sh
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"

MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
SPECDIR="campaigns/edpp-study/specs/var_oracle"
OUT="${OUT:-campaigns/edpp-study/out/var_oracle}"
SCORER="${SCORER:-balanced}"
CAP="${CAP:-16}"
RATES="${RATES:-4 8 16}"
SEEDS="${SEEDS:-42 7 123}"
mkdir -p "$SPECDIR" "$OUT"
[[ -x ./blis ]] || go build -o blis main.go

# Size-aware c_xfer (corrected transfer model), on by default for the EDPP/VaR arms — the same
# corrected basis repro_spectrum.sh uses. CXSIZE=0 reverts to the flat --edpp-c-xfer.
XF=(); XLBL=""
if [[ "${CXSIZE:-1}" == "1" ]]; then XF=(--edpp-c-xfer-size-aware); XLBL=" +c_xfer-size"; fi

# VAROR=1 (default): compose the VaR rule with --edpp-oracle-output-len for a fully clean ceiling.
VOR=(); VORLBL=""
if [[ "${VAROR:-1}" == "1" ]]; then VOR=(--edpp-oracle-output-len); VORLBL=" +oracle-o_r"; fi

# JOINT=1: enumerate (decode, prefill) candidates and pick the VaR/dpp argmin.
JF=(); JLBL="reduced"
if [[ "${JOINT:-0}" == "1" ]]; then JF=(--edpp-joint); JLBL="JOINT"; fi

# VARCONGW=<w>: drift-plus-VaR — keep the congestion drift and add VaR, with congestion weight w.
# NORM=1: add per-decision min-max auto-normalization (weight becomes scale-free ≈1).
CONG=(); CONGLBL=""
if [[ -n "${VARCONGW:-}" ]]; then
  CONG=(--edpp-var-congestion --edpp-var-congestion-weight "$VARCONGW"); CONGLBL=" +cong(w=$VARCONGW)"
  if [[ "${NORM:-0}" == "1" ]]; then CONG+=(--edpp-var-normalize); CONGLBL="$CONGLBL+norm"; fi
fi

TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2 --max-num-running-reqs "$CAP")
case "$SCORER" in
  llmd)     DEC=() ;;
  balanced) DEC=(--decode-routing-scorers "queue-depth:1") ;;
  *) echo "SCORER must be llmd|balanced" >&2; exit 1 ;;
esac

arch_dims(){
  case "$1" in
    decode)        echo "256 512"   ;;
    mixed)         echo "2048 128"  ;;
    prefill_lean)  echo "8192 64"   ;;
    prefill_bound) echo "16000 16"  ;;
    *) echo "unknown archetype: $1" >&2; exit 1 ;;
  esac
}
ARCH_ORDER="${ARCH_ORDER:-decode mixed prefill_lean prefill_bound}"

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

# auto_slo(): idle probe -> per-archetype SLO so goodput is comparable across archetypes.
# Uses a FIXED probe seed so the goalposts do not move when $SEEDS is swept.
auto_slo(){ # $1=in $2=out ; sets globals SLO_E2E SLO_TTFT
  spec "$1" "$2" 0.5 "${SLO_PROBE_SEED:-42}"
  ./blis run --model "$MODEL" --workload-spec "$SPECDIR/w.yaml" "${TOPO[@]}" ${DEC[@]+"${DEC[@]}"} \
    --pd-decider always --slo-ttft "standard=999s" --slo-itl "standard=999s" --slo-e2e "standard=999s" \
    --metrics-path "$OUT/idle.json" >/dev/null 2>&1
  local ie it; ie=$(val "$OUT/idle.json" e2e_p99_ms); it=$(val "$OUT/idle.json" ttft_p99_ms)
  SLO_E2E=$(python3 -c "print(max(int($ie*2),1000))"); SLO_TTFT=$(python3 -c "print(max(int($it*3),1000))")
  IDLE_E2E=$ie; IDLE_TTFT=$it
}

run_policy(){ # $1=tag $2..=extra flags ; echoes goodput
  local tag="$1"; shift
  ./blis run --model "$MODEL" --workload-spec "$SPECDIR/w.yaml" "${TOPO[@]}" ${DEC[@]+"${DEC[@]}"} "$@" \
    --slo-ttft "standard=${SLO_TTFT}ms" --slo-itl "standard=100ms" --slo-e2e "standard=${SLO_E2E}ms" \
    --metrics-path "$OUT/$tag.json" >"$OUT/$tag.out" 2>/dev/null || true
  gp "$OUT/$tag.json"
}

echo "VaR ORACLE vs least-ttft [$JLBL$VORLBL$XLBL$CONGLBL]  topology=1P2D cap=$CAP scorer=$SCORER seeds=[$SEEDS]" >&2
echo "(goodput = standard-class slo_attainment; higher is better. BAR: var:* should CLEARLY beat least-ttft.)" >&2
for name in $ARCH_ORDER; do
  set -- $(arch_dims "$name"); IN=$1; O=$2
  auto_slo "$IN" "$O"
  echo "== $name (in=$IN out=$O) idle e2e_p99=${IDLE_E2E}ms ttft_p99=${IDLE_TTFT}ms -> SLO e2e=${SLO_E2E}ms ttft=${SLO_TTFT}ms ==" >&2
  printf "   %-5s %-5s| %-7s %-7s %-11s %-8s %-9s %-9s %-11s\n" \
    "rate" "seed" "never" "always" "least-ttft" "edpp" "var:flip" "var:util" "var:hazard" >&2
  # Common EDPP basis (rollforward estimator, real τ_itl). VaR arms add --edpp-rule var,
  # --edpp-var-metric, and --edpp-tau-e2e (so g()'s E2E composite matches the goodput SLO).
  for r in $RATES; do
    for s in $SEEDS; do
      spec "$IN" "$O" "$r" "$s"
      EC=(--edpp-coeffs "$COEFFS" --edpp-tadm-estimator rollforward --edpp-tau-itl 100ms --edpp-tau-ttft "${SLO_TTFT}ms")
      VC=("${EC[@]}" --edpp-tau-e2e "${SLO_E2E}ms" --edpp-rule var)
      N=$(run_policy vo_n --pd-decider never)
      A=$(run_policy vo_a --pd-decider always)
      if [[ "${JOINT:-0}" == "1" ]]; then L="n/a"; else
        L=$(run_policy vo_l --pd-decider edpp "${EC[@]}" ${XF[@]+"${XF[@]}"} --edpp-rule least-ttft)
      fi
      E=$(run_policy vo_e --pd-decider edpp "${EC[@]}" ${JF[@]+"${JF[@]}"} ${XF[@]+"${XF[@]}"})
      VF=$(run_policy vo_vf --pd-decider edpp "${VC[@]}" ${JF[@]+"${JF[@]}"} ${VOR[@]+"${VOR[@]}"} ${XF[@]+"${XF[@]}"} ${CONG[@]+"${CONG[@]}"} --edpp-var-metric flip)
      VU=$(run_policy vo_vu --pd-decider edpp "${VC[@]}" ${JF[@]+"${JF[@]}"} ${VOR[@]+"${VOR[@]}"} ${XF[@]+"${XF[@]}"} ${CONG[@]+"${CONG[@]}"} --edpp-var-metric util)
      VH=$(run_policy vo_vh --pd-decider edpp "${VC[@]}" ${JF[@]+"${JF[@]}"} ${VOR[@]+"${VOR[@]}"} ${XF[@]+"${XF[@]}"} ${CONG[@]+"${CONG[@]}"} --edpp-var-metric hazard)
      printf "   %-5s %-5s| %-7s %-7s %-11s %-8s %-9s %-9s %-11s\n" "$r" "$s" "$N" "$A" "$L" "$E" "$VF" "$VU" "$VH" >&2
    done
  done
done
echo "done -> $OUT" >&2
