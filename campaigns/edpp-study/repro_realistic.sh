#!/usr/bin/env bash
# Realistic re-baseline of the EDPP policy study (v2). Replaces the constant-token caricatures
# (256/512, 16000/16, ...) with PROVENANCE-BACKED inference-perf catalog workloads that have
# realistic, VARIABLE output lengths, and replaces the fixed rate grid (4/8/16) with a per-archetype
# load sweep normalized to utilization ρ = achieved_throughput / λ*.
#
# Archetypes (variable output; see spec headers for catalog provenance):
#   decode   = synthetic-data-generation (in ~500 tail→15k, OUT ~4000±2500) — decode-bound
#   balanced = mixed (fabricated: ~1500 in / ~1500 out)
#   prefill  = summarization-rag (2-client ~2k/~60k in, OUT ~500±300) — prefill-bound
#
# SLOs are DERIVED per archetype (not asserted): the TTFT/ITL p99 the server delivers at ρ≈0.75–0.85
# ("70–80% utilized"), measured by MODE=lambda. λ* = achieved-throughput plateau (SLO-free ⇒ no
# circularity). ITL comes out ~uniform (~50–60ms); TTFT is per-workload (170ms decode/balanced,
# 2300ms prefill — the 60k prefill). Size-aware c_xfer ON; NO artificial concurrency cap.
#
# MODES:  lambda = λ*/SLO derivation sweep   policy = never/always/prefix/least-ttft/edpp (E2)
#         scorer = llm-d vs balanced (E1)     oracle = static-fraction yardstick + dynamic (E3/E4)
set -u
cd "$(dirname "$0")/../.." || exit 1
[[ -x ./blis ]] || go build -o blis main.go
MODE="${MODE:-policy}"
MODEL="${MODEL:-meta-llama/llama-3.3-70b-instruct}"
COEFFS="${COEFFS:-scripts/calibration/coeffs-llama70b-h100-tp4.json}"
SEED="${SEED:-42}"
NREQ="${NREQ:-600}"
OUT="campaigns/edpp-study/out/realistic"; mkdir -p "$OUT"
TOPO=(--num-instances 3 --prefill-instances 1 --decode-instances 2)   # NO --max-num-running-reqs
SCORER="${SCORER:-balanced}"
case "$SCORER" in llmd) DEC=() ;; balanced) DEC=(--decode-routing-scorers "queue-depth:1") ;; esac
ARCHES="${ARCHES:-decode balanced prefill}"

# --- per-archetype config (case, not assoc array: macOS bash 3.2) ---
base_for(){ case "$1" in
  decode)   echo "inference-perf-batch-synthetic-data-generation.yaml" ;;
  balanced) echo "campaigns/edpp-study/specs/mixed_rate1.0.yaml" ;;
  prefill)  echo "inference-perf-batch-summarization-rag.yaml" ;;
esac; }
# offered rates mapped to ρ = {0.5, 0.7, 0.85, 1.0, 1.2} from the measured utilization curve (MODE=lambda)
rates_for(){ case "$1" in
  decode)   echo "0.75 1.0 1.5 2.5 3.5" ;;
  balanced) echo "3 4 6 8 12" ;;
  prefill)  echo "1.5 2.0 2.5 3.0 4.0" ;;
esac; }
rho_labels(){ echo "0.5 0.7 0.85 1.0 1.2"; }
ttft_for(){ case "$1" in decode) echo 170 ;; balanced) echo 180 ;; prefill) echo 2300 ;; esac; }   # ms, p99@~75-85% util
itl_for(){  case "$1" in decode) echo 60  ;; balanced) echo 50  ;; prefill) echo 60   ;; esac; }   # ms
in_out(){   case "$1" in decode) echo "~500/4000" ;; balanced) echo "~1500/1500" ;; prefill) echo "~2k+60k/500" ;; esac; }

rw(){ # $1=base $2=rate -> $OUT/w.yaml  (rewrite offered rate, request count, seed)
  python3 - "$1" "$2" "$NREQ" "$SEED" > "$OUT/w.yaml" <<'PY'
import sys,re
t=open(sys.argv[1]).read()
t=re.sub(r"^aggregate_rate:.*$",f"aggregate_rate: {sys.argv[2]}",t,flags=re.M)
t=re.sub(r"^num_requests:.*$",f"num_requests: {sys.argv[3]}",t,flags=re.M)
t=re.sub(r"^seed:.*$",f"seed: {sys.argv[4]}",t,flags=re.M)
sys.stdout.write(t)
PY
}
# count-weighted goodput over classes that actually have requests (skip phantom 0-count class)
gp(){ python3 -c "import json;d=json.load(open('$1'));pc=d.get('per_class',{});vs=[(v['count'],v['slo_attainment']) for v in pc.values() if v.get('count',0)>0];print(round(sum(c*a for c,a in vs)/sum(c for c,_ in vs),3) if vs else 'na')"; }
split(){ grep -oE "decode_instance=instance_[12]" "$1" 2>/dev/null | sort | uniq -c | awk '{printf "%s=%s ",$2,$1}'; }

run(){ # $1=tag $2=arch $3..=extra flags ; echoes goodput
  local tag="$1" arch="$2"; shift 2
  ./blis run --model "$MODEL" --workload-spec "$OUT/w.yaml" "${TOPO[@]}" ${DEC[@]+"${DEC[@]}"} "$@" \
    --slo-ttft "standard=$(ttft_for "$arch")ms,batch=$(ttft_for "$arch")ms" \
    --slo-itl  "standard=$(itl_for "$arch")ms,batch=$(itl_for "$arch")ms" \
    --slo-e2e  "standard=999s,batch=999s" \
    --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true
  gp "$OUT/$tag.json"
}
EC(){ echo "--pd-decider edpp --edpp-coeffs $COEFFS --edpp-tadm-estimator rollforward --edpp-c-xfer-size-aware"; }

if [[ "$MODE" == "policy" ]]; then
  echo "REALISTIC POLICY COMPARISON  1P2D  scorer=$SCORER  seed=$SEED  size-aware c_xfer  (goodput)" >&2
  echo "SLO = per-archetype TTFT/ITL p99 @ ρ≈0.75-0.85 (derived).  Load axis = ρ (utilization)." >&2
  for arch in $ARCHES; do
    b="$(base_for "$arch")"; set -- $(rates_for "$arch"); rlist="$*"; set -- $(rho_labels); rholist="$*"
    echo "== $arch (in/out $(in_out "$arch"))  TTFT=$(ttft_for "$arch")ms ITL=$(itl_for "$arch")ms ==" >&2
    printf "   %-5s %-6s| %-8s %-8s %-8s %-10s %-8s\n" "ρ" "rate" "never" "always" "prefix16" "least-ttft" "edpp" >&2
    i=1
    for r in $rlist; do
      rho=$(echo $rholist | cut -d' ' -f$i); i=$((i+1))
      rw "$b" "$r"
      N=$(run "n" "$arch" --pd-decider never)
      A=$(run "a" "$arch" --pd-decider always)
      P=$(run "p" "$arch" --pd-decider prefix-threshold --pd-prefix-threshold 16)
      L=$(run "l" "$arch" $(EC) --edpp-rule least-ttft)
      E=$(run "e" "$arch" $(EC))
      printf "   %-5s %-6s| %-8s %-8s %-8s %-10s %-8s\n" "$rho" "$r" "$N" "$A" "$P" "$L" "$E" >&2
    done
  done
  echo "DONE" >&2

elif [[ "$MODE" == "ablate" ]]; then
  # Term ablation on realistic workloads. Arms reachable from config (size-aware c_xfer ON in all):
  #   least-ttft | drift-only (--edpp-tau-ttft 999s --edpp-v 0) | drift+z (--edpp-v 0) | full
  # ORACLE=1 adds --edpp-oracle-output-len to every arm (true o_r; variable output ⇒ should bite now).
  OF=""; OLBL=""
  if [[ "${ORACLE:-0}" == "1" ]]; then OF="--edpp-oracle-output-len"; OLBL=" +oracle-o_r"; fi
  JF=""; JLBL="reduced"
  if [[ "${JOINT:-0}" == "1" ]]; then JF="--edpp-joint"; JLBL="JOINT"; fi   # E9: joint P/D argmin; least-ttft absent
  echo "REALISTIC TERM ABLATION [$JLBL$OLBL]  1P2D scorer=$SCORER seed=$SEED size-aware c_xfer  (goodput)" >&2
  for arch in $ARCHES; do
    b="$(base_for "$arch")"; set -- $(rates_for "$arch"); rlist="$*"; set -- $(rho_labels); rholist="$*"
    T=$(ttft_for "$arch")
    echo "== $arch (in/out $(in_out "$arch"))  TTFT=${T}ms ITL=$(itl_for "$arch")ms ==" >&2
    printf "   %-5s %-6s| %-10s %-10s %-9s %-8s\n" "ρ" "rate" "least-ttft" "drift-only" "drift+z" "full" >&2
    i=1
    for r in $rlist; do
      rho=$(echo $rholist | cut -d' ' -f$i); i=$((i+1)); rw "$b" "$r"
      if [[ "${JOINT:-0}" == "1" ]]; then L="n/a"; else L=$(run "abl" "$arch" $(EC) $OF --edpp-rule least-ttft); fi
      D=$(run "abd" "$arch" $(EC) $JF $OF --edpp-tau-ttft 999s --edpp-v 0)
      Z=$(run "abz" "$arch" $(EC) $JF $OF --edpp-tau-ttft "${T}ms" --edpp-v 0)
      F=$(run "abf" "$arch" $(EC) $JF $OF --edpp-tau-ttft "${T}ms")
      printf "   %-5s %-6s| %-10s %-10s %-9s %-8s\n" "$rho" "$r" "$L" "$D" "$Z" "$F" >&2
    done
  done
  echo "DONE" >&2

elif [[ "$MODE" == "scorer" ]]; then
  # E1: llm-d shipped decode scorer (precise-prefix-cache:2,queue-depth:1) vs load-balanced (queue-depth:1).
  # With a shared prefix group the prefix scorer can pin all decode onto one instance. Run at ρ≈0.85.
  echo "REALISTIC SCORER COMPARISON  1P2D seed=$SEED  (goodput; llm-d default vs balanced)  @ρ≈0.85" >&2
  for arch in $ARCHES; do
    b="$(base_for "$arch")"; set -- $(rates_for "$arch"); r=$(echo "$*" | cut -d' ' -f3); rw "$b" "$r"
    echo "== $arch (in/out $(in_out "$arch")) rate=$r ==" >&2
    for sc in llmd balanced; do
      case "$sc" in llmd) DEC=() ;; balanced) DEC=(--decode-routing-scorers "queue-depth:1") ;; esac
      for pol in never always; do
        g=$(run "sc_${sc}_${pol}" "$arch" --pd-decider "$pol")
        printf "   %-9s %-7s goodput=%s\n" "$sc" "$pol" "$g" >&2
      done
    done
  done
  DEC=(--decode-routing-scorers "queue-depth:1"); echo "DONE" >&2

elif [[ "$MODE" == "oracle" ]]; then
  # E3: static disaggregation-fraction yardstick via --pd-plan, + dynamic edpp/least-ttft, per archetype
  # at a saturating ρ (default the ρ=0.85 point). Interior max ⇒ neither corner optimal.
  for arch in $ARCHES; do
    b="$(base_for "$arch")"; set -- $(rates_for "$arch"); r=$(echo "$*" | cut -d' ' -f${ORHO:-4}); rw "$b" "$r"
    echo "== STATIC-FRACTION ORACLE  $arch rate=$r seed=$SEED ==" >&2
    for f in ${FRACS:-0 20 35 50 75 100}; do
      python3 -c "
import csv
w=csv.DictWriter(open('$OUT/plan.csv','w',newline=''),fieldnames=['request_id','decode_instance','prefill_instance']);w.writeheader()
for i in range($NREQ):
    w.writerow({'request_id':f'request_{i}','decode_instance':('instance_1' if i%2==0 else 'instance_2'),'prefill_instance':('instance_0' if (i%100)<$f else '')})"
      g=$(run "f$f" "$arch" --pd-plan "$OUT/plan.csv")
      printf "   f=%-3s goodput=%s\n" "$f" "$g" >&2
    done
    printf "   %-12s goodput=%s\n" "least-ttft" "$(run o_l "$arch" $(EC) --edpp-rule least-ttft)" >&2
    printf "   %-12s goodput=%s\n" "edpp"       "$(run o_e "$arch" $(EC))" >&2
  done
  echo "DONE" >&2

elif [[ "$MODE" == "class" ]]; then
  # E6/E7: SLO-class heterogeneity on the realistic 2-class RAG workload (interactive vector-qa
  # [standard] 78% + batch doc-read 22%, shared prefix). Per-class goodput under never/always and
  # EDPP with a SINGLE τ vs PER-CLASS τ. Does per-class machinery protect the interactive class?
  b="campaigns/edpp-study/specs/rag_rate2.0.yaml"; [[ -f "$b" ]] || b="inference-perf-batch-summarization-rag.yaml"
  R="${RATE:-2.5}"; rw "$b" "$R"
  STT="standard=500ms,batch=6000ms"; SIT="standard=60ms,batch=200ms"   # goodput SLOs (all arms)
  gpc(){ python3 -c "import json;v=json.load(open('$1')).get('per_class',{}).get('$2',{});print(v.get('slo_attainment') if v.get('count',0)>0 else 'na')"; }
  runc(){ local tag="$1"; shift; ./blis run --model "$MODEL" --workload-spec "$OUT/w.yaml" "${TOPO[@]}" ${DEC[@]+"${DEC[@]}"} "$@" \
      --slo-ttft "$STT" --slo-itl "$SIT" --slo-e2e "standard=999s,batch=999s" --metrics-path "$OUT/$tag.json" >/dev/null 2>&1 || true; }
  echo "REALISTIC SLO-CLASS (E6/E7)  RAG 2-class  rate=$R seed=$SEED size-aware c_xfer" >&2
  echo "  goodput SLO: interactive(standard) TTFT500/ITL60 ; batch TTFT6s/ITL200" >&2
  printf "   %-18s %-12s %-12s\n" "arm" "interactive" "batch" >&2
  runc never   --pd-decider never
  runc always  --pd-decider always
  runc edpp1   $(EC) --edpp-tau-ttft 500ms
  runc edpppc  $(EC) --edpp-tau-ttft-classes "standard=500ms,batch=6000ms" --edpp-tau-itl-classes "standard=60ms,batch=200ms"
  for a in never always edpp1 edpppc; do
    printf "   %-18s %-12s %-12s\n" "$a" "$(gpc "$OUT/$a.json" standard)" "$(gpc "$OUT/$a.json" batch)" >&2
  done
  echo "DONE" >&2

elif [[ "$MODE" == "lambda" ]]; then
  RATES="${RATES:-0.25 0.5 0.75 1.0 1.5 2.0 3.0}"
  for arch in $ARCHES; do
    b="$(base_for "$arch")"
    echo "==================== $arch  ($b) ====================" >&2
    printf "  %-6s %-9s %-9s %-9s %-9s %-s\n" "rate" "tok/s" "goodput" "ttftp99" "itlp99" "verdict" >&2
    for r in $RATES; do
      rw "$b" "$r"
      ./blis run --model "$MODEL" --workload-spec "$OUT/w.yaml" "${TOPO[@]}" ${DEC[@]+"${DEC[@]}"} \
        $(EC) --slo-ttft "standard=$(ttft_for "$arch")ms,batch=$(ttft_for "$arch")ms" \
        --slo-itl "standard=$(itl_for "$arch")ms,batch=$(itl_for "$arch")ms" --slo-e2e "standard=999s,batch=999s" \
        --saturation-report "$OUT/sat.json" --metrics-path "$OUT/m_${arch}_$r.json" >/dev/null 2>&1
      tps=$(python3 -c "import json;print(int(json.load(open('$OUT/m_${arch}_$r.json')).get('tokens_per_sec',0)))")
      g=$(gp "$OUT/m_${arch}_$r.json")
      t=$(python3 -c "import json;print(int(json.load(open('$OUT/m_${arch}_$r.json')).get('ttft_p99_ms',0)))")
      il=$(python3 -c "import json;print(round(json.load(open('$OUT/m_${arch}_$r.json')).get('itl_p99_ms',0),1))")
      v=$(python3 -c "import json;print(json.load(open('$OUT/sat.json')).get('classification','?'))" 2>/dev/null||echo '?')
      printf "  %-6s %-9s %-9s %-9s %-9s %-s\n" "$r" "$tps" "$g" "$t" "$il" "$v" >&2
    done
  done
  echo "DONE" >&2
fi
