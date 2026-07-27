#!/usr/bin/env bash
# Reproduce every simulation-based figure and table in the INFOCOM joint PD-routing paper.
# Excludes the real-vLLM calibration figures (titer_modeb.png, latency_fit.png) — those need a live
# GPU server and live under scripts/calibration/. See REPRODUCE_PAPER.md for the figure-by-figure map.
#
# Run from anywhere; the script cd's to the repo root. Deterministic, CPU-only.
#   bash campaigns/edpp-study/reproduce_paper.sh
#   SEEDS="42" bash campaigns/edpp-study/reproduce_paper.sh          # faster smoke, numbers drift a little
#   COPY_TO_PAPER=1 bash campaigns/edpp-study/reproduce_paper.sh     # also copy PNGs into ../infocom/figures
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"; cd "$REPO"
STUDY="campaigns/edpp-study"
ANALYZE="$STUDY/analyze"
COPY_TO_PAPER="${COPY_TO_PAPER:-0}"
PAPER_FIGS="$REPO/../infocom/figures"

command -v go       >/dev/null || { echo "need go on PATH" >&2; exit 1; }
command -v python3  >/dev/null || { echo "need python3 on PATH" >&2; exit 1; }

echo ">>> building blis" >&2
[[ -x ./blis ]] || go build -o blis main.go

step() { echo; echo "============================================================"; echo ">>> $1"; echo "============================================================"; }

# --- Fig ttft_d_admission.png --------------------------------------------------
step "1/5  collocated TTFT estimator (repro_ttft_d_local.sh)"
bash "$STUDY/repro_ttft_d_local.sh"

# --- Fig ttft_p_disagg.png -----------------------------------------------------
step "2/5  disaggregated prefill TTFT (repro_ttft_p_local.sh)"
bash "$STUDY/repro_ttft_p_local.sh"

# --- Fig hetero_ratio_sweep.png ------------------------------------------------
step "3/5  heterogeneity ratio sweep (repro_hetero_ratio_sweep_gp.sh + plot)"
bash "$STUDY/repro_hetero_ratio_sweep_gp.sh"
python3 "$ANALYZE/hetero_ratio_sweep.py" \
  --csv "$STUDY/out/hetero_ratio/ratio_sweep.csv" \
  --out "$STUDY/out/hetero_ratio/hetero_ratio_sweep.png"

# --- Fig pd_provisioning.png ---------------------------------------------------
step "4/5  topology / provisioning matrix (repro_topology_matrix_gp.sh + plot)"
bash "$STUDY/repro_topology_matrix_gp.sh"
python3 "$ANALYZE/topology_matrix.py" \
  --csv "$STUDY/out/topo_matrix/topo_matrix.csv" \
  --out "$STUDY/out/topo_matrix/pd_provisioning.png"

# --- Fig policy_grid.png + Tab grid + Tab regret -------------------------------
step "5/5  goodput dominance grid (repro_var_dominance_goodput.sh)"
GRID_LOG="$STUDY/out/var_dom_gp/grid.txt"; mkdir -p "$STUDY/out/var_dom_gp"
# The campaign prints the grid and per-policy worst-case regret to stderr; capture it for transcription.
bash "$STUDY/repro_var_dominance_goodput.sh" 2> >(tee "$GRID_LOG" >&2)

# --- Summary -------------------------------------------------------------------
step "DONE — generated artifacts"
cat <<EOF >&2
Auto-plotted PNGs (compare against FINDINGS.md checkpoints):
  ttft_d_admission.png  <-  $STUDY/out/ttft_d_local/fig_admission.png
  ttft_p_disagg.png     <-  $STUDY/out/ttft_p_local/fig_ttft_p.png
  hetero_ratio_sweep.png <- $STUDY/out/hetero_ratio/hetero_ratio_sweep.png
  pd_provisioning.png   <-  $STUDY/out/topo_matrix/pd_provisioning.png

Manual (transcription) artifacts from the goodput grid — see $GRID_LOG:
  policy_grid.png       <-  transcribe grid into ../infocom/figures/plot_policy_grid.py (data={...}), then run it
  Tab. grid / regret    <-  transcribe per-cell goodputs + worst-case regret column into the paper .tex
  Headline to confirm:  deployable full rule worst-case regret 0.034, true-length 0.060, alternatives 0.39-0.94
EOF

if [[ "$COPY_TO_PAPER" == "1" ]]; then
  if [[ -d "$PAPER_FIGS" ]]; then
    cp "$STUDY/out/ttft_d_local/fig_admission.png"        "$PAPER_FIGS/ttft_d_admission.png"
    cp "$STUDY/out/ttft_p_local/fig_ttft_p.png"           "$PAPER_FIGS/ttft_p_disagg.png"
    cp "$STUDY/out/hetero_ratio/hetero_ratio_sweep.png"   "$PAPER_FIGS/hetero_ratio_sweep.png"
    cp "$STUDY/out/topo_matrix/pd_provisioning.png"       "$PAPER_FIGS/pd_provisioning.png"
    echo ">>> copied 4 auto-plotted PNGs into $PAPER_FIGS (policy_grid.png still manual)" >&2
  else
    echo ">>> COPY_TO_PAPER=1 but $PAPER_FIGS not found; skipped copy" >&2
  fi
fi
