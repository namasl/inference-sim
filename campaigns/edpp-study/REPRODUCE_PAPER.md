# Reproducing the INFOCOM paper figures and tables (simulator only)

This is the one-page recipe for regenerating every **simulation-based** figure and table in the
joint PD-routing paper. It excludes the two real-vLLM calibration figures (`titer_modeb.png` teacher-
forced iteration-law validation, and `latency_fit.png`), which need a live GPU server and are covered
separately in `scripts/calibration/`.

Everything here is deterministic and CPU-only. Run every command **from the repo root**
(`inference-sim/`), on branch `feat/edpp-estimator-validation`.

```bash
go build -o blis main.go          # the repro scripts also do this if ./blis is absent
```

## One-shot driver

```bash
bash campaigns/edpp-study/reproduce_paper.sh
```

Runs all five simulation campaigns in sequence, produces the four auto-plotted PNGs under
`campaigns/edpp-study/out/…`, and prints the goodput grid you need for the fifth figure and the two
tables. Override seeds for a faster smoke run: `SEEDS="42" bash campaigns/edpp-study/reproduce_paper.sh`
(fewer seeds means the numbers move slightly off the paper's three- and ten-seed means).

If you also have the paper tree checked out beside the repo (`../infocom/figures/`), pass
`COPY_TO_PAPER=1` to copy the four auto-plotted PNGs into it. The colleague reproducing results does
not need this.

## Figure-by-figure

Each row: the command, the file it writes, the paper figure it becomes, and the number to sanity-check
against. Authoritative expected numbers live in `FINDINGS.md` and `STUDY_REPORT.md`; the paper's own
captions and `\cref{tab:regret}` carry the headline values repeated below.

### Fig. `ttft_d_admission.png` — collocated TTFT estimator vs realized

```bash
bash campaigns/edpp-study/repro_ttft_d_local.sh
```
- Writes `campaigns/edpp-study/out/ttft_d_local/fig_admission.png` (also `fig_prefill.png`, `fig_ttft.png`).
- Paper figure: `ttft_d_admission.png` (rename/copy of `fig_admission.png`).
- Check: rollforward and fluid estimators track realized to ~1.2× up to capacity, then under-predict by
  one to two orders of magnitude in overload (the snapshot-blind blow-up). See `FINDINGS.md` Stage C.

### Fig. `ttft_p_disagg.png` — disaggregated prefill-path TTFT

```bash
bash campaigns/edpp-study/repro_ttft_p_local.sh
```
- Writes `campaigns/edpp-study/out/ttft_p_local/fig_ttft_p.png` (also `fig_prefill_adm.png`).
- Paper figure: `ttft_p_disagg.png` (rename/copy of `fig_ttft_p.png`).
- Check: the estimate stays within a small bounded factor across the whole load range (median ~2.08×
  under-prediction on the batch archetype, p50 63ms vs 120ms). See `FINDINGS.md` "ttft_p".

### Fig. `hetero_ratio_sweep.png` — decode-speed heterogeneity sweep

```bash
bash campaigns/edpp-study/repro_hetero_ratio_sweep_gp.sh
python3 campaigns/edpp-study/analyze/hetero_ratio_sweep.py \
  --csv campaigns/edpp-study/out/hetero_ratio/ratio_sweep.csv \
  --out campaigns/edpp-study/out/hetero_ratio/hetero_ratio_sweep.png
```
- The `_gp` script writes only the CSV; the analyze script draws the PNG.
- Paper figure: `hetero_ratio_sweep.png`.
- Check: our rule and drift-plus-penalty stay on the best-static-split optimum across the ratio range;
  our worst-case regret against that split is ~0.019, while least-TTFT and Kairos break early.

### Fig. `pd_provisioning.png` — topology / provisioning matrix

```bash
bash campaigns/edpp-study/repro_topology_matrix_gp.sh
python3 campaigns/edpp-study/analyze/topology_matrix.py \
  --csv campaigns/edpp-study/out/topo_matrix/topo_matrix.csv \
  --out campaigns/edpp-study/out/topo_matrix/pd_provisioning.png
```
- The `_gp` script writes only the CSV; the analyze script draws the PNG.
- Paper figure: `pd_provisioning.png`.
- Check: our rule is the only policy whose worst-case regret stays small on every shape
  (1P3D 0.042 / 2P2D 0.007 / 3P1D 0.004). Others exceed 0.23 on at least one shape.

### Fig. `policy_grid.png` + Tab. `tab:grid` + Tab. `tab:regret` — the goodput matrix

```bash
bash campaigns/edpp-study/repro_var_dominance_goodput.sh
```
- Writes per-arm goodput into `campaigns/edpp-study/out/var_dom_gp/` and prints the full grid and the
  worst-case regret per policy to stderr.
- This one campaign feeds three paper artifacts, and its numbers are **transcribed by hand**, not
  auto-plotted:
  - `policy_grid.png` — the printed matrix is copied into the `data = {…}` literal at the top of
    `../infocom/figures/plot_policy_grid.py`, then `cd ../infocom/figures && python3 plot_policy_grid.py`
    writes `policy_grid.png` in place.
  - `tab:grid` and `tab:regret` — the per-cell goodputs and the worst-case-regret column are typed into
    the paper `.tex`.
- Check (the paper headline): deployable full rule carries worst-case regret **0.034**, the true-length
  variant **0.060**, and every alternative lands between **0.39 and 0.94**. If your run reproduces those
  three numbers, the grid matches.

## Why some steps are manual

The four sweep/estimator figures are fully scripted: run the command, get the PNG. The goodput grid
(`policy_grid.png` and both tables) is not — its numbers are transcribed from the campaign's stdout into
a plotting-script literal and into the LaTeX tables. So "reproduce" for those means run
`repro_var_dominance_goodput.sh`, then confirm the printed grid matches `tab:grid` and the 0.034 / 0.060
regret headline. Regenerating the PNG or tables from a fresh run means re-transcribing.

## Where to read next

- `README.md` — full orientation: what the system is, which source files implement the policy, the trace-CSV column dictionary.
- `FINDINGS.md` — the authoritative log of every checkpoint number a correct run must reproduce.
- `STUDY_REPORT.md` — the long-form narrative with the full dominance grid.
