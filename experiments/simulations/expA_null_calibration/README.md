# Experiment A: Null Calibration

This simulation checks whether BioRSP permutation p-values are calibrated under a strict null with donor structure.

Calibration here means:

- p-values are approximately Uniform(0,1) for non-underpowered genes
- empirical Type I error at alpha=0.05 is near 0.05
- left-tail rates (`p<=0.01`, `p<=0.005`, min-p mass) are near nominal with CIs

## BH feasibility at current settings

With plus-one correction and `n_perm=300`, the minimum attainable p-value is:

- `min_attainable_p = 1/(n_perm+1) = 1/301 ~= 0.003322`

For full-run BH over `m=2000` genes:

- `q=0.05 -> q/m = 2.5e-5`
- `q=0.10 -> q/m = 5e-5`

Both are far below `0.003322`, so full BH is mathematically infeasible at these settings.
Zero BH rejections in that regime are structurally forced and not treated as evidence of FDR control.

Experiment A therefore reports full-BH feasibility explicitly and suppresses infeasible full-BH
Type I metrics.

## Multiple-testing validation modes

- `panel_bh` (default): deterministic random panel BH on a testable panel size.
  - defaults: `bh_panel_size=15`, `bh_panel_strategy=prevalence_stratified_fixed`
  - writes `results/bh_panel_validation.csv`
- `uniformity_only`: suppress panel BH and report uniformity + tail + fixed-alpha Type I only.

## Why D_eff gating is used

Underpowered status is based on donor-effective support (D_eff), not an "all donors must pass" rule.

For each gene:

- compute per-donor foreground/background counts
- mark a donor informative if `n_fg >= min_fg_per_donor` and `n_bg >= min_bg_per_donor`
- `D_eff` is the number of informative donors

A gene is underpowered if any of:

- `prev < p_min`
- `n_fg_total < min_fg_total`
- `D_eff < d_eff_min`
- `n_perm < min_perm`

## What permutations condition on

Permutation testing is gene-conditional:

- use the realized foreground mask `f` for that gene
- shuffle `f` within each donor (donor-stratified) when possible
- this preserves each donor's foreground count exactly
- compute `T = max(abs(E_phi))` per permutation
- use plus-one p-value correction:
  - `p = (1 + sum(T_perm >= T_obs)) / (1 + n_perm)`

No null-distribution reuse is done across genes.

## Commands

Full run:

```bash
python experiments/simulations/expA_null_calibration/run_expA_null_calibration.py \
  --outdir experiments/simulations/expA_null_calibration
```

Full run with explicit panel-BH settings:

```bash
python experiments/simulations/expA_null_calibration/run_expA_null_calibration.py \
  --outdir experiments/simulations/expA_null_calibration \
  --bh_validation_mode panel_bh \
  --bh_panel_size 15 \
  --bh_panel_strategy prevalence_stratified_fixed
```

Uniformity-only reporting mode:

```bash
python experiments/simulations/expA_null_calibration/run_expA_null_calibration.py \
  --outdir experiments/simulations/expA_null_calibration \
  --bh_validation_mode uniformity_only
```

Test mode (tiny CI/local sanity check):

```bash
python experiments/simulations/expA_null_calibration/run_expA_null_calibration.py \
  --test_mode --outdir experiments/simulations/expA_null_calibration
```

`test_mode` writes to `<outdir>/test_mode/` and is intentionally noisy due to small `n_perm` and gene counts.

Each run also writes `REPORT.md` with calibration tables, plot embeds, and critical interpretation.

## Recompute evaluation artifacts from existing metrics

If `results/metrics_long.csv` already exists, regenerate summaries/plots/report without rerunning simulation:

```bash
python experiments/simulations/expA_null_calibration/evaluate_expA_null_calibration.py \
  --outdir experiments/simulations/expA_null_calibration
```
