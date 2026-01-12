# Calibration Benchmark Report

**Generated:** 2026-01-12 18:38:06

**Directory:** `/Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/calibration`

## Parameters

- **outdir:** /Users/cytronicoder/Documents/GitHub/biorsp-swordfish/analysis/outputs/calibration
- **seed:** 42
- **n_reps:** 100
- **N:** [1000]
- **shape:** ['disk']
- **null_type:** ['iid']
- **n_permutations:** 100
- **mode:** quick
- **n_workers:** 2
- **checkpoint_every:** 25
- **resume:** False
- **permutation_scope:** all

## Summary Statistics

| benchmark   | shape   |    N | null_type   | permutation_scheme   |   n_reps |   fpr_05 |   fpr_05_ci_low |   fpr_05_ci_high |   fpr_01 |   fpr_01_ci_low |   fpr_01_ci_high |   ks_stat |   ks_pval |   abstain_rate | calibrated   |
|:------------|:--------|-----:|:------------|:---------------------|---------:|---------:|----------------:|-----------------:|---------:|----------------:|-----------------:|----------:|----------:|---------------:|:-------------|
| calibration | disk    | 1000 | iid         | global               |      100 |     0.03 |       0.0102543 |        0.0845208 |     0.03 |       0.0102543 |        0.0845208 | 0.0834653 |    0.4641 |              0 | True         |

## Interpretation

### What does this mean?

A well-calibrated method should reject 5.0% of null hypotheses at α=0.05. We test this across different spatial shapes, distortions, and null models.

✅ **Good calibration:** Average FPR = 0.030 (close to nominal 0.050)

📊 **At α=0.01:** Mean FPR = 0.030

✅ **KS test:** P-values appear uniformly distributed (mean KS p = 0.464)

### Recommended actions

- Method is well-calibrated for null hypothesis testing.

---

*This report was generated automatically by the BioRSP simulation framework.*