# Paper assets (auto-generated)

Use external-test metrics as headline numbers; cite run IDs and this file path in the manuscript.

## Frozen split (manifest)

- Source: `data\reference_population_master.json`
- Split seed: `20260325`
- Ratios: {'train': 0.7, 'validation': 0.15, 'external_test': 0.15}
- Counts: {'total': 500, 'train': 350, 'validation': 75, 'external_test': 75}

## Latest postprocess batch

- Timestamp: `20260331_120841`
- Split regeneration seed (postprocess): `20260325`
- Binary threshold (evaluation): `60.0`
- Run IDs:
  - `full`: `auto_full_500_20260331_120841`
  - `random`: `auto_random_500_20260331_120841`
  - `naive`: `auto_naive_500_20260331_120841`
- Extra paper table run IDs (`paper_run_ids`):
  - `my_run`

## Figures (bind to this folder in the paper)

- Directory: `outputs/figures_latest_20260331_120841/`
- **Figure 1** — `outputs/figures_latest_20260331_120841/achieved_rate_by_rating.png`
- **Figure 2** — `outputs/figures_latest_20260331_120841/baseline_top10_lift.png`
- **Figure 3** — `outputs/figures_latest_20260331_120841/external_calibration.png`
- **Figure 4** — `outputs/figures_latest_20260331_120841/external_roc.png`
- **Figure 5** — `outputs/figures_latest_20260331_120841/external_score_hist.png`
- **Figure 6** — `outputs/figures_latest_20260331_120841/rating_distribution.png`

Suggested manuscript labels:
- `rating_distribution.png`: MQR rating counts (full reference population).
- `achieved_rate_by_rating.png`: Observed positive rate by rating.
- `external_score_hist.png`: External test score distribution by label.
- `external_roc.png`: External test ROC.
- `external_calibration.png`: External test calibration (reliability).
- `baseline_top10_lift.png`: External test top-decile lift vs baselines.

## Summary table (external test)

| Run ID | Baseline | N eval | AUC | Top-decile k | Top-decile hit | Pos rate | Top-decile lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `my_run` | full | 75 | 0.7244 | 8 | 1.0000 | 0.7200 | 1.3889 |
| `auto_full_500_20260331_120841` | full | 75 | 0.7244 | 8 | 1.0000 | 0.7200 | 1.3889 |
| `auto_random_500_20260331_120841` | random | 75 | 0.4709 | 8 | 0.7500 | 0.7200 | 1.0417 |
| `auto_naive_500_20260331_120841` | naive | 75 | 0.5000 | 8 | 0.7500 | 0.7200 | 1.0417 |

## Bootstrap intervals

- See `outputs/bootstrap_external.csv` for percentile intervals on external AUC and top-decile lift (when enabled).

## Threshold sensitivity

- See `outputs/threshold_sensitivity_external.csv` for fixed cutoffs on `mqr_score` and the validation Youden row (`val_youden_then_external`).

## Reproducibility snippet

```text
python postprocess_report.py --split-seed <manifest_seed> --skip-split   # or omit skip to regenerate splits
```
