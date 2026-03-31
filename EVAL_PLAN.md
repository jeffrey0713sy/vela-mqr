# Vela MQR - Evaluation Plan and Current Status

## Objective

Establish reproducible and decision-useful evidence that the pipeline can rank higher-potential markets better than chance on a frozen external test set.

## Current Status (Completed)

- 500-market reference population completed.
- Frozen split generated with seed `20260324`.
- Split ratio fixed at `70/15/15`:
  - `train`: 350
  - `validation`: 75
  - `external_test`: 75
- Baseline and full-pipeline evaluations executed on the same split.
- Result artifacts exported and committed for reporting.

## Latest Evaluation Snapshot (500-Market Run, postprocess 20260331)

- Overall positive rate (`achieved_scale=True`): `345/500 = 69.0%`
- External AUC (full pipeline): `0.7244` · 95% CI `[0.580, 0.847]`
- External AUC (random baseline): `0.4709` · 95% CI `[0.327, 0.624]`
- ECE before/after temperature scaling: `0.2995 → 0.2481`
- External top-decile size (`k`): `8`
- Top-decile hit rate (full pipeline): `1.0000`
- External base positive rate: `0.7200`
- Top-decile lift (full pipeline): `1.3889x` · 95% CI `[1.229, 1.630]` ← **CI lower bound > 1.0**
- Top-decile lift (random baseline): `1.0417x` · 95% CI `[0.563, 1.442]`
- L5 precision/recall: `0/0` (threshold-range mismatch; see threshold sensitivity at score 28.63)
- Bootstrap: 1000 resamples, seed `20260331`; see `outputs/bootstrap_external.csv`

## KPI Priority (for Reporting and Paper)

1. Primary KPI: `top-decile lift` on external test.
2. Secondary KPI: external AUC.
3. Calibration (ECE) as supporting diagnostic, not the headline KPI.
4. L5 metrics reported transparently, but interpreted with threshold caveat.

## Artifacts (Now Available)

- `data/splits/train.json`
- `data/splits/validation.json`
- `data/splits/external_test.json`
- `data/splits/manifest.json`
- `outputs/results_summary.csv`
- `outputs/confusion_matrix.csv`
- `outputs/metrics_report.md`
- `outputs/metrics_report_template.md`
- `outputs/bootstrap_external.csv`
- `outputs/threshold_sensitivity_external.csv`
- `outputs/paper_assets.md`
- `outputs/last_postprocess.json`
- `outputs/figures_latest_20260331_120841/` (6 figures)

## Completed Checklist

- [x] Complete 500-market generation run.
- [x] Freeze split policy and persist split files.
- [x] Run baseline + full model evaluation on same split.
- [x] Export summary table and confusion matrix.
- [x] Produce supervisor-facing one-page update.
- [x] Add bootstrap confidence intervals for AUC and top-decile lift on external test.
- [x] Run sensitivity checks for alternative threshold schemes (preserving frozen split).
- [x] Add paper-ready table and figure references directly tied to run IDs.

## Open Risks and How We Frame Them

- **Label distribution mismatch:** current sample has higher success prevalence than expected real-world startup outcomes.
- **L5=0 in current run:** due to threshold/score-range mismatch, not pipeline failure.
- **External validity gap:** outcomes are still generated labels, not yet fully validated against independent real-world portfolio records.

## Next Actions

- [x] Add bootstrap confidence intervals for AUC and top-decile lift on external test.
- [x] Run sensitivity checks for alternative threshold schemes (while preserving frozen split).
- [x] Add paper-ready table and figure references directly tied to run IDs.
- [ ] Plan external validation against independently verified market outcomes.

## Reporting Rules (Keep)

- Do not mix train/validation/external metrics into one headline number.
- Treat external-test metrics as the default headline evidence.
- Keep limitation statements explicit when discussing calibration or L5.
