# Metrics Report

## Run Metadata
- Run ID: auto_naive_500_20260331_120841
- Model config: naive_baseline_v1
- Baseline mode: naive
- Threshold: 60.0

## Dataset Summary
- N total (evaluated on external_test with labels): 75
- N train: 350
- N validation: 75
- N external_test: 75
- Positive rate (external_test eval subset): 0.7200

## Core Metrics
- AUC: 0.5000
- ECE before scaling: 0.0429
- ECE after scaling: 0.0038
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000

## L5 Performance
- L5 precision: 0.0000
- L5 recall: 0.0000
- L5 hit rate: 0.0000
- Random baseline hit rate: 0.10
- L5 lift vs random: 0.0000

## Robust Top-Decile Ranking Metric
- Top-decile K: 8
- Top-decile hit rate: 0.7500
- Observed positive rate: 0.7200
- Top-decile lift vs positive rate: 1.0417

## Bootstrap (external test; 1000 resamples, seed=20260331)
- AUC: 0.5000 (95% percentile interval: 0.5000–0.5000)
- Top-decile lift vs positive rate: 1.0417 (95% interval: 0.5625–1.3889)
- Rows appended to `outputs/bootstrap_external.csv`

## Files Generated
- outputs/results_summary.csv
- outputs/confusion_matrix.csv
