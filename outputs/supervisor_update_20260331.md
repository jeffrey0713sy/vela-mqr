# Vela MQR — Supervisor Update
**Date:** 2026-03-31 · **Run batch:** `postprocess 20260331_120841`

---

## 1. What Was Done

Three pending evaluation tasks have been completed this session:

| Task | Status |
|------|--------|
| Bootstrap CIs for external AUC and top-decile lift | Done (B=1000) |
| Threshold sensitivity across 13 score cutoffs + Youden rule | Done |
| Paper-ready figure/table index tied to run IDs | Done |

All results are reproducible from frozen split seed `20260325` and master data as of the commit.

---

## 2. Headline Numbers (External Test, N=75)

| Metric | Full pipeline | Random baseline | Naive baseline |
|--------|:---:|:---:|:---:|
| AUC | **0.7244** | 0.4709 | 0.5000 |
| AUC 95% CI | **[0.580, 0.847]** | [0.327, 0.624] | [0.500, 0.500] |
| Top-decile k | 8 | 8 | 8 |
| Top-decile hit rate | **1.000** | 0.750 | 0.750 |
| Top-decile lift vs positive rate | **1.389x** | 1.042x | 1.042x |
| Top-decile lift 95% CI | **[1.229, 1.630]** | [0.563, 1.442] | [0.563, 1.389] |

> **Key claim:** The full pipeline's top-decile lift CI lower bound (1.229) is strictly above 1.0,
> providing 95th-percentile statistical support that ranking outperforms chance on the external test.

---

## 3. Calibration

| | Before temperature scaling | After (T_opt=3.0) |
|---|:---:|:---:|
| ECE (full pipeline) | 0.2995 | **0.2481** |

Pipeline is over-confident; temperature scaling reduces ECE by ~17%. Calibration is a
supporting diagnostic — not the primary KPI.

---

## 4. Threshold Sensitivity

Evaluated at 13 fixed `mqr_score` cutoffs (40–80) and one validation-Youden rule
(`threshold=28.63, J=0.526`) across all three baselines. See
`outputs/threshold_sensitivity_external.csv`.

Key observation:
- At the **default threshold (60.0)**, full pipeline precision = 1.000, recall = 0.167 — very
  conservative; all positively-labelled predictions are correct but most positives are missed.
- At the **Youden threshold (28.63)** selected on validation, recall increases substantially
  while precision decreases — interpreted as a softer but broader net.
- L5=0 at threshold 60.0 is a **score-range artefact**, not a pipeline failure. The current
  L5 label has no markets scoring ≥60 in the external test set.

---

## 5. Figures Generated

Directory: `outputs/figures_latest_20260331_120841/`

| # | File | Description |
|---|------|-------------|
| 1 | `rating_distribution.png` | MQR rating counts across full 500-market population |
| 2 | `achieved_rate_by_rating.png` | Observed T+5 positive rate by rating |
| 3 | `external_score_hist.png` | Score distribution by label on external test |
| 4 | `external_roc.png` | ROC curve (full pipeline, AUC=0.7244) |
| 5 | `external_calibration.png` | Reliability diagram |
| 6 | `baseline_top10_lift.png` | Top-decile lift comparison: full vs random vs naive |

---

## 6. Limitations (Explicit)

- **Label distribution:** 69% positive rate in sample is higher than expected real-world outcomes. Results should be interpreted as conditional on this distribution.
- **Generated labels:** T+5 outcomes are model-generated, not independently verified portfolio records. External validity gap remains the primary open risk.
- **L5 precision=0:** A threshold-range artefact at the current 60.0 cutoff; not a failure of the L5 rating concept.
- **External test N=75:** Bootstrap CIs are wide; a larger test set would narrow them.

---

## 7. Remaining Open Action

| Action | Status |
|--------|--------|
| Plan external validation against independently verified market outcomes | **Pending** — requires data access agreement |

---

## 8. Reproducibility

```bash
# Reproduce this entire batch from scratch (uses frozen split):
python postprocess_report.py --skip-split --paper-run-ids my_run

# Key artifacts:
#   outputs/bootstrap_external.csv   — bootstrap CIs
#   outputs/threshold_sensitivity_external.csv  — threshold sweep
#   outputs/paper_assets.md          — run-ID-bound table + figure index
#   outputs/last_postprocess.json    — full batch metadata
```

Split: seed=`20260325`, 500 markets, 350/75/75 train/val/test.
