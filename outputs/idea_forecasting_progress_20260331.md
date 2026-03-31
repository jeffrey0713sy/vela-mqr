# Idea Forecasting — Progress Update (for supervisor)

**Date:** 2026-03-31  
**Track:** Vela Summer 2025 — Idea Forecasting (Crunchbase historical labels)

## Setup and protocol

- Dataset: `idea_training.xlsx` exported to `data/idea_training/idea_training.csv`
- Rows: 35,823 companies (US, founded 2010-2017, funding >= $100k)
- Label: `is_outlier` (>= $250M proxy per provided SQL definition)
- Temporal split: **train = founded year <= 2015**, **test = 2016-2017**
- Class balance:
  - Train positive rate: **3.8204%**
  - Test positive rate: **3.5354%**

## Model results on temporal holdout (2016-2017)

| Model | AUC | PR-AUC | Top-decile precision (k=863) | Top-decile lift | Lift 95% CI |
|------|:---:|:---:|:---:|:---:|:---:|
| Lexicon baseline | 0.7948 | 0.1353 | 0.1333 | 3.769x | [3.310, 4.265] |
| TF-IDF + Logistic Regression | **0.8153** | **0.1899** | **0.1599** | **4.523x** | **[4.005, 5.101]** |

Additional ranking quality (TF-IDF + LR):
- Precision@10 = **0.60**
- Lift@10 = **16.97x**
- Precision@100 = **0.33**
- Lift@100 = **9.33x**

## Interpretation

- Strong signal exists in inception-stage text + founding year under strict temporal split.
- TF-IDF + LR consistently outperforms the lexicon baseline across AUC, PR-AUC, and lift.
- Top-decile lift CI for TF-IDF + LR remains well above 1.0, supporting above-chance ranking on future years.

## Reproducibility

```bash
python scripts/import_idea_training.py --input "C:/Users/91069/Downloads/idea_training.xlsx"
python idea_forecast_eval.py --scorer lexicon --bootstrap-b 300 --out-dir outputs/idea_forecast_lexicon
python idea_forecast_eval.py --scorer tfidf_lr --bootstrap-b 200 --out-dir outputs/idea_forecast_tfidf
```

## Next actions

1. Add Vela Search lookback context (`search_context`) for a subset, then full run.
2. Evaluate LLM-based scorer on a cost-controlled slice and compare to TF-IDF + LR.
3. Build explainability appendix (top features / representative true positives & false positives).
