# Vela MQR - Meeting Update (for Supervisor)

## Current Status

- The end-to-end pipeline is operational and reproducible:
  - Role 0 seed generation (Gemini + search)
  - Role 1 extraction (OpenAI)
  - Role 2 independent verification (Gemini)
  - Role 3 anonymous re-scoring (OpenAI)
  - Composite score + L1-L5 rating
- 500-market reference population has been completed and frozen for evaluation.
- Reproducible split files are generated (`350/75/75`, seed `20260324`).
- Baseline and full-pipeline comparisons are generated automatically via postprocess scripts.

## What Was Improved Since Last Discussion

- Added startup API preflight checks in `run_scale_pipeline.py` to fail fast on invalid/leaked keys.
- Pipeline can now avoid long wasted runs before authentication errors are detected.
- Progress saving and resume behavior remain active for large-batch generation.

## Supervisor Feedback Interpreted as Action Items

1. Increase sample size for statistical significance (target >=500).
2. Explicitly separate training vs validation vs external test sets.
3. Move from process demo to evidence-based performance reporting.
4. Focus on high-value market detection quality (especially L5).
5. Add stronger external validation to support publication claims.

## Evaluation Snapshot (Latest 500-Market Run)

- **Population:** 500 rated markets (overall positive rate: `69.0%`, 345/500)
- **External test set:** 75 markets
- **Full pipeline external AUC:** `0.7244` (random baseline: `0.4709`)
- **Calibration (ECE):** `0.2995 -> 0.2481` after temperature scaling
- **Top-decile lift (primary KPI):** `1.3889x` (random baseline: `1.0417x`)
- **L5 metrics:** currently `0` due to threshold/score-range mismatch (not pipeline failure)

## What Was Improved Since Last Discussion

- Added startup API preflight checks in `run_scale_pipeline.py` to fail fast on invalid/leaked keys.
- Added duplicate-topic pre-check + dynamic topic expansion + 503 retry backoff in generation.
- Added incremental autosave for Phase 1 to avoid progress loss during long runs.
- Added split/evaluation/report scripts:
  - `split_dataset.py`
  - `evaluate_splits.py`
  - `postprocess_report.py`
  - `make_figures.py`
- Added one-command reporting path: `python run_scale_pipeline.py --target 500 --auto-report`

## Risks and Controls (for Paper Framing)

- **Label distribution mismatch to real-world startup success rates:** report as limitation and future normalization direction.
- **Calibration under imbalance:** keep AUC + top-decile lift as main decision metrics; avoid overclaiming absolute probabilities.
- **L5 threshold mismatch:** explicitly document as threshold-design issue, not code/output failure.
- **External validity:** next phase is validation with independently verified market outcomes.

## One-Paragraph Verbal Update

We completed the full 500-market run and switched to a frozen, reproducible evaluation setup. The external test result shows meaningful ranking signal (AUC 0.7244 vs random 0.4709), and our top-decile lift is 1.3889x, which is now the main KPI for identifying high-potential markets. We also hardened the pipeline with API preflight checks, duplicate prevention, retry/backoff, and autosave, so generation is much more reliable. For paper positioning, we clearly state that L5=0 is currently a threshold-range mismatch and that label distribution normalization is a future-work item.
