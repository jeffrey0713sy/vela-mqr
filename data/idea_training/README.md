# Idea Forecasting dataset (Vela Summer 2025)

Place the canonical spreadsheet here after export, or run the importer.

## Files (local, not committed)

- `idea_training.csv` — produced by `scripts/import_idea_training.py` from `idea_training.xlsx`
- `idea_training_enriched.csv` — optional; adds `search_context` via `scripts/idea_enrich_vela.py`

Git ignores `*.csv` in this folder to avoid committing Crunchbase-derived rows; keep your xlsx outside the repo or symlink.

## Reproduce

```bash
pip install -r requirements-idea.txt
python scripts/import_idea_training.py --input "path/to/idea_training.xlsx"
python idea_forecast_eval.py --scorer lexicon
python idea_forecast_eval.py --scorer tfidf_lr
python run_idea_pipeline.py --input "path/to/idea_training.xlsx"
```

Temporal split (fixed): **train** = founded year ≤ 2015, **test** = 2016–2017. Labels: `is_outlier` (≥$250M proxy per Vela SQL). Do not use `total_funding_usd` as a feature for inception forecasting (leakage).

## Environment (optional)

- `VELA_SEARCH_API_KEY`, `VELA_SEARCH_BASE_URL` — see `vela_search_context.py`
- `OPENAI_API_KEY` — for `--scorer llm` (expensive; use `--llm-max-rows`)
