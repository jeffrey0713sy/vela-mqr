# Dataset Splits (Frozen Protocol)

This folder stores frozen dataset splits for reproducible evaluation.

## Version

- Split version: `v1`
- Random seed: `20260324`
- Freeze date: `2026-03-24`

## Policy

- `train.json`: 70%
- `validation.json`: 15%
- `external_test.json`: 15%

The split must be generated once and reused across all model comparisons.

## Record Format

Each file contains a JSON array of market IDs:

```json
[
  "market_001",
  "market_002"
]
```

If IDs are unavailable, use a stable unique key such as `market_name + ref_year`.

## Rules

1. Do not modify split membership after viewing evaluation outcomes.
2. Report train/validation/external metrics separately.
3. Use the same split for baseline and full pipeline.
