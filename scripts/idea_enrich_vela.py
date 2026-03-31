"""
Add `search_context` column by calling Vela Search (optional).

Requires API env vars; rows get empty context when offline.

Usage:
  python scripts/idea_enrich_vela.py --input data/idea_training/idea_training.csv --output data/idea_training/idea_training_enriched.csv --max-rows 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vela_search_context import fetch_idea_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich idea CSV with Vela Search context.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    rows = list(csv.DictReader(in_path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("No rows in input")

    fieldnames = list(rows[0].keys())
    if "search_context" not in fieldnames:
        fieldnames.append("search_context")

    lim = args.max_rows if args.max_rows > 0 else len(rows)
    for i, row in enumerate(rows[:lim]):
        name = row.get("name") or row.get("company_name") or ""
        short = row.get("short_description") or ""
        founded = row.get("founded_on") or row.get("founded_at") or ""
        ctx = fetch_idea_context(name=str(name), short_description=str(short), founded_on=founded)
        row["search_context"] = ctx
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{lim} ...")

    # Pass-through remaining rows without API calls
    for row in rows[lim:]:
        row["search_context"] = row.get("search_context", "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
