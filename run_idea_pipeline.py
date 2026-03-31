"""
One-shot: import xlsx -> CSV (optional) -> temporal idea_forecast_eval.

Examples:
  python run_idea_pipeline.py --input "%USERPROFILE%/Downloads/idea_training.xlsx"
  python run_idea_pipeline.py --skip-import --scorer tfidf_lr --bootstrap-b 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import + evaluate idea forecasting CSV")
    parser.add_argument("--input", type=str, default="", help="Path to idea_training.xlsx")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--csv", type=str, default="data/idea_training/idea_training.csv")
    parser.add_argument("--scorer", type=str, default="lexicon")
    parser.add_argument("--bootstrap-b", type=int, default=0)
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    if not args.skip_import:
        if not args.input.strip():
            raise SystemExit("Provide --input path/to/idea_training.xlsx or use --skip-import")
        run(
            [
                args.python,
                str(ROOT / "scripts" / "import_idea_training.py"),
                "--input",
                args.input.strip(),
                "--output",
                args.csv,
            ]
        )

    cmd = [
        args.python,
        str(ROOT / "idea_forecast_eval.py"),
        "--csv",
        args.csv,
        "--scorer",
        args.scorer,
    ]
    if args.bootstrap_b > 0:
        cmd += ["--bootstrap-b", str(args.bootstrap_b)]
    run(cmd)


if __name__ == "__main__":
    main()
