"""
Post-run automation:
1) regenerate frozen splits (train/validation/external_test)
2) evaluate full/random/naive baselines on external_test
3) regenerate PPT-ready figures for the current dataset size

This is intended to be triggered after run_scale_pipeline completes.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("[postprocess] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto postprocess report for Vela MQR.")
    parser.add_argument("--split-seed", type=int, default=20260325)
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--report-model-config", type=str, default="full_pipeline_v2.1")
    parser.add_argument("--skip-split", action="store_true", help="Do not regenerate split files.")
    parser.add_argument("--bootstrap-b", type=int, default=1000, help="0 disables bootstrap in evaluate_splits.")
    parser.add_argument("--bootstrap-seed", type=int, default=20260331)
    parser.add_argument(
        "--paper-run-ids",
        type=str,
        default="",
        help="Comma-separated run_id values to keep in paper_assets.md (merged with prior last_postprocess.json).",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    python = sys.executable

    # regenerate splits unless requested otherwise
    if not args.skip_split:
        run_cmd([python, str(root / "split_dataset.py"), "--seed", str(args.split_seed)])

    n_total = None
    try:
        import json

        master = json.loads((root / "data/reference_population_master.json").read_text(encoding="utf-8"))
        n_total = len(master.get("markets", []))
    except Exception:
        n_total = "unknown"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_full_id = f"auto_full_{n_total}_{stamp}"
    run_rand_id = f"auto_random_{n_total}_{stamp}"
    run_naive_id = f"auto_naive_{n_total}_{stamp}"

    boot_args: list[str] = []
    if args.bootstrap_b > 0:
        boot_args = [
            "--bootstrap-b",
            str(args.bootstrap_b),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
        ]

    # evaluate baselines (no external API calls; pure compute)
    run_cmd(
        [
            python,
            str(root / "evaluate_splits.py"),
            "--model-config",
            "full_pipeline_v2.1",
            "--baseline",
            "full",
            "--threshold",
            str(args.threshold),
            "--run-id",
            run_full_id,
            *boot_args,
        ]
    )
    run_cmd(
        [
            python,
            str(root / "evaluate_splits.py"),
            "--model-config",
            "random_baseline_v1",
            "--baseline",
            "random",
            "--threshold",
            str(args.threshold),
            "--run-id",
            run_rand_id,
            *boot_args,
        ]
    )
    run_cmd(
        [
            python,
            str(root / "evaluate_splits.py"),
            "--model-config",
            "naive_baseline_v1",
            "--baseline",
            "naive",
            "--threshold",
            str(args.threshold),
            "--run-id",
            run_naive_id,
            *boot_args,
        ]
    )

    run_cmd([python, str(root / "threshold_sensitivity.py")])

    # generate figures
    figures_out = root / "outputs" / f"figures_latest_{stamp}"
    run_cmd(
        [
            python,
            str(root / "make_figures.py"),
            "--outdir",
            str(figures_out.relative_to(root)),
            "--run-label",
            f"postprocess {stamp}",
        ]
    )

    paper_persist: list[str] = []
    last_pp = root / "outputs" / "last_postprocess.json"
    if last_pp.is_file():
        try:
            prev = json.loads(last_pp.read_text(encoding="utf-8"))
            for x in prev.get("paper_run_ids") or []:
                xs = str(x).strip()
                if xs and xs not in paper_persist:
                    paper_persist.append(xs)
        except Exception:
            pass
    for x in args.paper_run_ids.split(","):
        xs = x.strip()
        if xs and xs not in paper_persist:
            paper_persist.append(xs)

    meta = {
        "timestamp": stamp,
        "split_seed": args.split_seed,
        "run_ids": {"full": run_full_id, "random": run_rand_id, "naive": run_naive_id},
        "figures_dir": str(figures_out.relative_to(root)).replace("\\", "/"),
        "threshold": args.threshold,
        "bootstrap_B": args.bootstrap_b,
        "bootstrap_seed": args.bootstrap_seed,
        "paper_run_ids": paper_persist,
    }
    (root / "outputs" / "last_postprocess.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    run_cmd([python, str(root / "paper_assets.py"), "--root", str(root)])

    print("[postprocess] Done. Figures at:", figures_out.as_posix())


if __name__ == "__main__":
    main()

