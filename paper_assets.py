"""
Write outputs/paper_assets.md: frozen split summary, latest evaluation run IDs,
paths to figures, and a paper-style summary table (from results_summary.csv notes).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_baseline_from_notes(notes: str) -> str:
    m = re.search(r"baseline=([^, ]+)", notes or "")
    return m.group(1) if m else ""


def parse_float_from_notes(notes: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}=([-+]?\d*\.?\d+)", notes or "")
    return float(m.group(1)) if m else float("nan")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_f(x: float, nd: int = 4) -> str:
    if x != x:  # NaN
        return ""
    return f"{x:.{nd}f}"


def figure_list(fig_dir: Path) -> list[str]:
    if not fig_dir.is_dir():
        return []
    return sorted(p.name for p in fig_dir.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready asset index (markdown).")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--manifest", type=str, default="data/splits/manifest.json")
    parser.add_argument("--last-postprocess", type=str, default="outputs/last_postprocess.json")
    parser.add_argument("--results-summary", type=str, default="outputs/results_summary.csv")
    parser.add_argument("--bootstrap", type=str, default="outputs/bootstrap_external.csv")
    parser.add_argument("--threshold-csv", type=str, default="outputs/threshold_sensitivity_external.csv")
    parser.add_argument("--out", type=str, default="outputs/paper_assets.md")
    parser.add_argument(
        "--include-run-ids",
        type=str,
        default="",
        help="comma-separated run_id values to always include in the summary table (merged with last_postprocess.json paper_run_ids)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = root / args.manifest
    post_path = root / args.last_postprocess
    summary_path = root / args.results_summary
    boot_path = root / args.bootstrap
    thr_path = root / args.threshold_csv
    out_path = root / args.out

    lines: list[str] = []
    lines.append("# Paper assets (auto-generated)")
    lines.append("")
    lines.append("Use external-test metrics as headline numbers; cite run IDs and this file path in the manuscript.")
    lines.append("")

    if manifest_path.is_file():
        man = load_json(manifest_path)
        lines.append("## Frozen split (manifest)")
        lines.append("")
        lines.append(f"- Source: `{man.get('source', '')}`")
        lines.append(f"- Split seed: `{man.get('seed', '')}`")
        lines.append(f"- Ratios: {man.get('ratios', {})}")
        lines.append(f"- Counts: {man.get('counts', {})}")
        lines.append("")

    run_ids: dict[str, str] = {}
    fig_rel = ""
    batch_stamp = ""
    meta: dict = {}
    paper_run_ids_meta: list[str] = []
    if post_path.is_file():
        meta = load_json(post_path)
        run_ids = meta.get("run_ids") or {}
        batch_stamp = str(meta.get("timestamp", ""))
        fig_rel = str(meta.get("figures_dir", "")).replace("\\", "/")
        paper_run_ids_meta = [str(x).strip() for x in (meta.get("paper_run_ids") or []) if str(x).strip()]
        lines.append("## Latest postprocess batch")
        lines.append("")
        lines.append(f"- Timestamp: `{batch_stamp}`")
        lines.append(f"- Split regeneration seed (postprocess): `{meta.get('split_seed', '')}`")
        lines.append(f"- Binary threshold (evaluation): `{meta.get('threshold', '')}`")
        lines.append("- Run IDs:")
        for k, v in run_ids.items():
            lines.append(f"  - `{k}`: `{v}`")
        if paper_run_ids_meta:
            lines.append("- Extra paper table run IDs (`paper_run_ids`):")
            for rid in paper_run_ids_meta:
                lines.append(f"  - `{rid}`")
        lines.append("")

    if fig_rel:
        fig_dir = root / fig_rel
        figs = figure_list(fig_dir)
        lines.append("## Figures (bind to this folder in the paper)")
        lines.append("")
        lines.append(f"- Directory: `{fig_rel}/`")
        for i, name in enumerate(figs, start=1):
            lines.append(f"- **Figure {i}** — `{fig_rel}/{name}`")
        lines.append("")
        lines.append("Suggested manuscript labels:")
        labels = [
            ("rating_distribution.png", "MQR rating counts (full reference population)."),
            ("achieved_rate_by_rating.png", "Observed positive rate by rating."),
            ("external_score_hist.png", "External test score distribution by label."),
            ("external_roc.png", "External test ROC."),
            ("external_calibration.png", "External test calibration (reliability)."),
            ("baseline_top10_lift.png", "External test top-decile lift vs baselines."),
        ]
        for fn, cap in labels:
            if fn in figs:
                lines.append(f"- `{fn}`: {cap}")
        lines.append("")

    cli_extra = [x.strip() for x in args.include_run_ids.split(",") if x.strip()]
    priority_ids: list[str] = []
    for x in paper_run_ids_meta + cli_extra:
        if x not in priority_ids:
            priority_ids.append(x)

    want = set(run_ids.values()) | set(priority_ids)
    if summary_path.is_file() and want:
        rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
        picked = [r for r in rows if r.get("run_id") in want]
        order = ["full", "random", "naive"]

        def sort_key(r: dict) -> tuple[int, int, str]:
            rid = r.get("run_id", "")
            if rid in priority_ids:
                return (0, priority_ids.index(rid), "")
            notes = r.get("notes", "")
            b = parse_baseline_from_notes(notes)
            try:
                i = order.index(b)
            except ValueError:
                i = len(order)
            return (1, i, rid)

        picked.sort(key=sort_key)
        lines.append("## Summary table (external test)")
        lines.append("")
        lines.append("| Run ID | Baseline | N eval | AUC | Top-decile k | Top-decile hit | Pos rate | Top-decile lift |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in picked:
            notes = r.get("notes", "")
            b = parse_baseline_from_notes(notes) or r.get("model_config", "")
            lift = parse_float_from_notes(notes, "top10_lift")
            hit = parse_float_from_notes(notes, "top10_hit")
            pr = parse_float_from_notes(notes, "pos_rate")
            k = notes.split("top10_k=")
            top_k = ""
            if len(k) > 1:
                top_k = k[1].split(",")[0].strip()
            lines.append(
                f"| `{r.get('run_id','')}` | {b} | {r.get('n_total','')} | {r.get('auc','')} | "
                f"{top_k} | {fmt_f(hit)} | {fmt_f(pr)} | {fmt_f(lift)} |"
            )
        lines.append("")

    if boot_path.is_file():
        lines.append("## Bootstrap intervals")
        lines.append("")
        lines.append(f"- See `{args.bootstrap}` for percentile intervals on external AUC and top-decile lift (when enabled).")
        lines.append("")

    if thr_path.is_file():
        lines.append("## Threshold sensitivity")
        lines.append("")
        lines.append(
            f"- See `{args.threshold_csv}` for fixed cutoffs on `mqr_score` and the validation Youden row (`val_youden_then_external`)."
        )
        lines.append("")

    lines.append("## Reproducibility snippet")
    lines.append("")
    lines.append("```text")
    lines.append(f"python postprocess_report.py --split-seed <manifest_seed> --skip-split   # or omit skip to regenerate splits")
    lines.append("```")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_path.as_posix())


if __name__ == "__main__":
    main()
