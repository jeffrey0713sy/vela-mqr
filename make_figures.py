"""
Generate PPT-ready figures for the current Vela MQR reference population.

The script is dataset-size agnostic and uses:
- data/reference_population_master.json
- data/splits/external_test.json
- outputs/results_summary.csv (for baseline top10-lift comparison)

Outputs:
- outputs/figures_latest/*.png  (or custom via --outdir)
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_baseline_from_notes(notes: str) -> str:
    m = re.search(r"baseline=([^, ]+)", notes or "")
    return m.group(1) if m else ""


def parse_float_from_notes(notes: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}=([-+]?\d*\.?\d+)", notes or "")
    return float(m.group(1)) if m else float("nan")


def sigmoid_prob_from_score(score: float) -> float:
    # Must match evaluate_splits.py's mapping.
    logit = -2.0 + 0.04 * float(score)
    logit = max(-500, min(500, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def compute_roc(scores: list[float], labels: list[int]):
    paired = list(zip(scores, labels))
    paired.sort(key=lambda x: x[0], reverse=True)
    thresholds = sorted(set(s for s, _ in paired), reverse=True)
    tpr = []
    fpr = []
    P = sum(labels)
    N = len(labels) - P
    if P == 0 or N == 0:
        return [0.0, 1.0], [0.0, 1.0], 0.5
    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        tpr.append(tp / P)
        fpr.append(fp / N)
    tpr = [0.0] + tpr + [1.0]
    fpr = [0.0] + fpr + [1.0]
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return fpr, tpr, auc


def reliability_diagram(probs: list[float], labels: list[int], n_bins: int = 10):
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    xs, ys, counts = [], [], []
    for b in bins:
        if not b:
            continue
        avg_p = sum(x for x, _ in b) / len(b)
        avg_y = sum(y for _, y in b) / len(b)
        xs.append(avg_p)
        ys.append(avg_y)
        counts.append(len(b))
    return xs, ys, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPT figures for Vela MQR.")
    parser.add_argument("--outdir", type=str, default="outputs/figures_latest")
    parser.add_argument("--master", type=str, default="data/reference_population_master.json")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--results-summary", type=str, default="outputs/results_summary.csv")
    parser.add_argument("--run-label", type=str, default="", help="appended to figure titles for run traceability")
    args = parser.parse_args()

    root = Path(".").resolve()
    title_suffix = f" — {args.run_label.strip()}" if args.run_label.strip() else ""
    master_path = root / args.master
    splits_dir = root / args.splits_dir
    out_dir = root / args.outdir
    ensure_dir(out_dir)

    master = read_json(master_path)
    markets = master.get("markets", [])

    labels = ["L1", "L2", "L3", "L4", "L5"]

    # 1) Rating distribution
    rating_counts = {f"L{i}": 0 for i in range(1, 6)}
    achieved_by_rating = {r: {"n": 0, "pos": 0} for r in labels}
    for m in markets:
        r = str(m.get("mqr_rating", "")).upper()
        if r not in rating_counts:
            continue
        rating_counts[r] += 1
        achieved = bool(m.get("t5_outcome", {}).get("achieved_scale", False))
        achieved_by_rating[r]["n"] += 1
        achieved_by_rating[r]["pos"] += int(achieved)

    counts = [rating_counts[x] for x in labels]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, counts, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"])
    plt.title(f"MQR Rating Distribution (N={len(markets)}){title_suffix}")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(c), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_dir / "rating_distribution.png", dpi=200)
    plt.close()

    # 2) Achieved rate by rating
    achieved_rates = []
    for r in labels:
        n = achieved_by_rating[r]["n"]
        pos = achieved_by_rating[r]["pos"]
        achieved_rates.append(pos / n if n else 0.0)

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, achieved_rates, color="#2ca02c")
    plt.title(f"T+5 Achieved-Scale Rate by MQR Rating{title_suffix}")
    plt.xlabel("Rating")
    plt.ylabel("Achieved rate")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    for i, v in enumerate(achieved_rates):
        plt.text(i, v + 0.02, f"{v:.2%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "achieved_rate_by_rating.png", dpi=200)
    plt.close()

    # 3) External test ROC & calibration + score histogram (full pipeline score)
    external_ids = read_json(splits_dir / "external_test.json")
    id_to_market = {}
    for m in markets:
        mid = str(m.get("id", "")).strip()
        if mid:
            id_to_market[mid] = m

    ext_markets = [id_to_market[k] for k in external_ids if k in id_to_market]
    ext_markets = [
        m
        for m in ext_markets
        if m.get("mqr_score") is not None and m.get("t5_outcome", {}).get("achieved_scale") is not None
    ]

    ext_scores = [float(m["mqr_score"]) for m in ext_markets]
    ext_labels = [1 if m.get("t5_outcome", {}).get("achieved_scale") else 0 for m in ext_markets]
    ext_probs = [sigmoid_prob_from_score(s) for s in ext_scores]

    pos_scores = [s for s, y in zip(ext_scores, ext_labels) if y == 1]
    neg_scores = [s for s, y in zip(ext_scores, ext_labels) if y == 0]

    plt.figure(figsize=(8, 4.5))
    plt.hist(neg_scores, bins=15, alpha=0.6, label="Negative (no achieved scale)", color="#ff7f0e")
    plt.hist(pos_scores, bins=15, alpha=0.6, label="Positive (achieved scale)", color="#2ca02c")
    plt.title(f"External Test: Score Distribution by Label (N={len(ext_scores)}){title_suffix}")
    plt.xlabel("Composite mqr_score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "external_score_hist.png", dpi=200)
    plt.close()

    fpr, tpr, auc = compute_roc(ext_scores, ext_labels)
    plt.figure(figsize=(6.8, 5))
    plt.plot(fpr, tpr, lw=2, label=f"Full pipeline (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
    plt.title(f"External Test ROC Curve{title_suffix}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "external_roc.png", dpi=200)
    plt.close()

    xs, ys, counts_bin = reliability_diagram(ext_probs, ext_labels, n_bins=10)
    plt.figure(figsize=(6.8, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="Perfect calibration")
    plt.scatter(xs, ys, s=[max(20, c * 25) for c in counts_bin], color="#1f77b4")
    plt.title(f"External Test Calibration (Reliability Diagram){title_suffix}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive rate")
    plt.ylim(0, 1.05)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "external_calibration.png", dpi=200)
    plt.close()

    # 4) Baseline comparison bar chart (top-decile lift) from results_summary
    results_path = root / args.results_summary
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    ext_size = len(external_ids)

    candidates = []
    for r in rows:
        if r.get("n_external_test") != str(ext_size):
            continue
        baseline = parse_baseline_from_notes(r.get("notes", ""))
        if baseline not in ("full", "random", "naive"):
            continue
        lift = parse_float_from_notes(r.get("notes", ""), "top10_lift")
        auc_val = float(r.get("auc", "nan"))
        run_id = r.get("run_id", "")
        candidates.append((baseline, lift, auc_val, run_id))

    # Take the last occurrence per baseline (most recent run appended last).
    latest = {}
    for b, lift, auc_val, run_id in candidates:
        latest[b] = (lift, auc_val, run_id)

    order = ["full", "random", "naive"]
    if all(b in latest for b in order):
        plt.figure(figsize=(7.2, 4.5))
        colors = {"full": "#1f77b4", "random": "#ff7f0e", "naive": "#2ca02c"}
        lifts = [latest[b][0] for b in order]
        plt.bar(order, lifts, color=[colors[b] for b in order])
        plt.title(f"External Test: Top-10% Lift vs Positive Rate (n={ext_size}){title_suffix}")
        plt.xlabel("Baseline")
        plt.ylabel("Lift (Top10 hit / positive rate)")
        plt.grid(axis="y", alpha=0.25)
        for i, v in enumerate(lifts):
            plt.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "baseline_top10_lift.png", dpi=200)
        plt.close()

    print("Figures generated in:", out_dir.as_posix())


if __name__ == "__main__":
    main()

