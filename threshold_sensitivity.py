"""
Threshold sensitivity on frozen splits: fixed score cutoffs on external_test,
plus optional validation-split Youden J threshold applied only to external_test metrics.

Writes: outputs/threshold_sensitivity_external.csv
"""

import argparse
import csv
from pathlib import Path

from evaluate_splits import (
    confusion_binary,
    prf1_from_conf,
    read_json,
    load_split_eval,
    stable_market_key,
)


def youden_j(scores: list[float], labels: list[int], thr: float) -> float:
    preds = [1 if s >= thr else 0 for s in scores]
    P = sum(labels)
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("-inf")
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    tpr = tp / P
    tnr = tn / N
    return tpr + tnr - 1.0


def best_threshold_youden(scores: list[float], labels: list[int]) -> tuple[float, float]:
    cands = sorted(set(scores))
    if not cands:
        return float("nan"), float("-inf")
    best_thr, best_j = cands[0], float("-inf")
    for thr in cands:
        j = youden_j(scores, labels, thr)
        if j > best_j:
            best_j, best_thr = j, thr
    return best_thr, best_j


def row_for_threshold(scores: list[float], labels: list[int], thr: float) -> dict:
    preds = [1 if s >= thr else 0 for s in scores]
    conf = confusion_binary(preds, labels)
    p, r, f1 = prf1_from_conf(conf)
    return {
        "tp": conf["tp"],
        "tn": conf["tn"],
        "fp": conf["fp"],
        "fn": conf["fn"],
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sensitivity (frozen splits).")
    parser.add_argument("--master", type=str, default="data/reference_population_master.json")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="40,45,50,52.5,55,57.5,60,62.5,65,67.5,70,75,80",
        help="comma-separated mqr_score cutoffs for scheme=fixed",
    )
    parser.add_argument("--random-seed", type=int, default=20260324)
    parser.add_argument("--baselines", type=str, default="full,random,naive")
    args = parser.parse_args()

    import random

    master = read_json(Path(args.master))
    by_key = {stable_market_key(m): m for m in master.get("markets", [])}
    splits_dir = Path(args.splits_dir)
    train_ids = read_json(splits_dir / "train.json")
    val_ids = read_json(splits_dir / "validation.json")
    test_ids = read_json(splits_dir / "external_test.json")

    thr_list = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    out_path = Path(args.outputs_dir) / "threshold_sensitivity_external.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "baseline",
        "scheme",
        "threshold",
        "val_youden_j",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
    rows_out: list[list] = []

    val_thr_full: float | None = None
    val_j_full: float | None = None
    rng_probe = random.Random(args.random_seed)
    v_scores, v_labels, _ = load_split_eval(master, val_ids, train_ids, by_key, "full", rng_probe)
    val_thr_full, val_j_full = best_threshold_youden(v_scores, v_labels)

    for bl in baselines:
        rng = random.Random(args.random_seed)
        ext_scores, ext_labels, _ = load_split_eval(master, test_ids, train_ids, by_key, bl, rng)
        for thr in thr_list:
            m = row_for_threshold(ext_scores, ext_labels, thr)
            rows_out.append(
                [
                    bl,
                    "fixed_cutoff",
                    f"{thr:.4f}",
                    "",
                    m["tp"],
                    m["tn"],
                    m["fp"],
                    m["fn"],
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                ]
            )
        if bl == "full":
            m = row_for_threshold(ext_scores, ext_labels, val_thr_full)
            rows_out.append(
                [
                    bl,
                    "val_youden_then_external",
                    f"{val_thr_full:.4f}",
                    f"{val_j_full:.4f}",
                    m["tp"],
                    m["tn"],
                    m["fp"],
                    m["fn"],
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                ]
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path.as_posix()}")
    print(f"Validation Youden (full scores): threshold={val_thr_full:.4f}, J={val_j_full:.4f}")


if __name__ == "__main__":
    main()
