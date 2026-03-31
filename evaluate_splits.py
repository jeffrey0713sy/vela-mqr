"""
Evaluate Vela MQR results on frozen dataset splits.

Outputs:
- append one row to outputs/results_summary.csv
- write outputs/confusion_matrix.csv
- write outputs/metrics_report.md
- optional: append rows to outputs/bootstrap_external.csv (--bootstrap-b > 0)
"""

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path


def stable_market_key(market: dict) -> str:
    mid = str(market.get("id", "")).strip()
    if mid:
        return mid
    name = str(market.get("market_name", "")).strip()
    year = str(market.get("ref_year", "")).strip()
    return f"{name}::{year}"


def compute_auc(scores: list[float], labels: list[int], *, ndigits: int | None = 4) -> float:
    if len(set(labels)) < 2:
        val = 0.5
    else:
        concordant = 0.0
        total_pairs = 0
        for s_i, l_i in zip(scores, labels):
            for s_j, l_j in zip(scores, labels):
                if l_i == 1 and l_j == 0:
                    total_pairs += 1
                    if s_i > s_j:
                        concordant += 1.0
                    elif s_i == s_j:
                        concordant += 0.5
        val = (concordant / total_pairs) if total_pairs else 0.5
    return round(val, ndigits) if ndigits is not None else float(val)


def compute_ece(probs: list[float], labels: list[int], n_bins: int = 5) -> float:
    if not probs:
        return 1.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    n = len(probs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = sum(x[0] for x in b) / len(b)
        avg_y = sum(x[1] for x in b) / len(b)
        ece += (len(b) / n) * abs(avg_p - avg_y)
    return round(ece, 4)


def temperature_scale(raw_prob: float, t: float = 1.3) -> float:
    raw_prob = max(1e-6, min(1 - 1e-6, raw_prob))
    logit = math.log(raw_prob / (1 - raw_prob))
    scaled = logit / t
    return 1 / (1 + math.exp(-scaled))


def fit_temperature(probs: list[float], labels: list[int]) -> float:
    best_t = 1.0
    best_ece = float("inf")
    for t in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        scaled = [temperature_scale(p, t) for p in probs]
        ece = compute_ece(scaled, labels)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t


def sigmoid_prob_from_score(score: float) -> float:
    logit = -2.0 + 0.04 * float(score)
    logit = max(-500, min(500, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def confusion_binary(preds: list[int], labels: list[int]) -> dict:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def prf1_from_conf(conf: dict) -> tuple[float, float, float]:
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def top_decile_metrics(scores: list[float], labels: list[int]) -> tuple[int, float, float, float]:
    """
    Compute robust ranking metric:
    - select top 10% by score (at least 1 sample)
    - hit rate among selected
    - lift vs observed positive rate in this split
    """
    n = len(scores)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    k = max(1, math.ceil(n * 0.10))
    idx_sorted = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_idx = idx_sorted[:k]
    top_hits = sum(labels[i] for i in top_idx)
    top_hit_rate = top_hits / k if k else 0.0
    pos_rate = (sum(labels) / n) if n else 0.0
    lift_vs_pos_rate = (top_hit_rate / pos_rate) if pos_rate > 0 else 0.0
    return k, top_hit_rate, pos_rate, lift_vs_pos_rate


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _scores_for_markets(
    eval_markets: list[dict],
    train_ids: list[str],
    by_key: dict,
    baseline: str,
    rng: random.Random,
) -> tuple[list[float], list[int], list[int]]:
    scores_full = [float(m["mqr_score"]) for m in eval_markets]
    labels = [1 if m.get("t5_outcome", {}).get("achieved_scale") else 0 for m in eval_markets]
    if baseline == "full":
        scores = list(scores_full)
        pred_l5 = [1 if str(m.get("mqr_rating", "")).upper() == "L5" else 0 for m in eval_markets]
    elif baseline == "random":
        scores = [rng.uniform(0, 100) for _ in eval_markets]
        pred_l5 = [1 if rng.random() < 0.10 else 0 for _ in eval_markets]
    else:
        train_markets = [by_key[k] for k in train_ids if k in by_key]
        train_eval = [m for m in train_markets if m.get("t5_outcome", {}).get("achieved_scale") is not None]
        _ = (
            sum(1 if m.get("t5_outcome", {}).get("achieved_scale") else 0 for m in train_eval) / len(train_eval)
            if train_eval
            else 0.5
        )
        naive_score = sum(scores_full) / len(scores_full)
        scores = [naive_score for _ in eval_markets]
        pred_l5 = [0 for _ in eval_markets]
    return scores, labels, pred_l5


def load_split_eval(
    master: dict,
    split_keys: list[str],
    train_ids: list[str],
    by_key: dict,
    baseline: str,
    rng: random.Random,
) -> tuple[list[float], list[int], list[int]]:
    test_markets = [by_key[k] for k in split_keys if k in by_key]
    eval_markets = [
        m
        for m in test_markets
        if m.get("mqr_score") is not None and m.get("t5_outcome", {}).get("achieved_scale") is not None
    ]
    if not eval_markets:
        raise ValueError("no evaluable markets in split")
    return _scores_for_markets(eval_markets, train_ids, by_key, baseline, rng)


def bootstrap_auc_lift(
    scores: list[float],
    labels: list[int],
    *,
    n_boot: int,
    boot_seed: int,
) -> tuple[list[float], list[float]]:
    """Pair bootstrap on external-test rows; returns lists of AUC and top-decile lift per draw."""
    n = len(labels)
    rng = random.Random(boot_seed)
    aucs: list[float] = []
    lifts: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s_b = [scores[i] for i in idx]
        l_b = [labels[i] for i in idx]
        aucs.append(compute_auc(s_b, l_b, ndigits=None))
        _k, _hr, _pr, lift = top_decile_metrics(s_b, l_b)
        lifts.append(lift)
    return aucs, lifts


def bootstrap_ci_sorted(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    s = sorted(samples)
    n = len(s)
    k_lo = int((alpha / 2) * n)
    k_hi = int((1 - alpha / 2) * n) - 1
    k_hi = max(k_lo, min(n - 1, k_hi))
    return s[k_lo], s[k_hi]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MQR on frozen split files.")
    parser.add_argument("--master", type=str, default="data/reference_population_master.json")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--model-config", type=str, default="full_pipeline_v2.1")
    parser.add_argument("--baseline", type=str, default="full", choices=["full", "random", "naive"])
    parser.add_argument("--threshold", type=float, default=60.0, help="binary threshold on mqr_score")
    parser.add_argument("--random-seed", type=int, default=20260324)
    parser.add_argument("--bootstrap-b", type=int, default=0, help="if >0, percentile CI for external AUC and top-decile lift")
    parser.add_argument("--bootstrap-seed", type=int, default=20260331)
    parser.add_argument("--runtime-min", type=float, default=0.0)
    parser.add_argument("--estimated-cost-usd", type=float, default=0.0)
    args = parser.parse_args()

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    master = read_json(Path(args.master))
    markets = master.get("markets", [])
    by_key = {stable_market_key(m): m for m in markets}

    splits_dir = Path(args.splits_dir)
    train_ids = read_json(splits_dir / "train.json")
    val_ids = read_json(splits_dir / "validation.json")
    test_ids = read_json(splits_dir / "external_test.json")

    # Evaluate on external test only
    rng = random.Random(args.random_seed)
    scores, labels, pred_l5 = load_split_eval(master, test_ids, train_ids, by_key, args.baseline, rng)
    test_markets = [by_key[k] for k in test_ids if k in by_key]
    eval_markets = [
        m
        for m in test_markets
        if m.get("mqr_score") is not None and m.get("t5_outcome", {}).get("achieved_scale") is not None
    ]
    if args.baseline == "full":
        probs = [sigmoid_prob_from_score(s) for s in scores]
    elif args.baseline == "random":
        probs = [s / 100.0 for s in scores]
    else:
        train_markets = [by_key[k] for k in train_ids if k in by_key]
        train_eval = [m for m in train_markets if m.get("t5_outcome", {}).get("achieved_scale") is not None]
        train_pos_rate = (
            sum(1 if m.get("t5_outcome", {}).get("achieved_scale") else 0 for m in train_eval) / len(train_eval)
            if train_eval else 0.5
        )
        probs = [train_pos_rate for _ in eval_markets]

    auc = compute_auc(scores, labels)
    ece_before = compute_ece(probs, labels)
    t_opt = fit_temperature(probs, labels)
    probs_scaled = [temperature_scale(p, t_opt) for p in probs]
    ece_after = compute_ece(probs_scaled, labels)

    # Binary metrics from threshold on score
    preds = [1 if s >= args.threshold else 0 for s in scores]
    conf = confusion_binary(preds, labels)
    precision, recall, f1 = prf1_from_conf(conf)

    # L5 metrics
    n_pred_l5 = sum(pred_l5)
    actual_positive = labels
    l5_tp = sum(1 for p, y in zip(pred_l5, actual_positive) if p == 1 and y == 1)
    l5_fn = sum(1 for p, y in zip(pred_l5, actual_positive) if p == 0 and y == 1)
    l5_precision = (l5_tp / n_pred_l5) if n_pred_l5 else 0.0
    l5_recall = (l5_tp / (l5_tp + l5_fn)) if (l5_tp + l5_fn) else 0.0
    l5_hit_rate = l5_precision
    l5_random_rate = 0.10
    l5_lift = (l5_hit_rate / l5_random_rate) if l5_random_rate > 0 else 0.0
    top_k, top10_hit_rate, pos_rate, top10_lift_vs_pos = top_decile_metrics(scores, labels)

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    boot_md = ""
    if args.bootstrap_b > 0:
        aucs_b, lifts_b = bootstrap_auc_lift(
            scores, labels, n_boot=args.bootstrap_b, boot_seed=args.bootstrap_seed
        )
        auc_lo, auc_hi = bootstrap_ci_sorted(aucs_b)
        lift_lo, lift_hi = bootstrap_ci_sorted(lifts_b)
        boot_path = outputs_dir / "bootstrap_external.csv"
        boot_hdr = [
            "run_id",
            "baseline",
            "bootstrap_B",
            "bootstrap_seed",
            "auc_point",
            "auc_ci_low",
            "auc_ci_high",
            "top10_lift_point",
            "top10_lift_ci_low",
            "top10_lift_ci_high",
        ]
        boot_row = [
            run_id,
            args.baseline,
            args.bootstrap_b,
            args.bootstrap_seed,
            f"{auc:.4f}",
            f"{auc_lo:.4f}",
            f"{auc_hi:.4f}",
            f"{top10_lift_vs_pos:.4f}",
            f"{lift_lo:.4f}",
            f"{lift_hi:.4f}",
        ]
        boot_ex = boot_path.exists()
        with boot_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not boot_ex:
                w.writerow(boot_hdr)
            w.writerow(boot_row)
        boot_md = f"""
## Bootstrap (external test; {args.bootstrap_b} resamples, seed={args.bootstrap_seed})
- AUC: {auc:.4f} (95% percentile interval: {auc_lo:.4f}–{auc_hi:.4f})
- Top-decile lift vs positive rate: {top10_lift_vs_pos:.4f} (95% interval: {lift_lo:.4f}–{lift_hi:.4f})
- Rows appended to `{boot_path.as_posix()}`
"""

    # Append summary row
    summary_path = outputs_dir / "results_summary.csv"
    header = [
        "run_id",
        "model_config",
        "n_total",
        "n_train",
        "n_validation",
        "n_external_test",
        "auc",
        "ece_before",
        "ece_after",
        "l5_precision",
        "l5_recall",
        "l5_hit_rate",
        "l5_random_rate",
        "l5_lift_vs_random",
        "runtime_min",
        "estimated_cost_usd",
        "notes",
    ]
    row = [
        run_id,
        args.model_config,
        len(eval_markets),
        len(train_ids),
        len(val_ids),
        len(test_ids),
        f"{auc:.4f}",
        f"{ece_before:.4f}",
        f"{ece_after:.4f}",
        f"{l5_precision:.4f}",
        f"{l5_recall:.4f}",
        f"{l5_hit_rate:.4f}",
        f"{l5_random_rate:.2f}",
        f"{l5_lift:.4f}",
        f"{args.runtime_min:.2f}",
        f"{args.estimated_cost_usd:.2f}",
        (
            f"baseline={args.baseline}, threshold={args.threshold}, T_opt={t_opt}, seed={args.random_seed}, "
            f"top10_k={top_k}, top10_hit={top10_hit_rate:.4f}, pos_rate={pos_rate:.4f}, top10_lift={top10_lift_vs_pos:.4f}"
        ),
    ]

    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

    # Write/append confusion matrix
    conf_path = outputs_dir / "confusion_matrix.csv"
    conf_exists = conf_path.exists()
    with conf_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not conf_exists:
            writer.writerow(["run_id", "baseline", "threshold", "tp", "tn", "fp", "fn", "precision", "recall", "f1"])
        writer.writerow([
            run_id,
            args.baseline,
            args.threshold,
            conf["tp"],
            conf["tn"],
            conf["fp"],
            conf["fn"],
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
        ])

    # Write report
    report = f"""# Metrics Report

## Run Metadata
- Run ID: {run_id}
- Model config: {args.model_config}
- Baseline mode: {args.baseline}
- Threshold: {args.threshold}

## Dataset Summary
- N total (evaluated on external_test with labels): {len(eval_markets)}
- N train: {len(train_ids)}
- N validation: {len(val_ids)}
- N external_test: {len(test_ids)}
- Positive rate (external_test eval subset): {sum(labels)/len(labels):.4f}

## Core Metrics
- AUC: {auc:.4f}
- ECE before scaling: {ece_before:.4f}
- ECE after scaling: {ece_after:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1: {f1:.4f}

## L5 Performance
- L5 precision: {l5_precision:.4f}
- L5 recall: {l5_recall:.4f}
- L5 hit rate: {l5_hit_rate:.4f}
- Random baseline hit rate: {l5_random_rate:.2f}
- L5 lift vs random: {l5_lift:.4f}

## Robust Top-Decile Ranking Metric
- Top-decile K: {top_k}
- Top-decile hit rate: {top10_hit_rate:.4f}
- Observed positive rate: {pos_rate:.4f}
- Top-decile lift vs positive rate: {top10_lift_vs_pos:.4f}
{boot_md}
## Files Generated
- {summary_path.as_posix()}
- {conf_path.as_posix()}
"""
    write_text(outputs_dir / "metrics_report.md", report)

    print("Evaluation completed.")
    print(f"Run ID: {run_id}")
    print(f"Baseline mode: {args.baseline}")
    print(f"Evaluated N: {len(eval_markets)}")
    print(f"AUC: {auc:.4f} | ECE(before/after): {ece_before:.4f}/{ece_after:.4f}")
    print(f"L5 precision: {l5_precision:.4f} | L5 lift: {l5_lift:.4f}")
    print(
        f"Top-decile: k={top_k} | hit_rate={top10_hit_rate:.4f} | "
        f"pos_rate={pos_rate:.4f} | lift={top10_lift_vs_pos:.4f}"
    )
    if args.bootstrap_b > 0:
        print(
            f"Bootstrap ({args.bootstrap_b}): AUC CI in report; "
            f"see {outputs_dir / 'bootstrap_external.csv'}"
        )
    print(f"Updated: {summary_path}")
    print(f"Wrote: {conf_path}")
    print(f"Wrote: {outputs_dir / 'metrics_report.md'}")


if __name__ == "__main__":
    main()
