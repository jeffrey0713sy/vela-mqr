"""
Idea Forecasting evaluation: temporal split (train <=2015, test 2016-2017).

Matches Vela Summer 2025 brief: inception-visible text + founding year only
(no total_funding_usd as a feature). Labels: is_outlier (~$250M proxy).

Outputs: outputs/idea_forecast/metrics.json, summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def parse_year(founded_on: Any) -> int | None:
    if founded_on is None:
        return None
    s = str(founded_on).strip()
    if len(s) >= 4 and s[:4].isdigit():
        y = int(s[:4])
        return y if 1990 <= y <= 2030 else None
    return None


def row_text(row: dict[str, str]) -> str:
    parts = [
        row.get("short_description") or "",
        row.get("long_description") or "",
        row.get("search_context") or "",
    ]
    yr = parse_year(row.get("founded_on") or row.get("founded_at"))
    if yr is not None:
        parts.append(f"founding_year {yr}")
    return " \n ".join(p for p in parts if p).strip()


def row_label(row: dict[str, str]) -> int:
    v = row.get("is_outlier", "0")
    if isinstance(v, (int, float)):
        return 1 if float(v) >= 1.0 else 0
    s = str(v).strip().lower()
    return 1 if s in ("1", "1.0", "true", "yes") else 0


def roc_auc_trapezoid(scores: list[float], labels: list[int]) -> float:
    if len(set(labels)) < 2:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    p = sum(labels)
    n = len(labels) - p
    tp = fp = 0
    auc = 0.0
    prev_tpr = prev_fpr = 0.0
    for _, y in paired:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / p
        fpr = fp / n
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr
    return round(auc, 6)


def average_precision(scores: list[float], labels: list[int]) -> float:
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    tp = 0
    ap = 0.0
    for i, (_, y) in enumerate(paired, start=1):
        if y == 1:
            tp += 1
            ap += tp / i
    return round(ap / n_pos, 6)


def precision_at_k(scores: list[float], labels: list[int], k: int) -> float:
    n = len(scores)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    idx = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    hits = sum(labels[i] for i in idx)
    return hits / k


def lift_at_k(scores: list[float], labels: list[int], k: int) -> float:
    base = sum(labels) / len(labels) if labels else 0.0
    if base <= 0:
        return 0.0
    return precision_at_k(scores, labels, k) / base


def top_decile_lift(scores: list[float], labels: list[int]) -> tuple[int, float, float, float]:
    n = len(scores)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    k = max(1, math.ceil(n * 0.10))
    p = precision_at_k(scores, labels, k)
    base = sum(labels) / n
    lift = p / base if base > 0 else 0.0
    return k, p, base, lift


def bootstrap_ci_sorted(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    s = sorted(samples)
    n = len(s)
    k_lo = int((alpha / 2) * n)
    k_hi = int((1 - alpha / 2) * n) - 1
    k_hi = max(k_lo, min(n - 1, k_hi))
    return s[k_lo], s[k_hi]


def fit_lexicon_log_odds(train_texts: list[str], train_y: list[int], year_tokens: bool) -> dict[str, float]:
    pos = sum(train_y)
    neg = len(train_y) - pos
    if pos == 0 or neg == 0:
        return {}
    from collections import Counter

    pos_c: Counter[str] = Counter()
    neg_c: Counter[str] = Counter()
    for t, y in zip(train_texts, train_y):
        toks = set(tokenize(t))
        if year_tokens:
            for m in re.finditer(r"\b(20\d{2})\b", t):
                toks.add(f"__y{m.group(1)}__")
        for w in toks:
            if y == 1:
                pos_c[w] += 1
            else:
                neg_c[w] += 1
    vocab = set(pos_c) | set(neg_c)
    log_odds: dict[str, float] = {}
    a = 0.5
    for w in vocab:
        pw = pos_c[w]
        nw = neg_c[w]
        lo = math.log((pw + a) / (pos - pw + a)) - math.log((nw + a) / (neg - nw + a))
        if abs(lo) > 1e-6:
            log_odds[w] = lo
    return log_odds


def predict_lexicon(texts: list[str], log_odds: dict[str, float], year_tokens: bool) -> list[float]:
    scores = []
    for t in texts:
        toks = set(tokenize(t))
        if year_tokens:
            for m in re.finditer(r"\b(20\d{2})\b", t):
                toks.add(f"__y{m.group(1)}__")
        s = sum(log_odds.get(w, 0.0) for w in toks)
        scores.append(s)
    return scores


def fit_predict_tfidf_lr(
    train_texts: list[str],
    train_y: list[int],
    test_texts: list[str],
) -> list[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    ntr = len(train_texts)
    min_df = 3 if ntr > 2000 else 1
    max_feat = min(80000, max(512, ntr * 20))

    vec = TfidfVectorizer(
        max_features=max_feat,
        min_df=min_df,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_tr = vec.fit_transform(train_texts)
    clf = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        random_state=42,
        solver="saga",
    )
    clf.fit(X_tr, train_y)
    X_te = vec.transform(test_texts)
    return clf.predict_proba(X_te)[:, 1].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Idea forecasting temporal eval")
    parser.add_argument("--csv", type=str, default="data/idea_training/idea_training.csv")
    parser.add_argument(
        "--scorer",
        type=str,
        choices=["random", "lexicon", "tfidf_lr", "llm"],
        default="lexicon",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-year", type=int, default=2015)
    parser.add_argument("--test-min-year", type=int, default=2016)
    parser.add_argument("--test-max-year", type=int, default=2017)
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--bootstrap-b", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260331)
    parser.add_argument("--year-tokens", action="store_true", help="add __y2014__ style tokens from text")
    parser.add_argument("--llm-max-rows", type=int, default=150)
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--out-dir", type=str, default="outputs/idea_forecast")
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-test-rows", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    csv_path = (root / args.csv).resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Missing CSV: {csv_path} — run scripts/import_idea_training.py first")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    parsed: list[tuple[dict[str, str], int, int]] = []
    for row in rows:
        y = row_label(row)
        yr = parse_year(row.get("founded_on") or row.get("founded_at"))
        if yr is None or yr < args.min_year or yr > args.test_max_year:
            continue
        parsed.append((row, y, yr))

    train = [(r, y) for r, y, yr in parsed if yr <= args.train_max_year]
    test = [(r, y) for r, y, yr in parsed if args.test_min_year <= yr <= args.test_max_year]

    if len(train) < args.min_train_rows or len(test) < args.min_test_rows:
        raise SystemExit(
            f"Too few rows: train={len(train)} test={len(test)} "
            f"(need ≥{args.min_train_rows} / ≥{args.min_test_rows}, or lower flags for smoke tests)"
        )

    train_texts = [row_text(r) for r, _ in train]
    train_y = [y for _, y in train]
    test_texts = [row_text(r) for r, _ in test]
    test_y = [y for _, y in test]

    rng = random.Random(args.seed)
    scores: list[float]

    if args.scorer == "random":
        scores = [rng.random() for _ in test_texts]
    elif args.scorer == "lexicon":
        lo = fit_lexicon_log_odds(train_texts, train_y, year_tokens=args.year_tokens)
        scores = predict_lexicon(test_texts, lo, year_tokens=args.year_tokens)
    elif args.scorer == "tfidf_lr":
        try:
            scores = fit_predict_tfidf_lr(train_texts, train_y, test_texts)
        except ImportError as e:
            raise SystemExit("Install scikit-learn: pip install -r requirements-idea.txt") from e
    else:
        from idea_llm_scorer import score_ideas_llm

        n = min(args.llm_max_rows, len(test_texts))
        if n < len(test_texts):
            print(f"LLM: scoring first {n} of {len(test_texts)} test rows only")
        sub = test_texts[:n]
        scores = score_ideas_llm(sub, model=args.llm_model)
        test_y = test_y[:n]
        test_texts = sub

    base = sum(test_y) / len(test_y)
    auc = roc_auc_trapezoid(scores, test_y)
    ap = average_precision(scores, test_y)
    k10, p10, _, lift10 = top_decile_lift(scores, test_y)

    ks = sorted(
        set(
            [
                10,
                50,
                100,
                500,
                max(1, len(test_y) // 100),
                max(1, len(test_y) // 20),
                k10,
            ]
        )
    )
    pk = {f"p@{k}": round(precision_at_k(scores, test_y, k), 6) for k in ks if k <= len(test_y)}
    lifts = {f"lift@{k}": round(lift_at_k(scores, test_y, k), 6) for k in ks if k <= len(test_y)}

    boot = {}
    if args.bootstrap_b > 0:
        rng_b = random.Random(args.bootstrap_seed)
        n = len(test_y)
        aucs, aps, lifts_b = [], [], []
        for _ in range(args.bootstrap_b):
            idx = [rng_b.randrange(n) for _ in range(n)]
            s_b = [scores[i] for i in idx]
            y_b = [test_y[i] for i in idx]
            aucs.append(roc_auc_trapezoid(s_b, y_b))
            aps.append(average_precision(s_b, y_b))
            _, _, _, l10 = top_decile_lift(s_b, y_b)
            lifts_b.append(l10)
        boot = {
            "B": args.bootstrap_b,
            "seed": args.bootstrap_seed,
            "auc_ci": list(bootstrap_ci_sorted(aucs)),
            "ap_ci": list(bootstrap_ci_sorted(aps)),
            "top_decile_lift_ci": list(bootstrap_ci_sorted(lifts_b)),
        }

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "scorer": args.scorer,
        "n_train": len(train_y),
        "n_test": len(test_y),
        "train_pos_rate": round(sum(train_y) / len(train_y), 6),
        "test_pos_rate": round(base, 6),
        "train_max_year": args.train_max_year,
        "test_years": [args.test_min_year, args.test_max_year],
        "auc": auc,
        "average_precision": ap,
        "top_decile_k": k10,
        "top_decile_precision": round(p10, 6),
        "top_decile_lift_vs_base": round(lift10, 6),
        "precision_at_k": pk,
        "lift_at_k": lifts,
        "bootstrap": boot,
        "csv": str(csv_path.relative_to(root)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = f"""# Idea forecast eval ({args.scorer})

- Train: founded year ≤ {args.train_max_year} (n={len(train_y)}, pos_rate={report['train_pos_rate']:.4f})
- Test: {args.test_min_year}–{args.test_max_year} (n={len(test_y)}, pos_rate={report['test_pos_rate']:.4f})
- AUC: **{auc}** | PR-AUC (average precision): **{ap}**
- Top-decile: k={k10}, precision={p10:.4f}, lift vs base={lift10:.4f}x

## Precision / lift at k
{json.dumps(pk, indent=2)}

{json.dumps(lifts, indent=2)}

## Bootstrap
{json.dumps(boot, indent=2) if boot else "(disabled)"}
"""
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "precision_at_k" and k != "lift_at_k"}, indent=2))
    print(f"Wrote {out_dir / 'metrics.json'} and summary.md")


if __name__ == "__main__":
    main()
