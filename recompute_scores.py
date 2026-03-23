"""
Vela MQR — recompute_scores.py
用已有的 step1/step2 数据重新计算所有市场的分数。

特点：
- 零 API 调用，本地纯 Python 计算，几秒钟跑完
- 适用于修改了 config.py 或 pipeline_step3.py 评分逻辑后的批量更新
- 输出到新文件，保留原始 reference_population_master.json 不变
- 打印变动摘要，方便对比前后差异

用法：
    python recompute_scores.py
    python recompute_scores.py --input data/reference_population_master.json
    python recompute_scores.py --output data/reference_population_v21.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pipeline_step3 import (
    compute_dimension_scores,
    compute_composite_score,
    assign_rating,
    compute_outlook,
    evaluate_population,
)
from config import DIMENSIONS


def recompute(input_path: str, output_path: str) -> None:
    print("=" * 62)
    print("  Vela MQR — 重新计算分数（v2.1 逻辑）")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print("=" * 62)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    markets = data.get("markets", [])
    print(f"\n  总市场数: {len(markets)}")

    # 统计变动
    rating_before  = defaultdict(int)
    rating_after   = defaultdict(int)
    score_deltas   = []
    skipped        = 0
    changed_rating = 0

    for i, m in enumerate(markets):
        step1 = m.get("step1")
        step2 = m.get("step2")

        if not step1 or not step2:
            skipped += 1
            continue

        old_score  = m.get("mqr_score", None)
        old_rating = m.get("mqr_rating", "?")
        rating_before[old_rating] += 1

        # ── 重算维度分数 ──────────────────────────────────────
        dim_scores = compute_dimension_scores(step1, step2)

        # ── 重算复合分（不用 Role 3，因为现有数据没有 Role 3）──
        # 如果原始数据有 Role 3，使用 Role 3 分数
        role3_result = m.get("scoring", {}).get("role3_result")
        used_role3   = bool(role3_result) and m.get("scoring", {}).get("composite", {}).get("used_role3", False)

        composite = compute_composite_score(
            dim_scores,
            use_role3=used_role3,
            role3_result=role3_result,
        )

        # ── 重算展望和评级 ────────────────────────────────────
        outlook = compute_outlook(dim_scores, step2)
        rating  = assign_rating(composite["composite_score"], outlook)

        new_score  = composite["composite_score"]
        new_rating = rating["rating"]
        rating_after[new_rating] += 1

        # 记录变动
        if old_score is not None:
            delta = round(new_score - old_score, 2)
            score_deltas.append(delta)

        if old_rating != new_rating:
            changed_rating += 1
            print(f"  [{i+1:>3}] {m.get('market_name','?')[:45]:<45} "
                  f"{old_rating} → {new_rating}  "
                  f"(分: {old_score:.1f} → {new_score:.1f})")

        # ── 更新市场数据 ──────────────────────────────────────
        m["mqr_score"]  = new_score
        m["mqr_rating"] = new_rating

        if "scoring" not in m:
            m["scoring"] = {}
        m["scoring"]["dimension_scores"] = dim_scores
        m["scoring"]["composite"]        = composite
        m["scoring"]["rating"]           = rating

    # ── 打印摘要 ─────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  重算完成  |  跳过（无 step1/2）: {skipped}  |  评级变动: {changed_rating}")

    if score_deltas:
        avg_delta = sum(score_deltas) / len(score_deltas)
        max_delta = max(score_deltas, key=abs)
        print(f"  分数变动  |  平均: {avg_delta:+.2f}  |  最大: {max_delta:+.2f}")

    print(f"\n  {'评级':<6} {'变动前':>6} {'变动后':>6}")
    print(f"  {'─'*6} {'─'*6} {'─'*6}")
    for r in ["L5", "L4", "L3", "L2", "L1"]:
        b = rating_before.get(r, 0)
        a = rating_after.get(r, 0)
        diff = a - b
        diff_str = f"({diff:+d})" if diff != 0 else ""
        print(f"  {r:<6} {b:>6} {a:>6}  {diff_str}")

    # ── 重算整体 AUC/ECE ────────────────────────────────────
    print(f"\n  重算 AUC / ECE...")
    eval_result = evaluate_population(markets)
    if eval_result.get("auc") is not None:
        print(f"  AUC:            {eval_result['auc']}")
        print(f"  ECE (缩放前):   {eval_result['ece_before_scaling']}")
        print(f"  ECE (缩放后):   {eval_result['ece_after_scaling']}")
        print(f"  Naive ECE:      {eval_result['naive_ece']}")
        print(f"  vs 朴素基线:    {eval_result['ece_vs_naive']:+.4f}  {'' if eval_result['calibration_valid'] else '⚠ 差于基线'}")
        print(f"  {eval_result['calibration_note']}")

    # ── 保存 ────────────────────────────────────────────────
    data["recomputed_at"]  = datetime.now().isoformat()
    data["recompute_note"] = "v2.1: competition moved to RESIDUAL_DIMS; LOW agreement uses averaged raw_score"
    data["eval_result"]    = eval_result

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ 已保存: {output_path}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vela MQR 分数重算工具")
    parser.add_argument(
        "--input",
        default="data/reference_population_master.json",
        help="输入 JSON 路径",
    )
    parser.add_argument(
        "--output",
        default="data/reference_population_v21.json",
        help="输出 JSON 路径（不覆盖原文件）",
    )
    args = parser.parse_args()
    recompute(args.input, args.output)
