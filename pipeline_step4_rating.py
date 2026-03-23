"""
Vela MQR — pipeline_step4_rating.py
Step 4 升级版：百分位 + 逻辑回归 + 最近邻比较

有了参考人口（≥30个市场）后替代简单阈值评级。
"""

import argparse
import json
import math
from datetime import datetime
from config import DIMENSIONS


# ============================================================
# 工具函数
# ============================================================

def _sigmoid(x: float) -> float:
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def fit_logistic_regression(scores: list, labels: list) -> tuple:
    """梯度下降逻辑回归，纯 Python 实现。"""
    if len(scores) < 5:
        return -2.0, 0.04
    b0, b1 = -2.0, 0.04
    lr, n  = 0.001, len(scores)
    for _ in range(2000):
        g0, g1 = 0.0, 0.0
        for s, y in zip(scores, labels):
            err = _sigmoid(b0 + b1 * s) - y
            g0 += err
            g1 += err * s
        b0 -= lr * g0 / n
        b1 -= lr * g1 / n
    return round(b0, 4), round(b1, 4)


def get_percentile(score: float, pop: list) -> float:
    if not pop:
        return 50.0
    return round(sum(1 for s in pop if s < score) / len(pop) * 100, 1)


def percentile_to_rating(p: float) -> str:
    if p >= 90: return "L5"
    if p >= 70: return "L4"
    if p >= 45: return "L3"
    if p >= 25: return "L2"
    return "L1"


def cosine_sim(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    return round(dot / (na * nb), 4) if na and nb else 0.0


def get_dim_vector(market: dict) -> list:
    ds = market.get("scoring", {}).get("dimension_scores", {})
    return [ds.get(d, {}).get("adjusted_score", 50) / 100.0 for d in DIMENSIONS]


def nearest_neighbours(target_vec: list, pool: list, k: int = 3) -> list:
    sims = []
    for m in pool:
        if "scoring" not in m:
            continue
        v = get_dim_vector(m)
        if len(v) == len(target_vec):
            sims.append((cosine_sim(target_vec, v), m))
    sims.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, m in sims[:k]:
        t5 = m.get("t5_outcome", {})
        results.append({
            "market_name":     m.get("market_name", "?"),
            "ref_year":        m.get("ref_year", "?"),
            "mqr_rating":      m.get("mqr_rating", "?"),
            "mqr_score":       m.get("mqr_score", 0),
            "achieved_scale":  t5.get("achieved_scale"),
            "outcome_summary": t5.get("outcome_summary", "")[:120],
            "similarity":      sim,
        })
    return results


# ============================================================
# 核心：单市场评级
# ============================================================

def rate_with_population(
    composite_score: float,
    market_structure_type: str,
    dimension_scores: dict,
    population: list,
) -> dict:
    """使用参考人口为单个市场生成完整评级。"""

    # 同类型市场优先，不足 10 个时用全体
    same = [
        m for m in population
        if m.get("scoring", {}).get("composite", {}).get("market_structure_type") == market_structure_type
        and "mqr_score" in m
    ]
    ref = same if len(same) >= 10 else [m for m in population if "mqr_score" in m]

    pop_scores = [m["mqr_score"] for m in ref]
    pct        = get_percentile(composite_score, pop_scores)
    rating     = percentile_to_rating(pct)

    # 逻辑回归（需要 T+5 标签）
    lr_data = [(m["mqr_score"], 1 if m.get("t5_outcome", {}).get("achieved_scale") else 0)
               for m in ref if "mqr_score" in m and m.get("t5_outcome")]
    b0, b1 = fit_logistic_regression(
        [x[0] for x in lr_data],
        [x[1] for x in lr_data],
    ) if len(lr_data) >= 5 else (-2.0, 0.04)

    p_scale      = _sigmoid(b0 + b1 * composite_score)
    p_out_of_20  = round(p_scale * 20)

    # 最近邻
    target_vec = [
        dimension_scores.get(d, {}).get("adjusted_score", 50) / 100.0
        for d in DIMENSIONS
    ]
    nn = nearest_neighbours(target_vec, ref, k=3)

    return {
        "rating":             rating,
        "percentile":         pct,
        "p_scale":            round(p_scale, 3),
        "p_scale_text":       f"约 {p_out_of_20}/20 的同类市场实现了规模化",
        "nearest_neighbours": nn,
        "population_size":    len(ref),
        "same_type_size":     len(same),
        "regression_params":  {"b0": b0, "b1": b1},
    }


# ============================================================
# 批量：对整个参考人口重新评级
# ============================================================

def rate_population(input_path: str, output_path: str) -> None:
    print(f"  加载: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    markets = data["markets"]
    scored  = [m for m in markets if "mqr_score" in m]
    print(f"  总市场: {len(markets)}  已评分: {len(scored)}")

    for i, m in enumerate(scored):
        composite  = m["mqr_score"]
        mst        = m.get("scoring", {}).get("composite", {}).get(
                         "market_structure_type", "technology_enablement")
        dim_scores = m.get("scoring", {}).get("dimension_scores", {})

        # 留一法：排除自身
        pop = [x for x in scored if x.get("market_name") != m.get("market_name")]

        r4 = rate_with_population(composite, mst, dim_scores, pop)
        m["step4_rating"] = r4
        m["mqr_rating"]   = r4["rating"]

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(scored)}")

    data["rated_at"] = datetime.now().isoformat()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 分布统计
    dist = {}
    for m in markets:
        r = m.get("mqr_rating", "?")
        dist[r] = dist.get(r, 0) + 1

    print(f"\n  保存: {output_path}")
    print("  评级分布：")
    for r in ["L5", "L4", "L3", "L2", "L1"]:
        c = dist.get(r, 0)
        print(f"  {r}: {c:3d}  {'█' * max(c // 2, 0)}")


# ============================================================
# 命令行
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/reference_population_master.json")
    p.add_argument("--output", default="data/rated_population.json")
    args = p.parse_args()
    rate_population(args.input, args.output)
