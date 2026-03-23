"""
Vela MQR — pipeline_step3.py
Step 3 + Step 4：分类 → 分数 → 复合分 → L1-L5 评级

v2.1 变更：
1. 修复 LOW agreement 分数计算 bug：
   原逻辑计算了 avg 但实际用的是 r1_cls 对应的分数，avg 被丢弃。
   修正为：LOW agreement 时用 (r1_score + g_score) / 2 作为 raw_score。
2. 修复 Role 3 prompt 中 adopted=60（应为70）和 defined=80（应为75）的笔误。
3. evaluate_population 加入朴素基线 ECE，便于判断校准是否有实际价值。
4. assign_rating 加入 near_boundary 标记，避免压线评级误导。
"""

import json
import math
import os
import re
import time

from config import (
    CLASSIFICATION_SCORES,
    CAUSAL_WEIGHTS,
    CONFIDENCE_MULTIPLIER,
    RATING_THRESHOLDS,
    RATING_BUFFER,
    DIMENSIONS,
    PRIMARY_DIMS,
    RESIDUAL_DIMS,
    RESIDUAL_WEIGHT,
    PRIMARY_WEIGHT,
)


# ============================================================
# Step 3A：分类词 → 数值分数（固定映射表）
# ============================================================

def classification_to_score(dim: str, classification: str) -> int:
    dim_map = CLASSIFICATION_SCORES.get(dim, {})
    cls = classification.strip().lower()
    if cls in dim_map:
        return dim_map[cls]
    for key, score in dim_map.items():
        if key in cls or cls in key:
            return score
    return 50


def compute_dimension_scores(step1_result: dict, step2_result: dict) -> dict:
    """
    对每个维度计算最终分数。

    agreement 策略：
      HIGH   → 用 Gemini 分类（已有网络搜索验证）
      MEDIUM → 取 r1/gemini 两个分数的平均值，final_cls 选更靠近平均的那个
      LOW    → 取 r1/gemini 两个分数的平均值作为 raw_score（v2.1 修复）
               original bug: avg 被计算但未使用，raw_score 仍来自 r1_cls

    置信度乘数：HIGH×1.0 / MEDIUM×0.88 / LOW×0.72×0.9=0.648
    """
    scores = {}

    for dim in DIMENSIONS:
        s2 = step2_result.get(dim, {})
        s1 = step1_result.get(dim, {})

        r1_cls    = s1.get("classification", "")
        g_cls     = s2.get("gemini_classification", r1_cls)
        agreement = s2.get("agreement", "MEDIUM")

        r1_score = classification_to_score(dim, r1_cls)
        g_score  = classification_to_score(dim, g_cls)

        if agreement == "HIGH":
            final_cls = g_cls
            raw_score = g_score
            source    = "gemini"

        elif agreement == "MEDIUM":
            avg       = (r1_score + g_score) / 2
            final_cls = g_cls if abs(g_score - avg) <= abs(r1_score - avg) else r1_cls
            raw_score = round(avg)
            source    = "averaged"

        else:  # LOW
            # v2.1 修复：使用两端平均而非单独用 r1_cls 的分数
            # 原代码计算了 avg 但 raw_score = classification_to_score(final_cls) 覆盖了它
            avg       = (r1_score + g_score) / 2
            final_cls = r1_cls   # 展示用，保留 Role 1 分类
            raw_score = round(avg)
            source    = "conservative_conflict"

        conf_mult = CONFIDENCE_MULTIPLIER.get(agreement.upper(), 0.88)
        if agreement == "LOW":
            conf_mult *= 0.9   # LOW 额外惩罚 → 最终 0.648
        adjusted = raw_score * conf_mult

        scores[dim] = {
            "role1_classification":  r1_cls,
            "gemini_classification": g_cls,
            "final_classification":  final_cls,
            "raw_score":             raw_score,
            "agreement":             agreement,
            "confidence_multiplier": round(conf_mult, 3),
            "adjusted_score":        round(adjusted, 1),
            "source":                source,
        }

    # 市场结构类型
    st           = step2_result.get("market_structure_type", {})
    st_agreement = st.get("agreement", "LOW")
    if st_agreement in ("HIGH", "MEDIUM"):
        market_structure_type = st.get("gemini_classification", "technology_enablement")
    else:
        market_structure_type = step1_result.get("market_structure_type", {}).get(
            "classification", "technology_enablement"
        )

    scores["market_structure_type"] = market_structure_type
    return scores


# ============================================================
# Step 3B：Role 3 匿名评分（核心升级）
# ============================================================

def score_with_role3(
    dimension_scores: dict,
    market_structure_type: str,
    openai_client,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> dict:
    """
    Role 3：用去掉市场名称的匿名特征矩阵让 GPT-4o 重新打分。
    防止模型用参数记忆直接猜测历史结果（结果泄漏）。

    v2.1 修复：prompt 中 adopted=70（原60）、defined=75（原80），与 config.py 对齐。
    """

    anonymous_features = {}
    for dim in DIMENSIONS:
        d = dimension_scores.get(dim, {})
        anonymous_features[dim] = {
            "classification": d.get("final_classification", "unknown"),
            "agreement":      d.get("agreement", "MEDIUM"),
            "confidence":     d.get("confidence_multiplier", 0.88),
        }

    prompt = f"""你是一个量化市场质量分析师。

以下是一个历史市场在某参考年份的结构化特征矩阵。
注意：没有市场名称、公司名称或任何可识别信息。
市场结构类型：{market_structure_type}

特征矩阵：
{json.dumps(anonymous_features, ensure_ascii=False, indent=2)}

请根据以下评分标准，为每个维度独立评分 0-100：

timing 评分标准：
  pre_chasm=15, early_chasm=40, early_majority=75, late_majority=55, peak=30, decline=10

competition 评分标准：
  nascent=85, fragmented=65, consolidating=40, concentrated=25, monopoly=10

market_size 评分标准：
  micro=20, small=45, medium=65, large=80, massive=95

customer_readiness 评分标准：
  unaware=10, aware=25, interested=45, ready=65, adopting=85, adopted=70

regulatory 评分标准：
  unregulated=60, light_touch=85, moderate=55, heavy=25, prohibited=5

infrastructure 评分标准：
  undefined=10, emerging=30, developing=55, mature=80, commoditized=90

market_structure 评分标准：
  undefined=20, emerging=40, forming=60, defined=75, mature=65

根据 agreement 调整：HIGH→不调整，MEDIUM→×0.88，LOW→×0.648
IMPORTANT: Use the full 0-100 range. Do NOT cluster scores between 50-75.
A truly poor dimension should score 10-20. An exceptional one should score 85-95.

严格输出 JSON，不加任何解释：
{{
  "timing": <0-100>,
  "competition": <0-100>,
  "market_size": <0-100>,
  "customer_readiness": <0-100>,
  "regulatory": <0-100>,
  "infrastructure": <0-100>,
  "market_structure": <0-100>,
  "rationale": "<一句话：最影响评分的关键特征>"
}}"""

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"未找到 JSON：{raw[:100]}")

            parsed = json.loads(match.group(0))
            result = {}
            for dim in DIMENSIONS:
                score = parsed.get(dim, dimension_scores[dim]["adjusted_score"])
                result[dim] = {
                    "role3_score":   round(float(score), 1),
                    "mapping_score": dimension_scores[dim]["adjusted_score"],
                    "delta":         round(float(score) - dimension_scores[dim]["adjusted_score"], 1),
                }
            result["rationale"] = parsed.get("rationale", "")
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
                print(f"  [Role 3 重试 {attempt+1}] {e}")
            else:
                print(f"  [Role 3 失败，使用固定映射分数] {e}")
                return {
                    dim: {
                        "role3_score":   dimension_scores[dim]["adjusted_score"],
                        "mapping_score": dimension_scores[dim]["adjusted_score"],
                        "delta":         0.0,
                    }
                    for dim in DIMENSIONS
                }


# ============================================================
# Step 4：复合分计算（因果权重）
# ============================================================

def compute_composite_score(
    dimension_scores: dict,
    use_role3: bool = False,
    role3_result: dict = None,
) -> dict:
    """
    用市场结构类型特定的因果权重计算复合分。
    如果有 Role 3 结果，使用 Role 3 分数；否则用固定映射分数。

    v2.1：PRIMARY_DIMS = [timing, market_size]（共 2 个）
          RESIDUAL_DIMS = [competition, customer_readiness, regulatory,
                           infrastructure, market_structure]（共 5 个）
    """
    mst     = dimension_scores.get("market_structure_type", "technology_enablement")
    weights = CAUSAL_WEIGHTS.get(mst, CAUSAL_WEIGHTS["technology_enablement"])

    primary_weighted_sum = 0.0
    primary_weight_total = sum(weights[d] for d in PRIMARY_DIMS)
    dim_contributions    = {}

    for dim in PRIMARY_DIMS:
        w = weights[dim]
        if use_role3 and role3_result and dim in role3_result:
            adjusted = role3_result[dim]["role3_score"]
        else:
            adjusted = dimension_scores[dim]["adjusted_score"]

        effective_weight      = (w / primary_weight_total) * PRIMARY_WEIGHT
        contribution          = adjusted * effective_weight
        primary_weighted_sum += contribution
        dim_contributions[dim] = {
            "weight":         round(effective_weight, 3),
            "adjusted_score": adjusted,
            "contribution":   round(contribution, 2),
        }

    residual_scores = []
    for dim in RESIDUAL_DIMS:
        if use_role3 and role3_result and dim in role3_result:
            residual_scores.append(role3_result[dim]["role3_score"])
        else:
            residual_scores.append(dimension_scores[dim]["adjusted_score"])

    residual_avg          = sum(residual_scores) / len(residual_scores)
    residual_contribution = residual_avg * RESIDUAL_WEIGHT

    for i, dim in enumerate(RESIDUAL_DIMS):
        effective_weight = RESIDUAL_WEIGHT / len(RESIDUAL_DIMS)
        dim_contributions[dim] = {
            "weight":         round(effective_weight, 3),
            "adjusted_score": residual_scores[i],
            "contribution":   round(residual_scores[i] * effective_weight, 2),
        }

    composite = round(primary_weighted_sum + residual_contribution, 2)

    return {
        "composite_score":         composite,
        "market_structure_type":   mst,
        "weights_applied":         {d: weights[d] for d in PRIMARY_DIMS},
        "dimension_contributions": dim_contributions,
        "primary_contribution":    round(primary_weighted_sum, 2),
        "residual_contribution":   round(residual_contribution, 2),
        "used_role3":              use_role3 and role3_result is not None,
    }


# ============================================================
# 温度缩放（Temperature Scaling）
# ============================================================

def temperature_scale(raw_prob: float, T: float = 1.3) -> float:
    """
    校准概率估计。T > 1 → 更保守，T < 1 → 更激进，T = 1 → 不变。
    原理：Guo et al. (2017) 'On Calibration of Modern Neural Networks'
    """
    raw_prob = max(1e-6, min(1 - 1e-6, raw_prob))
    logit    = math.log(raw_prob / (1 - raw_prob))
    scaled   = logit / T
    return round(1 / (1 + math.exp(-scaled)), 4)


def fit_temperature(probs: list, labels: list) -> float:
    """在校准集上拟合最优温度参数 T。"""
    best_T   = 1.0
    best_ece = float("inf")
    for T in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        scaled = [temperature_scale(p, T) for p in probs]
        ece    = compute_ece(scaled, labels)
        if ece < best_ece:
            best_ece = ece
            best_T   = T
    return best_T


# ============================================================
# AUC + ECE（论文级评估指标）
# ============================================================

def compute_auc(scores: list, labels: list) -> float:
    """
    计算 AUC（判别力）。
    0.5=随机 | 0.7=有效信号 | 0.8+=强信号可发表
    """
    if len(set(labels)) < 2:
        return 0.5

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    concordant   = 0
    total_pairs  = 0
    for i, (s_i, l_i) in enumerate(zip(scores, labels)):
        for j, (s_j, l_j) in enumerate(zip(scores, labels)):
            if l_i == 1 and l_j == 0:
                total_pairs += 1
                if s_i > s_j:
                    concordant += 1
                elif s_i == s_j:
                    concordant += 0.5

    return round(concordant / total_pairs, 4) if total_pairs > 0 else 0.5


def compute_ece(probs: list, labels: list, n_bins: int = 5) -> float:
    """
    计算 ECE（期望校准误差）。ECE 越低越好，0=完美校准。
    """
    if not probs:
        return 1.0

    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    ece = 0.0
    n   = len(probs)
    for b in bins:
        if b:
            avg_p = sum(x[0] for x in b) / len(b)
            avg_y = sum(x[1] for x in b) / len(b)
            ece  += (len(b) / n) * abs(avg_p - avg_y)

    return round(ece, 4)


def evaluate_population(markets: list) -> dict:
    """
    对整个参考人口计算 AUC 和 ECE。

    v2.1 新增：naive_ece（全预测正例的朴素基线）。
    若 ece_before_scaling > naive_ece，说明概率校准差于朴素模型，
    绝对概率值不可信，仅 AUC 排序有参考意义。
    """
    scores, labels, probs = [], [], []

    for m in markets:
        score = m.get("mqr_score")
        t5    = m.get("t5_outcome", {})
        scale = t5.get("achieved_scale")

        if score is None or scale is None:
            continue

        scores.append(float(score))
        labels.append(1 if scale else 0)

        logit = -2.0 + 0.04 * float(score)
        p     = 1 / (1 + math.exp(-max(-500, min(500, logit))))
        probs.append(p)

    if not scores:
        return {"auc": None, "ece": None, "n": 0, "note": "没有足够数据"}

    auc = compute_auc(scores, labels)
    ece = compute_ece(probs, labels)

    # 温度缩放
    T_opt     = fit_temperature(probs, labels)
    scaled_p  = [temperature_scale(p, T_opt) for p in probs]
    ece_after = compute_ece(scaled_p, labels)

    # v2.1 新增：朴素基线（全部预测为正例的最简单模型）
    pos_rate  = sum(labels) / len(labels) if labels else 0.5
    naive_probs = [pos_rate] * len(labels)
    naive_ece   = compute_ece(naive_probs, labels)

    calibration_valid = ece < naive_ece

    return {
        "n":                  len(scores),
        "n_positive":         sum(labels),
        "n_negative":         len(labels) - sum(labels),
        "positive_rate":      round(pos_rate, 3),
        "auc":                auc,
        "ece_before_scaling": ece,
        "ece_after_scaling":  ece_after,
        "optimal_T":          T_opt,
        "ece_improvement":    round(ece - ece_after, 4),
        # --- v2.1 新增 ---
        "naive_ece":          round(naive_ece, 4),
        "ece_vs_naive":       round(ece - naive_ece, 4),   # 负数=优于基线，正数=差于基线
        "calibration_valid":  calibration_valid,
        "calibration_note": (
            "ECE 优于朴素基线，概率估计有参考价值" if calibration_valid
            else f"ECE({ece})高于朴素基线({naive_ece:.4f})，标签失衡({pos_rate:.0%}正例)，"
                 "绝对概率值不可信，仅 AUC 排序有意义"
        ),
        "interpretation": {
            "auc":  "0.5=随机 | 0.7=有效 | 0.8+=强（可发表）",
            "ece":  "越低越好，0=完美校准，需与 naive_ece 对比",
        },
    }


# ============================================================
# 加权余弦最近邻
# ============================================================

def weighted_cosine_similarity(vec_a: list, vec_b: list, weights: list) -> float:
    wa    = [a * w for a, w in zip(vec_a, weights)]
    wb    = [b * w for b, w in zip(vec_b, weights)]
    dot   = sum(x * y for x, y in zip(wa, wb))
    na    = sum(x ** 2 for x in wa) ** 0.5
    nb    = sum(x ** 2 for x in wb) ** 0.5
    denom = na * nb
    return round(dot / denom, 4) if denom > 1e-9 else 0.0


def find_nearest_neighbours(
    target_scores: dict,
    market_structure_type: str,
    reference_population: list,
    k: int = 3,
) -> list:
    """在参考人口中找最相似的 k 个市场（加权余弦）。"""
    weights_map = CAUSAL_WEIGHTS.get(market_structure_type, CAUSAL_WEIGHTS["technology_enablement"])
    raw_w   = [weights_map.get(d, RESIDUAL_WEIGHT / len(RESIDUAL_DIMS)) for d in DIMENSIONS]
    total_w = sum(raw_w)
    weights = [w / total_w for w in raw_w]

    target_vec = [target_scores.get(d, {}).get("adjusted_score", 50) / 100.0 for d in DIMENSIONS]

    similarities = []
    for m in reference_population:
        dim_s = m.get("scoring", {}).get("dimension_scores", {})
        if not dim_s:
            continue
        ref_vec = [dim_s.get(d, {}).get("adjusted_score", 50) / 100.0 for d in DIMENSIONS]
        sim     = weighted_cosine_similarity(target_vec, ref_vec, weights)
        t5      = m.get("t5_outcome", {})
        similarities.append({
            "market_name":     m.get("market_name", "?"),
            "ref_year":        m.get("ref_year", "?"),
            "mqr_rating":      m.get("mqr_rating", "?"),
            "mqr_score":       m.get("mqr_score", 0),
            "achieved_scale":  t5.get("achieved_scale"),
            "outcome_summary": t5.get("outcome_summary", "")[:120],
            "similarity":      sim,
            "structure_type":  m.get("scoring", {}).get("composite", {}).get("market_structure_type", "?"),
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]


# ============================================================
# Outlook Modifier
# ============================================================

def compute_outlook(dimension_scores: dict, step2_result: dict) -> str:
    """
    根据 agreement 分布和低分维度动态决定展望。
    ▲ Positive：多数维度 HIGH agreement，无低分维度
    → Stable：  正常情况
    ▼ Negative：有 LOW agreement 或多个低分维度
    """
    summary    = step2_result.get("_summary", {})
    low_count  = summary.get("low", 0)
    high_count = summary.get("high", 0)

    low_score_dims = [
        d for d in DIMENSIONS
        if dimension_scores.get(d, {}).get("adjusted_score", 50) < 35
    ]

    if low_count >= 2 or len(low_score_dims) >= 2:
        return "▼ Negative"
    elif low_count == 0 and high_count >= 5 and not low_score_dims:
        return "▲ Positive"
    else:
        return "→ Stable"


# ============================================================
# 评级函数
# ============================================================

def assign_rating(composite_score: float, outlook: str = "→ Stable") -> dict:
    """
    v2.1 新增 near_boundary 标记：
    距档位下边界不足 RATING_BUFFER 分时标记，提示结果不稳定。
    """
    rating = "L1"
    for level in ["L5", "L4", "L3", "L2", "L1"]:
        if composite_score >= RATING_THRESHOLDS[level]["min"]:
            rating = level
            break

    threshold     = RATING_THRESHOLDS[rating]["min"]
    near_boundary = (composite_score - threshold) < RATING_BUFFER

    info = RATING_THRESHOLDS[rating]
    return {
        "rating":        rating,
        "label":         info["label"],
        "probability":   info["prob"],
        "outlook":       outlook,
        "near_boundary": near_boundary,   # True = 距下档位不足 3 分，结果不稳定
    }


# ============================================================
# 分析师标记
# ============================================================

def generate_analyst_flags(dimension_scores: dict, composite_result: dict) -> list:
    flags = []
    for dim in DIMENSIONS:
        d   = dimension_scores[dim]
        adj = d["adjusted_score"]
        agr = d["agreement"]

        if agr == "LOW":
            flags.append({
                "priority":  "HIGH",
                "dimension": dim,
                "issue":     f"Role 1 ({d['role1_classification']}) 与 Gemini ({d['gemini_classification']}) 分类严重分歧",
                "action":    f"需要人工核实 {dim} 的正确分类",
            })
        elif adj < 30:
            flags.append({
                "priority":  "HIGH",
                "dimension": dim,
                "issue":     f"维度分数极低（{adj:.0f}/100），是主要风险点",
                "action":    "评估是否有具体的缓解因素",
            })
        elif adj < 45 and agr == "MEDIUM":
            flags.append({
                "priority":  "MEDIUM",
                "dimension": dim,
                "issue":     f"分数偏低（{adj:.0f}/100）且验证一致性中等",
                "action":    f"建议进行主动调研以确认 {dim} 状态",
            })
    return flags


# ============================================================
# 主流程（支持 Role 3）
# ============================================================

def run_scoring_pipeline(
    step1_result: dict,
    step2_result: dict,
    openai_client=None,
    use_role3: bool = True,
    reference_population: list = None,
) -> dict:
    """
    完整评分流水线。

    参数：
        step1_result:         Role 1 输出
        step2_result:         Role 2 输出
        openai_client:        OpenAI 客户端（用于 Role 3）
        use_role3:            是否启用 Role 3 匿名评分
        reference_population: 如果提供，调用 Step 4 百分位评级覆盖阈值评级
    """
    # Step 3A: 固定映射分数
    dim_scores = compute_dimension_scores(step1_result, step2_result)

    # Step 3B: Role 3 匿名评分
    role3_result = None
    if use_role3 and openai_client is not None:
        print("  [Role 3] 匿名特征矩阵评分...", end=" ", flush=True)
        role3_result = score_with_role3(
            dim_scores,
            dim_scores.get("market_structure_type", "technology_enablement"),
            openai_client,
        )
        if role3_result:
            deltas    = [abs(role3_result[d]["delta"]) for d in DIMENSIONS if d in role3_result]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            print(f"✓  平均偏差: {avg_delta:.1f}分")

    # Step 4: 复合分
    composite = compute_composite_score(
        dim_scores,
        use_role3=use_role3 and role3_result is not None,
        role3_result=role3_result,
    )

    # Outlook
    outlook = compute_outlook(dim_scores, step2_result)

    # 评级（阈值法）
    rating = assign_rating(composite["composite_score"], outlook)

    # 如果提供参考人口，用百分位法覆盖评级
    if reference_population:
        from pipeline_step4_rating import rate_with_population
        step4 = rate_with_population(
            composite["composite_score"],
            composite["market_structure_type"],
            dim_scores,
            reference_population,
        )
        rating["rating"]  = step4["rating"]
        rating["step4"]   = step4
        print(f"  [Step 4] 百分位评级: {step4['rating']} (P{step4['percentile']})")

    # 分析师标记
    flags = generate_analyst_flags(dim_scores, composite)

    result = {
        "dimension_scores": dim_scores,
        "composite":        composite,
        "rating":           rating,
        "analyst_flags":    flags,
    }

    if role3_result:
        result["role3_result"] = role3_result

    return result


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    mock_step1 = {
        "timing":             {"classification": "early_chasm",    "confidence": "medium", "rationale": "test"},
        "competition":        {"classification": "consolidating",  "confidence": "high",   "rationale": "test"},
        "market_size":        {"classification": "small",          "confidence": "high",   "rationale": "test"},
        "customer_readiness": {"classification": "adopting",       "confidence": "high",   "rationale": "test"},
        "regulatory":         {"classification": "moderate",       "confidence": "high",   "rationale": "test"},
        "infrastructure":     {"classification": "mature",         "confidence": "high",   "rationale": "test"},
        "market_structure":   {"classification": "defined",        "confidence": "high",   "rationale": "test"},
        "market_structure_type": {"classification": "technology_enablement", "confidence": "high", "rationale": "test"},
    }
    mock_step2 = {
        "timing":             {"role1_classification": "early_chasm",  "gemini_classification": "early_majority", "agreement": "MEDIUM", "confidence": "high", "evidence": "test", "key_fact": "test"},
        "competition":        {"role1_classification": "consolidating","gemini_classification": "consolidating",  "agreement": "HIGH",   "confidence": "high", "evidence": "test", "key_fact": "test"},
        "market_size":        {"role1_classification": "small",        "gemini_classification": "small",         "agreement": "HIGH",   "confidence": "high", "evidence": "test", "key_fact": "test"},
        "customer_readiness": {"role1_classification": "adopting",     "gemini_classification": "adopting",      "agreement": "HIGH",   "confidence": "high", "evidence": "test", "key_fact": "test"},
        "regulatory":         {"role1_classification": "moderate",     "gemini_classification": "heavy",         "agreement": "MEDIUM", "confidence": "high", "evidence": "test", "key_fact": "test"},
        "infrastructure":     {"role1_classification": "mature",       "gemini_classification": "mature",        "agreement": "HIGH",   "confidence": "high", "evidence": "test", "key_fact": "test"},
        "market_structure":   {"role1_classification": "defined",      "gemini_classification": "defined",       "agreement": "HIGH",   "confidence": "high", "evidence": "test", "key_fact": "test"},
        "market_structure_type": {"role1_classification": "technology_enablement", "gemini_classification": "technology_enablement", "agreement": "HIGH", "confidence": "high", "evidence": "test", "key_fact": "test"},
        "_summary": {"overall_agreement": "HIGH", "high": 5, "medium": 2, "low": 0, "total": 7},
    }

    print("🧪 测试（不调用 Role 3，节省 API）...")
    result = run_scoring_pipeline(mock_step1, mock_step2, use_role3=False)

    print(f"\n📊 复合分: {result['composite']['composite_score']}")
    print(f"📌 评级: {result['rating']['rating']} — {result['rating']['label']}")
    print(f"⚠  压线: {result['rating']['near_boundary']}")
    print(f"🔮 展望: {result['rating']['outlook']}")
    print(f"⚑  标记: {len(result['analyst_flags'])} 个")
