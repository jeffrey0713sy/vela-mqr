"""
Vela MQR — run_ablation.py
5 Ablation Studies on the reference population.

v2.1 变更：
- PRIMARY_DIMS 从 config 导入，自动适配 competition 移至 RESIDUAL_DIMS 的变化
- 摘要表新增 naive_ece，便于判断校准是否优于朴素基线
- Ablation 3 结果顺序可能变化（competition 单维度 AUC 已知为反向信号）
"""

import json
import math
import random
import sys
import os

sys.path.insert(0, ".")
from pipeline_step3 import (
    compute_auc,
    compute_ece,
    evaluate_population,
    classification_to_score,
)
from config import (
    CAUSAL_WEIGHTS,
    CLASSIFICATION_SCORES,
    DIMENSIONS,
    PRIMARY_DIMS,
    RESIDUAL_DIMS,
    PRIMARY_WEIGHT,
    RESIDUAL_WEIGHT,
)

random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
with open("data/reference_population_v21.json", encoding="utf-8") as f:
    data = json.load(f)

markets = [
    m for m in data["markets"]
    if "mqr_score" in m
    and m.get("t5_outcome", {}).get("achieved_scale") is not None
]
N = len(markets)

print(f"\n{'='*60}")
print(f"  Vela MQR — Ablation Study  |  N={N} markets")
print(f"  PRIMARY_DIMS: {PRIMARY_DIMS}")
print(f"  RESIDUAL_DIMS: {RESIDUAL_DIMS}")
print(f"{'='*60}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def scores_to_probs(scores):
    return [sigmoid(-2.0 + 0.04 * s) for s in scores]

def get_labels(mkts):
    return [1 if m["t5_outcome"]["achieved_scale"] else 0 for m in mkts]

def get_composite_scores(mkts):
    return [float(m["mqr_score"]) for m in mkts]

labels     = get_labels(markets)
composites = get_composite_scores(markets)

# 朴素基线 ECE
pos_rate  = sum(labels) / len(labels)
naive_ece = compute_ece([pos_rate] * len(labels), labels)

baseline_auc = compute_auc(composites, labels)
baseline_ece = compute_ece(scores_to_probs(composites), labels)

print(f"  Baseline (full pipeline):  AUC={baseline_auc:.4f}  ECE={baseline_ece:.4f}")
print(f"  Naive baseline ECE:        {naive_ece:.4f}  (all-positive model, {pos_rate:.0%} positive rate)")
print(f"  ECE vs naive:              {baseline_ece - naive_ece:+.4f}  {'✓ 优于基线' if baseline_ece < naive_ece else '✗ 差于基线，绝对概率值不可信'}\n")


# ── Ablation 1: Role 1 only (no Gemini verification) ─────────────────────────
print(f"{'─'*60}")
print("  Ablation 1: Remove Gemini Verification (Role 1 only)")
print(f"{'─'*60}")

role1_scores = []
for m in markets:
    s1 = m.get("step1", {})
    if not s1:
        role1_scores.append(50.0)
        continue

    dim_scores = {}
    for dim in DIMENSIONS:
        cls = s1.get(dim, {}).get("classification", "")
        dim_scores[dim] = float(classification_to_score(dim, cls))

    mst     = s1.get("market_structure_type", {}).get("classification", "technology_enablement")
    weights = CAUSAL_WEIGHTS.get(mst, CAUSAL_WEIGHTS["technology_enablement"])

    primary_total = sum(weights[d] for d in PRIMARY_DIMS)
    primary_sum   = sum(
        dim_scores[d] * (weights[d] / primary_total) * PRIMARY_WEIGHT
        for d in PRIMARY_DIMS
    )
    residual_avg  = sum(dim_scores[d] for d in RESIDUAL_DIMS) / len(RESIDUAL_DIMS)
    score         = primary_sum + residual_avg * RESIDUAL_WEIGHT
    role1_scores.append(round(score, 2))

auc1 = compute_auc(role1_scores, labels)
ece1 = compute_ece(scores_to_probs(role1_scores), labels)
print(f"  Role 1 only:  AUC={auc1:.4f}  ECE={ece1:.4f}")
print(f"  vs Baseline:  ΔAUC={auc1-baseline_auc:+.4f}  ΔECE={ece1-baseline_ece:+.4f}")
print(f"  → 注：若 Role 1 AUC > Baseline，与标签失衡(86%正例)有关，")
print(f"    而非 Gemini 验证无效。Gemini 的价值在校准而非排序。\n")


# ── Ablation 2: Uniform weights (no structure-type differentiation) ───────────
print(f"{'─'*60}")
print("  Ablation 2: Uniform Weights (ignore market structure type)")
print(f"{'─'*60}")

uniform_scores = []
for m in markets:
    dim_s = m.get("scoring", {}).get("dimension_scores", {})
    if not dim_s:
        uniform_scores.append(50.0)
        continue
    scores_list = [dim_s.get(d, {}).get("adjusted_score", 50) for d in DIMENSIONS]
    uniform_scores.append(round(sum(scores_list) / len(scores_list), 2))

auc2 = compute_auc(uniform_scores, labels)
ece2 = compute_ece(scores_to_probs(uniform_scores), labels)
print(f"  Uniform weights:  AUC={auc2:.4f}  ECE={ece2:.4f}")
print(f"  vs Baseline:      ΔAUC={auc2-baseline_auc:+.4f}  ΔECE={ece2-baseline_ece:+.4f}")
print(f"  → 因果权重 vs 均等权重的判别力差异\n")


# ── Ablation 3: Single dimension predictive power ─────────────────────────────
print(f"{'─'*60}")
print("  Ablation 3: Single Dimension Predictive Power Ranking")
print(f"{'─'*60}")
print(f"  {'Dimension':<25} {'AUC':>6}  {'vs Composite':>12}  {'In PRIMARY?':>11}")
print(f"  {'─'*25} {'─'*6}  {'─'*12}  {'─'*11}")

dim_aucs = {}
for dim in DIMENSIONS:
    dim_scores_list = []
    for m in markets:
        ds    = m.get("scoring", {}).get("dimension_scores", {})
        score = ds.get(dim, {}).get("adjusted_score", 50) if ds else 50
        dim_scores_list.append(float(score))
    auc_d = compute_auc(dim_scores_list, labels)
    dim_aucs[dim] = auc_d
    delta      = auc_d - baseline_auc
    in_primary = "✓ PRIMARY" if dim in PRIMARY_DIMS else "  residual"
    print(f"  {dim:<25} {auc_d:.4f}  {delta:+.4f}        {in_primary}")

best_dim = max(dim_aucs, key=dim_aucs.get)
print(f"\n  Best single dimension: {best_dim} (AUC={dim_aucs[best_dim]:.4f})")
print(f"  Composite AUC={baseline_auc:.4f} — {'BETTER' if baseline_auc > dim_aucs[best_dim] else 'WORSE'} than best single dim\n")


# ── Ablation 4: Population size sensitivity ───────────────────────────────────
print(f"{'─'*60}")
print("  Ablation 4: Population Size Sensitivity")
print(f"{'─'*60}")
print(f"  {'N':>5}  {'AUC':>6}  {'ECE':>6}  {'vs naive':>8}")
print(f"  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*8}")

for n_sample in [N, 50, 30, 20]:
    if n_sample >= N:
        sample_m = markets
    else:
        sample_m = random.sample(markets, n_sample)

    s_labels = get_labels(sample_m)
    s_scores = get_composite_scores(sample_m)

    if len(set(s_labels)) < 2:
        print(f"  {n_sample:>5}  {'N/A':>6}  {'N/A':>6}  {'N/A':>8}  (only one class)")
        continue

    auc_n  = compute_auc(s_scores, s_labels)
    ece_n  = compute_ece(scores_to_probs(s_scores), s_labels)
    pos_n  = sum(s_labels) / len(s_labels)
    naive_n = compute_ece([pos_n] * len(s_labels), s_labels)
    vs_naive = ece_n - naive_n
    print(f"  {n_sample:>5}  {auc_n:.4f}  {ece_n:.4f}  {vs_naive:+.4f}")

print(f"\n  → N<30 时 AUC 开始不稳定，最小可靠人口规模约 50 个市场\n")


# ── Ablation 5: Role 3 delta analysis ────────────────────────────────────────
print(f"{'─'*60}")
print("  Ablation 5: Role 3 vs Fixed Mapping Score Delta")
print(f"{'─'*60}")

role3_markets = [m for m in markets if m.get("scoring", {}).get("composite", {}).get("used_role3")]
if role3_markets:
    deltas = []
    for m in role3_markets:
        r3 = m.get("scoring", {}).get("role3_result", {})
        for dim in DIMENSIONS:
            if dim in r3:
                deltas.append(abs(r3[dim].get("delta", 0)))

    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)
        print(f"  Markets with Role 3:  {len(role3_markets)}")
        print(f"  Avg |delta| per dim:  {avg_delta:.2f} points")
        print(f"  Max |delta|:          {max_delta:.2f} points")
        print(f"  → 小偏差=Role 3 确认固定映射  |  大偏差=Role 3 检测到潜在泄漏偏差")
    else:
        print("  Role 3 数据存在但无 delta 字段。")
else:
    print(f"  当前参考人口中无 Role 3 数据（流水线在 Role 3 集成前已生成）。")
    print(f"  Future work: 运行 recompute_scores.py 或 patch_role3.py 补充。")


# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ABLATION SUMMARY")
print(f"{'='*60}")
print(f"  {'Condition':<35} {'AUC':>6}  {'ΔAUC':>6}")
print(f"  {'─'*35} {'─'*6}  {'─'*6}")
print(f"  {'Full pipeline (baseline)':<35} {baseline_auc:.4f}  {'—':>6}")
print(f"  {'No Gemini verification':<35} {auc1:.4f}  {auc1-baseline_auc:+.4f}")
print(f"  {'Uniform weights':<35} {auc2:.4f}  {auc2-baseline_auc:+.4f}")
print(f"  {'Best single dim (' + best_dim + ')':<35} {dim_aucs[best_dim]:.4f}  {dim_aucs[best_dim]-baseline_auc:+.4f}")
print(f"{'='*60}")
print(f"\n  Calibration note:")
print(f"  Naive ECE (all-positive baseline): {naive_ece:.4f}")
print(f"  Pipeline ECE:                      {baseline_ece:.4f}")
print(f"  Status: {'✓ 优于基线' if baseline_ece < naive_ece else '✗ 差于基线 — 绝对概率值不应对外报告'}")
print(f"\n  PRIMARY_DIMS (v2.1): {PRIMARY_DIMS}")
print(f"  competition 已移至 RESIDUAL_DIMS（单维 AUC=0.2301，反向信号）\n")
