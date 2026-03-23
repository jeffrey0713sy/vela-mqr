"""
Vela MQR — config.py
核心配置：分类映射表、因果权重、评级阈值

设计原则：Step 3 的分数由固定规则表决定，不依赖 AI 随机性。
这确保结果完全可复现，AUC 稳定。

v2.1 变更：
- competition 从 PRIMARY_DIMS 移至 RESIDUAL_DIMS
  原因：消融实验显示 competition 单维度 AUC=0.2301（反向相关），
  保留在主权重中会拖累整体判别力。
  competition 仍保留在系统中（占残余权重 6%），但不影响主权重计算。
- CAUSAL_WEIGHTS 移除 competition 键（原已为 0.00，现显式清理）
- RESIDUAL_DIMS 从 4 个扩展至 5 个，每个残余维度占 6%
"""

# ============================================================
# 1. 七个维度的分类词 → 数值映射（固定规则，不依赖 AI）
# ============================================================

CLASSIFICATION_SCORES = {

    # Timing: 越接近 early_majority 入口越好
    "timing": {
        "pre_chasm":      15,   # 太早，市场未形成
        "early_chasm":    40,   # 开始跨越，有机会
        "early_majority": 75,   # 最佳入场点
        "late_majority":  55,   # 已过黄金期，但仍可行
        "peak":           30,   # 增长见顶，风险高
        "decline":        10,   # 市场萎缩
    },

    # Competition: 对新入者而言竞争越少越好
    # 注意：单维度预测力弱（AUC=0.2301），已移至残余维度
    "competition": {
        "nascent":       85,   # 几乎无竞争，先发优势
        "fragmented":    65,   # 分散，可占据细分
        "consolidating": 40,   # 整合期，大玩家吃小玩家
        "concentrated":  25,   # 2-3 家主导，难以切入
        "monopoly":      10,   # 单一主导，几乎无机会
    },

    # Market Size: 越大越好，但要锚定真实类别收入
    "market_size": {
        "micro":   20,   # <$100M
        "small":   45,   # $100M-$1B
        "medium":  65,   # $1B-$10B
        "large":   80,   # $10B-$100B
        "massive": 95,   # >$100B
    },

    # Customer Readiness: 客户越靠近购买越好
    "customer_readiness": {
        "unaware":    10,
        "aware":      25,
        "interested": 45,
        "ready":      65,
        "adopting":   85,   # 已有人在买，最佳信号
        "adopted":    70,   # 已普及，增长放缓
    },

    # Regulatory: light_touch 最优，两端都不好
    "regulatory": {
        "unregulated":  60,   # 自由但可能突然被监管
        "light_touch":  85,   # 最优：有规则但不阻碍
        "moderate":     55,   # 合规成本存在但可管理
        "heavy":        25,   # 高合规壁垒
        "prohibited":    5,   # 法律禁止
    },

    # Infrastructure: 越成熟越好（降低执行风险）
    "infrastructure": {
        "undefined":    10,
        "emerging":     30,
        "developing":   55,
        "mature":       80,
        "commoditized": 90,   # 基础设施商品化，成本极低
    },

    # Market Structure: 结构越清晰对新入者越有利（可预期）
    "market_structure": {
        "undefined": 20,
        "emerging":  40,
        "forming":   60,
        "defined":   75,
        "mature":    65,   # 成熟但可能固化，略低于 defined
    },
}

# ============================================================
# 2. 市场结构类型识别的候选值
# ============================================================

MARKET_STRUCTURE_TYPES = [
    "winner_take_most",
    "platform_two_sided",
    "technology_enablement",
    "fragmented_niche",
    "regulated_infrastructure",
]

# ============================================================
# 3. 因果权重：按市场结构类型动态调整
#    来源：文档 Part 2.3 + Athey & Wager (2019) CATE 框架
#
#    v2.1：PRIMARY_DIMS = [timing, market_size]，共 2 个维度
#    timing + market_size 权重之和 = 1.0
#    competition 已移至 RESIDUAL_DIMS
# ============================================================

CAUSAL_WEIGHTS = {
    "winner_take_most": {
        "timing":      0.60,
        "market_size": 0.40,
    },
    "platform_two_sided": {
        "timing":      0.65,
        "market_size": 0.35,
    },
    "technology_enablement": {
        "timing":      0.70,
        "market_size": 0.30,
    },
    "fragmented_niche": {
        "timing":      0.30,
        "market_size": 0.70,
    },
    "regulated_infrastructure": {
        "timing":      0.35,
        "market_size": 0.65,
    },
}

# 主权重维度（2个）：占复合分的 70%
PRIMARY_DIMS   = ["timing", "market_size"]
PRIMARY_WEIGHT = 0.70

# 残余维度（5个）：占复合分的 30%，每个约 6%
# v2.1：competition 从 PRIMARY_DIMS 移入此列表
RESIDUAL_DIMS   = ["competition", "customer_readiness", "regulatory", "infrastructure", "market_structure"]
RESIDUAL_WEIGHT = 0.30

# ============================================================
# 4. 置信度乘数（agreement → 置信度调整）
# ============================================================

CONFIDENCE_MULTIPLIER = {
    "HIGH":   1.00,
    "MEDIUM": 0.88,
    "LOW":    0.72,
}

# ============================================================
# 5. L1-L5 评级阈值（基于复合分）
# ============================================================

RATING_THRESHOLDS = {
    "L5": {"min": 80, "label": "EXCEPTIONAL", "prob": "约 17-18/20 的同类市场实现了规模化"},
    "L4": {"min": 65, "label": "STRONG",      "prob": "约 13-14/20 的同类市场实现了规模化"},
    "L3": {"min": 50, "label": "FAVORABLE",   "prob": "约 10-11/20 的同类市场实现了规模化"},
    "L2": {"min": 35, "label": "NEUTRAL",     "prob": "约 7-8/20 的同类市场实现了规模化"},
    "L1": {"min":  0, "label": "SPECULATIVE", "prob": "约 3-4/20 的同类市场实现了规模化"},
}

# 评级缓冲带：距离档位边界不足此值时标记 near_boundary
RATING_BUFFER = 3.0

# ============================================================
# 6. 七个维度列表（有序，供循环使用）
# ============================================================

DIMENSIONS = [
    "timing",
    "competition",
    "market_size",
    "customer_readiness",
    "regulatory",
    "infrastructure",
    "market_structure",
]
