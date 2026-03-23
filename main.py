"""
Vela MQR — main.py
主流水线：输入市场描述 → 输出 L1-L5 评级报告

流程：
  Role 1 (GPT-4o)   → 结构化分类提取
  Role 2 (Gemini)   → 独立验证 + agreement
  Step 3 (Python)   → 固定映射表 → 分数
  Step 4 (Python)   → 因果权重 → 复合分 → L1-L5
"""

import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

import pipeline_step1 as step1
import pipeline_step2 as step2
from pipeline_step3 import run_scoring_pipeline
from config import DIMENSIONS, PRIMARY_WEIGHT, RESIDUAL_DIMS

load_dotenv()


# ============================================================
# 格式化报告
# ============================================================

def format_report(
    market_text: str,
    s1: dict,
    s2: dict,
    scoring: dict,
) -> str:
    """生成可读的文本报告。"""

    rating_info = scoring["rating"]
    composite   = scoring["composite"]
    dim_scores  = scoring["dimension_scores"]
    flags       = scoring["analyst_flags"]
    mst         = composite["market_structure_type"]

    TIER_LABELS = {
        (80, 101): ("L4-L5", "Attractive"),
        (60,  80): ("L3-L4", "Viable"),
        (40,  60): ("L2-L3", "Headwinds"),
        ( 0,  40): ("L1-L2", "Hostile"),
    }

    def score_tier(s):
        for (lo, hi), (_, label) in TIER_LABELS.items():
            if lo <= s < hi:
                return label
        return "Hostile"

    lines = [
        "=" * 62,
        "  VELA 市场质量评级 (MQR)",
        "=" * 62,
        f"  评级日期  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"  评级      : {rating_info['rating']} — {rating_info['label']}",
        f"  复合分    : {composite['composite_score']} / 100",
        f"  展望      : {rating_info['outlook']}",
        f"  概率说明  : {rating_info['probability']}",
        f"  市场结构  : {mst}",
        "",
        "=" * 62,
        "  维度分数",
        "=" * 62,
        f"  {'维度':<20} {'分类':<20} {'得分':>5} {'等级':<12} {'一致性'}",
        f"  {'-'*20} {'-'*20} {'-'*5} {'-'*12} {'-'*6}",
    ]

    for dim in DIMENSIONS:
        d = dim_scores[dim]
        tier = score_tier(d["adjusted_score"])
        agr_symbol = {"HIGH": "✓", "MEDIUM": "~", "LOW": "⚑"}[d["agreement"]]
        lines.append(
            f"  {dim:<20} {d['final_classification']:<20} "
            f"{d['adjusted_score']:>5.1f} {tier:<12} {agr_symbol} {d['agreement']}"
        )

    lines += [
        "",
        "=" * 62,
        "  权重分解（为何这样评分）",
        "=" * 62,
        f"  市场结构类型：{mst}",
        f"  主权重维度（timing/competition/market_size）占 70%",
        f"  具体权重分配：",
    ]
    for dim, w in composite["weights_applied"].items():
        pct = int(w * PRIMARY_WEIGHT * 100)
        contrib = composite["dimension_contributions"][dim]["contribution"]
        lines.append(f"    {dim:<20} {pct:>3}%  贡献: {contrib:.1f} 分")

    lines += [
        f"  残余维度各占 7.5%：",
    ]
    from config import RESIDUAL_DIMS
    for dim in RESIDUAL_DIMS:
        contrib = composite["dimension_contributions"][dim]["contribution"]
        lines.append(f"    {dim:<20}  7.5%  贡献: {contrib:.1f} 分")

    if flags:
        lines += [
            "",
            "=" * 62,
            "  ⚑ 分析师标记 — 需要关注",
            "=" * 62,
        ]
        for flag in flags:
            lines.append(f"  [{flag['priority']}] {flag['dimension']}")
            lines.append(f"    问题：{flag['issue']}")
            lines.append(f"    行动：{flag['action']}")
            lines.append("")

    lines += [
        "=" * 62,
        "  验证来源摘要",
        "=" * 62,
    ]
    summary = s2.get("_summary", {})
    lines.append(
        f"  总体一致性：{summary.get('overall_agreement', '?')}  "
        f"(HIGH:{summary.get('high',0)} MEDIUM:{summary.get('medium',0)} LOW:{summary.get('low',0)})"
    )

    lines.append("=" * 62)
    return "\n".join(lines)


# ============================================================
# 保存结果
# ============================================================

def save_result(market_text: str, s1: dict, s2: dict, scoring: dict) -> str:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rating = scoring["rating"]["rating"]
    out_file = out_dir / f"mqr_{rating}_{ts}.json"

    payload = {
        "pipeline_version": "v2.0_fixed_mapping_causal_weights",
        "created_at":       ts,
        "market_text":      market_text,
        "step1_result":     s1,
        "step2_result":     s2,
        "scoring":          scoring,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(out_file)


# ============================================================
# 主流程
# ============================================================

def run(market_text: str, verbose: bool = True) -> dict:
    """
    完整流水线。

    参数：
        market_text: 市场描述文本
        verbose: 是否打印详细过程

    返回：完整结果字典
    """
    openai_key  = os.getenv("OPENAI_API_KEY", "")
    gemini_key  = os.getenv("GEMINI_API_KEY", "")

    if not openai_key:
        raise ValueError("未找到 OPENAI_API_KEY")
    if not gemini_key:
        raise ValueError("未找到 GEMINI_API_KEY")

    openai_client = OpenAI(api_key=openai_key)
    gemini_client = genai.Client(api_key=gemini_key)

    # ── Role 1: 提取分类 ─────────────────────────────────────
    if verbose:
        print("\n📊 Role 1 (GPT-4o)：结构化分类提取...")
    s1 = step1.extract_classifications(market_text, openai_client)

    if verbose:
        print("\n  提取结果：")
        for dim in DIMENSIONS:
            cls = s1[dim]["classification"]
            conf = s1[dim]["confidence"]
            print(f"    {dim:<22} {cls:<20} [{conf}]")
        mst = s1.get("market_structure_type", {})
        print(f"    {'market_structure_type':<22} {mst.get('classification','?'):<20} [{mst.get('confidence','?')}]")

    # ── Role 2: Gemini 验证 ──────────────────────────────────
    if verbose:
        print("\n🔍 Role 2 (Gemini)：独立验证...")
    s2 = step2.verify_classifications(market_text, s1, gemini_client)

    # ── Step 3 + 4: 评分 + 评级 ──────────────────────────────
    if verbose:
        print("\n🧮 Step 3+4：计算分数和评级...")
    scoring = run_scoring_pipeline(s1, s2, openai_client=openai_client, use_role3=True)

    # ── 报告 ─────────────────────────────────────────────────
    report = format_report(market_text, s1, s2, scoring)

    if verbose:
        print("\n" + report)

    # ── 保存 ─────────────────────────────────────────────────
    out_path = save_result(market_text, s1, s2, scoring)
    if verbose:
        print(f"\n📁 结果已保存：{out_path}")

    return {
        "step1": s1,
        "step2": s2,
        "scoring": scoring,
        "report": report,
        "output_path": out_path,
    }


# ============================================================
# 命令行入口
# ============================================================

def main():
    print("Vela MQR v2.0 — 市场质量评级系统")
    print("=" * 62)
    print("请输入市场描述（输入完成后按两次 Enter）：\n")

    lines = []
    while True:
        line = input()
        if line == "":
            if lines:
                break
        else:
            lines.append(line)

    market_text = "\n".join(lines)
    if not market_text.strip():
        print("❌ 市场描述为空，退出。")
        return

    run(market_text)


if __name__ == "__main__":
    main()
