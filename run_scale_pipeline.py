"""
Vela MQR — run_scale_pipeline.py
批量编排：生成参考人口 + 跑完整评分流水线

功能：
1. Role 0：批量生成历史市场种子
2. 对每个市场跑 Step 1-4
3. 断点续传（崩了可以继续跑）
4. 自动保存中间结果

用法：
    python run_scale_pipeline.py              # 生成 120 个市场
    python run_scale_pipeline.py --target 350 # 生成 350 个市场
    python run_scale_pipeline.py --resume     # 从上次断点继续
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

import pipeline_step0 as step0
import pipeline_step1 as step1
import pipeline_step2 as step2
from pipeline_step3 import run_scoring_pipeline
from config import DIMENSIONS

load_dotenv()

DATA_DIR      = Path("data")
MASTER_FILE   = DATA_DIR / "reference_population_master.json"
PROGRESS_FILE = DATA_DIR / "_progress.json"


# ============================================================
# 存储工具
# ============================================================

def load_master() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    if MASTER_FILE.exists():
        with open(MASTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": "2.0",
        "created_at": datetime.now().isoformat(),
        "markets": [],
    }


def save_master(data: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    data["total_count"] = len(data["markets"])
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_progress(current_idx: int, total: int) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "current": current_idx,
            "total": total,
            "ts": datetime.now().isoformat(),
        }, f)


# ============================================================
# 单个市场完整流水线
# ============================================================

def run_full_pipeline_for_market(
    seed: dict,
    openai_client: OpenAI,
    gemini_client: genai.Client,
    verbose: bool = True,
) -> dict:
    market_name = seed.get("market_name", "Unknown")
    ref_year    = seed.get("ref_year", "?")

    bp = seed.get("base_profile", {})
    market_text = f"""
Market: {seed.get('market_name', '')}

Context: {bp.get('context', '')}

Buyers: {bp.get('buyers', '')}

Players: {bp.get('players', '')}

Key Metrics: {bp.get('key_metrics', '')}

Exclusions: {bp.get('exclusions', '')}

Reference year: {ref_year}
""".strip()

    # Step 1
    if verbose:
        print("    Step 1 (GPT-4o)...", end=" ", flush=True)
    s1 = step1.extract_classifications(market_text, openai_client)
    if verbose:
        mst_cls = s1.get("market_structure_type", {}).get("classification", "?")
        print(f"✓  mst={mst_cls}")

    # Step 2
    if verbose:
        print("    Step 2 (Gemini)...")
    s2 = step2.verify_classifications(market_text, s1, gemini_client, delay=1.0)
    overall_agr = s2.get("_summary", {}).get("overall_agreement", "?")
    if verbose:
        print(f"    overall_agreement={overall_agr}")

    # Step 3+4

    scoring = run_scoring_pipeline(s1, s2, openai_client=openai_client, use_role3=True)
    rating  = scoring["rating"]["rating"]
    score   = scoring["composite"]["composite_score"]

    if verbose:
        print(f"    📊 评级: {rating}  复合分: {score:.1f}")

    return {
        **seed,
        "step1":        s1,
        "step2":        s2,
        "scoring":      scoring,
        "mqr_rating":   rating,
        "mqr_score":    score,
        "processed_at": datetime.now().isoformat(),
    }


# ============================================================
# 参考人口统计
# ============================================================

def compute_population_stats(markets: list) -> dict:
    from collections import defaultdict
    import statistics

    by_type = defaultdict(list)
    for m in markets:
        if "mqr_score" not in m:
            continue
        st = m.get("scoring", {}).get("composite", {}).get("market_structure_type", "unknown")
        by_type[st].append({
            "score":          m["mqr_score"],
            "rating":         m["mqr_rating"],
            "achieved_scale": m.get("t5_outcome", {}).get("achieved_scale"),
        })

    stats = {}
    for st, records in by_type.items():
        scores = [r["score"] for r in records]
        scaled = [r for r in records if r["achieved_scale"] is True]
        stats[st] = {
            "count":        len(records),
            "mean_score":   round(statistics.mean(scores), 2) if scores else 0,
            "scores_sorted": sorted(scores),
            "scale_rate":   round(len(scaled) / len(records), 3) if records else 0,
            "scaled_count": len(scaled),
        }
    return stats


def get_percentile(score: float, population_scores: list) -> float:
    if not population_scores:
        return 50.0
    below = sum(1 for s in population_scores if s < score)
    return round(below / len(population_scores) * 100, 1)


# ============================================================
# 主编排
# ============================================================

def run_scale_pipeline(
    target: int = 120,
    resume: bool = False,
    save_every: int = 5,
    delay_between: float = 3.0,
) -> None:
    print("=" * 62)
    print("  Vela MQR — 规模化流水线")
    print(f"  目标: {target} 个市场  |  模式: {'续传' if resume else '全新'}")
    print("=" * 62)

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

    master = load_master()
    print(f"\n  已有市场: {len(master['markets'])} 个")

    # ── Phase 1: 生成种子 ─────────────────────────────
    existing = master["markets"]
    seeds_needed = target - len(existing)

    if seeds_needed > 0:
        print(f"\n  Phase 1: 生成 {seeds_needed} 个市场种子...")
        new_seeds = step0.generate_batch(
            target_count=seeds_needed,
            client=gemini_client,
            existing_markets=existing,
            delay=2.0,
        )
        for seed in new_seeds:
            if "mqr_rating" not in seed:
                existing.append(seed)
        master["markets"] = existing
        save_master(master)
        print(f"  种子完毕: {len(existing)} 个")
    else:
        print(f"  种子已足够，跳过 Phase 1")

    # ── Phase 2: 逐个评分 ─────────────────────────────
    unscored = [m for m in master["markets"] if "mqr_rating" not in m]
    print(f"\n  Phase 2: 待评分 {len(unscored)} 个市场")

    scored_count = 0
    failed_count = 0

    for i, seed in enumerate(unscored):
        name = seed.get("market_name", f"#{i+1}")
        print(f"\n[{i+1}/{len(unscored)}] {name[:55]}")

        try:
            result = run_full_pipeline_for_market(
                seed, openai_client, gemini_client, verbose=True
            )
            # 更新 master
            for j, m in enumerate(master["markets"]):
                if m.get("market_name") == seed.get("market_name"):
                    master["markets"][j] = result
                    break
            scored_count += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed_count += 1

        if (i + 1) % save_every == 0:
            master["population_stats"] = compute_population_stats(master["markets"])
            save_master(master)
            save_progress(i + 1, len(unscored))
            print(f"  💾 进度保存 ({i+1}/{len(unscored)})")

        time.sleep(delay_between)

    master["population_stats"] = compute_population_stats(master["markets"])
    save_master(master)

    # ── 最终统计 ──────────────────────────────────────
    all_rated = [m for m in master["markets"] if "mqr_rating" in m]
    rating_dist = {}
    for m in all_rated:
        r = m["mqr_rating"]
        rating_dist[r] = rating_dist.get(r, 0) + 1

    print(f"\n{'='*62}")
    print(f"  完成！评分: {scored_count}  失败: {failed_count}")
    for r in ["L5", "L4", "L3", "L2", "L1"]:
        c = rating_dist.get(r, 0)
        print(f"  {r}: {c:3d}  {'█' * (c // 2)}")
    print(f"  数据: {MASTER_FILE}")
    print("=" * 62)


# ============================================================
# 命令行
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vela MQR 规模化流水线")
    parser.add_argument("--target",     type=int, default=120)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--delay",      type=float, default=3.0)
    args = parser.parse_args()

    run_scale_pipeline(
        target=args.target,
        resume=args.resume,
        save_every=args.save_every,
        delay_between=args.delay,
    )
