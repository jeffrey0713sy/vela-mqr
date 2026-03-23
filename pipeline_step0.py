"""
Vela MQR — pipeline_step0.py
Role 0: Gemini + Google Search 生成历史市场种子

目标：
- 生成 2005-2019 年间的真实历史市场描述
- 每个市场必须有可验证的 T+5 结果（有没有规模化）
- 输出结构化 JSON，供 Step 1-4 处理

为什么用 2005-2019？
- T+5 结果完全可观测（最晚到 2024 年）
- 足够多样：云计算、移动、SaaS、AI 初期等多个浪潮

市场结构类型分布目标（350 个）：
- winner_take_most:       ~70 个 (20%)
- platform_two_sided:     ~70 个 (20%)
- technology_enablement:  ~90 个 (25%)
- fragmented_niche:       ~70 个 (20%)
- regulated_infrastructure: ~50 个 (15%)
"""

import json
import os
import re
import time
import random
from google import genai
from google.genai import types
from config import MARKET_STRUCTURE_TYPES


# ============================================================
# 市场种子主题库（确保多样性）
# ============================================================

MARKET_SEEDS = {
    "winner_take_most": [
        "US social networking platforms (2005-2008)",
        "US smartphone app stores (2008-2012)",
        "US cloud infrastructure IaaS (2006-2010)",
        "US online video streaming (2007-2012)",
        "US ride-hailing / TNC (2010-2014)",
        "US food delivery apps (2012-2016)",
        "US short-term home rental marketplace (2008-2013)",
        "US B2B cloud CRM software (2005-2009)",
        "US online search advertising (2005-2009)",
        "US e-commerce marketplace platforms (2005-2010)",
        "US cloud-based communication / UCaaS (2010-2015)",
        "US online gaming platforms (2005-2010)",
        "US digital music streaming (2008-2013)",
        "US online job recruitment platforms (2005-2010)",
        "US B2B cloud ERP for SMBs (2010-2015)",
    ],
    "platform_two_sided": [
        "US online peer-to-peer lending (2006-2010)",
        "US freelance marketplace platforms (2008-2013)",
        "US online real estate listing platforms (2005-2010)",
        "US local services marketplace (2010-2015)",
        "US online education / MOOC platforms (2011-2016)",
        "US B2B procurement marketplace (2008-2013)",
        "US online insurance comparison (2007-2012)",
        "US gig economy task platforms (2011-2016)",
        "US healthcare provider-patient matching (2010-2015)",
        "US B2B logistics / freight matching (2013-2018)",
        "US online legal services marketplace (2010-2015)",
        "US creator / influencer monetization platforms (2013-2018)",
        "US telehealth video consultation (2013-2018)",
        "US online pet care services marketplace (2013-2018)",
        "US on-demand home services platforms (2012-2017)",
    ],
    "technology_enablement": [
        "US cloud file storage and sync / EFSS (2008-2013)",
        "US B2B SaaS marketing automation (2010-2015)",
        "US developer tools cloud-native / DevOps (2012-2017)",
        "US AI / ML model deployment infrastructure (2016-2019)",
        "US IoT device management platforms (2013-2018)",
        "US container orchestration / Kubernetes ecosystem (2015-2019)",
        "US API-first infrastructure / iPaaS (2012-2017)",
        "US autonomous vehicle software stack (2015-2019)",
        "US drone delivery / UAV commercial (2015-2019)",
        "US AR / VR enterprise training (2015-2019)",
        "US blockchain enterprise solutions (2016-2019)",
        "US edge computing infrastructure (2016-2019)",
        "US natural language processing APIs (2012-2017)",
        "US robotic process automation / RPA (2015-2019)",
        "US AI-powered cybersecurity (2014-2019)",
    ],
    "fragmented_niche": [
        "US specialty vertical SaaS for construction (2010-2015)",
        "US legal tech practice management software (2010-2015)",
        "US restaurant management / POS SaaS (2010-2015)",
        "US healthcare revenue cycle management (2008-2013)",
        "US HR tech / applicant tracking systems (2008-2013)",
        "US field service management software (2010-2015)",
        "US e-discovery software (2007-2012)",
        "US niche B2B content marketing tools (2011-2016)",
        "US specialty pharmacy software (2010-2015)",
        "US agri-tech precision farming platforms (2013-2018)",
        "US fitness and wellness SaaS (2012-2017)",
        "US church / nonprofit management software (2010-2015)",
        "US veterinary practice management (2011-2016)",
        "US independent hotel / short-term rental management (2012-2017)",
        "US dental practice management software (2009-2014)",
    ],
    "regulated_infrastructure": [
        "US healthcare data interoperability / FHIR (2015-2019)",
        "US digital banking / neobank (2013-2018)",
        "US online mortgage origination (2010-2015)",
        "US digital wealth management / robo-advisor (2011-2016)",
        "US healthcare AI diagnostics imaging (2015-2019)",
        "US cannabis tech / seed-to-sale compliance (2014-2019)",
        "US energy grid management / smart grid SaaS (2010-2015)",
        "US government digital services / GovTech (2014-2019)",
        "US blockchain / crypto exchange infrastructure (2013-2018)",
        "US prescription digital therapeutics (2015-2019)",
        "US student loan refinancing platforms (2012-2017)",
        "US payments infrastructure / embedded finance (2011-2016)",
        "US online pharmacy / digital health prescriptions (2015-2019)",
        "US autonomous mobile robots logistics (2013-2018)",
        "US enterprise data privacy / GDPR compliance (2016-2019)",
    ],
}


# ============================================================
# 生成单个市场的 prompt
# ============================================================

def _build_seed_prompt(market_topic: str, structure_type: str) -> str:
    """构建市场种子生成 prompt。"""

    # 随机选择参考年（确保 T+5 可观测）
    ref_year = random.choice(list(range(2005, 2020)))

    return f"""你是一个风险投资市场研究专家。请使用 Google Search 搜索真实的历史信息，为以下市场生成一份详细的市场档案。

市场主题：{market_topic}
市场结构类型：{structure_type}
参考年份：{ref_year}

要求：
1. 所有数据必须真实可验证（公司名称、融资金额、市场规模来自真实来源）
2. 只描述参考年份时的状态（不描述之后发生的事）
3. 必须包含具体的玩家、融资数据、市场规模估算
4. T+5 结果必须基于搜索到的真实证据

请搜索后输出以下 JSON（严格格式，不加 markdown）：
{{
  "market_name": "<具体市场名称，英文>",
  "domain": "<市场领域描述>",
  "ref_year": {ref_year},
  "structure_type": "{structure_type}",
  "base_profile": {{
    "context": "<100-200字：{ref_year}年时的市场背景，包括关键技术拐点、投资者情绪>",
    "buyers": "<50-100字：主要买家群体、核心痛点、购买行为>",
    "players": "<50-100字：主要玩家、融资情况、市场份额>",
    "key_metrics": "<具体数字：市场规模、增速、融资总额等，必须有来源>",
    "exclusions": "<明确说明在参考年份时哪些事情还没发生>"
  }},
  "t5_outcome": {{
    "t5_year": {ref_year + 5},
    "achieved_scale": <true 或 false>,
    "outcome_summary": "<50-100字：T+5时市场发生了什么，为什么成功或失败>",
    "evidence": "<具体证据：公司估值、IPO、收购金额、或倒闭信息>"
  }}
}}"""


# ============================================================
# 生成函数
# ============================================================

def generate_market_seed(
    market_topic: str,
    structure_type: str,
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
) -> dict | None:
    """
    生成单个市场种子。
    返回结构化 dict，失败返回 None。
    """
    prompt = _build_seed_prompt(market_topic, structure_type)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.3,  # 稍微有点随机性，避免重复
                ),
            )
            raw = response.text or ""

            # 清理并解析
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"未找到 JSON: {raw[:200]}")

            data = json.loads(match.group(0))

            # 基本验证
            required = ["market_name", "domain", "ref_year", "base_profile", "t5_outcome"]
            for key in required:
                if key not in data:
                    raise ValueError(f"缺少字段: {key}")

            # 验证 T+5 结果有实质内容
            t5 = data["t5_outcome"]
            if not t5.get("evidence") or len(t5.get("evidence", "")) < 20:
                raise ValueError("T+5 证据不足")

            # 确保 ref_year 合理
            ref_year = int(data.get("ref_year", 2010))
            if not (2005 <= ref_year <= 2019):
                data["ref_year"] = random.randint(2005, 2019)

            data["structure_type"] = structure_type
            data["generation_status"] = "success"
            return data

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"    [JSON 解析失败，重试 {attempt+1}] {e}")
                time.sleep(5)
            else:
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [生成失败，重试 {attempt+1}] {e}")
                time.sleep(10 * (attempt + 1))
            else:
                print(f"    [放弃] {market_topic}: {e}")
                return None

    return None


def generate_batch(
    target_count: int,
    client: genai.Client,
    existing_markets: list | None = None,
    delay: float = 2.0,
) -> list:
    """
    批量生成市场种子，按结构类型分配配额。

    参数：
        target_count: 目标总数（建议 120 起步，最终 350）
        existing_markets: 已有市场列表（用于断点续传）
        delay: 每次请求间隔（秒）

    返回：生成的市场列表
    """
    existing = existing_markets or []
    existing_names = {m.get("market_name", "").lower() for m in existing}

    # 计算各类型配额
    quotas = {
        "winner_take_most":        int(target_count * 0.20),
        "platform_two_sided":      int(target_count * 0.20),
        "technology_enablement":   int(target_count * 0.25),
        "fragmented_niche":        int(target_count * 0.20),
        "regulated_infrastructure": int(target_count * 0.15),
    }

    # 统计已有各类型数量
    existing_counts = {st: 0 for st in MARKET_STRUCTURE_TYPES}
    for m in existing:
        st = m.get("structure_type", "")
        if st in existing_counts:
            existing_counts[st] += 1

    results = list(existing)
    total_generated = 0

    for structure_type, quota in quotas.items():
        remaining = quota - existing_counts.get(structure_type, 0)
        if remaining <= 0:
            print(f"  [{structure_type}] 已满足配额 ({quota}个)，跳过")
            continue

        print(f"\n  [{structure_type}] 需要生成 {remaining} 个市场...")

        # 打乱主题列表，增加多样性
        topics = MARKET_SEEDS.get(structure_type, []).copy()
        random.shuffle(topics)

        # 如果主题不够，循环使用（但参考年份会不同）
        topic_cycle = topics * (remaining // len(topics) + 2)

        generated_this_type = 0
        topic_idx = 0

        while generated_this_type < remaining and topic_idx < len(topic_cycle):
            topic = topic_cycle[topic_idx]
            topic_idx += 1

            print(f"    生成: {topic[:60]}...", end=" ", flush=True)

            market = generate_market_seed(topic, structure_type, client)

            if market is None:
                print("❌ 跳过")
                continue

            # 去重检查
            name = market.get("market_name", "").lower()
            if name in existing_names:
                print(f"⚠ 重复，跳过")
                continue

            existing_names.add(name)
            market["id"] = f"market_{len(results) + 1:03d}"
            results.append(market)
            generated_this_type += 1
            total_generated += 1

            print(f"✓ ({market['market_name'][:40]}, {market['ref_year']})")
            time.sleep(delay)

        print(f"  [{structure_type}] 完成: {generated_this_type} 个")

    print(f"\n  总计生成: {total_generated} 个新市场，累计: {len(results)} 个")
    return results


# ============================================================
# 快速测试（生成 3 个市场）
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    print("🌱 Role 0 测试：生成 3 个市场种子...\n")

    test_topics = [
        ("US cloud file storage and sync / EFSS (2008-2013)", "technology_enablement"),
        ("US ride-hailing / TNC (2010-2014)", "platform_two_sided"),
        ("US digital banking / neobank (2013-2018)", "regulated_infrastructure"),
    ]

    results = []
    for topic, stype in test_topics:
        print(f"生成: {topic}")
        market = generate_market_seed(topic, stype, client)
        if market:
            results.append(market)
            print(f"  ✓ {market['market_name']} ({market['ref_year']})")
            print(f"  T+5: {'✅ 规模化' if market['t5_outcome']['achieved_scale'] else '❌ 未规模化'}")
            print(f"  证据: {market['t5_outcome']['evidence'][:80]}...")
        print()

    print(f"\n生成成功: {len(results)}/3")
    if results:
        print(json.dumps(results[0], ensure_ascii=False, indent=2)[:500] + "...")
