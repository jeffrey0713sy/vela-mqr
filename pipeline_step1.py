"""
Vela MQR — pipeline_step1.py
Role 1: GPT-4o 结构化分类提取

核心原则：
- 输出分类词（如 pre_chasm），不输出分数
- 每个维度用受限选项（forced choice），减少模型偏见
- 同时识别市场结构类型（决定后续权重）
- 去掉市场名称对评分的影响（在 Step 3 之后才用）
"""

import json
import os
import re
import time
from openai import OpenAI
from config import DIMENSIONS, MARKET_STRUCTURE_TYPES

# ============================================================
# 每个维度的提取 prompt 模板（受限选项）
# ============================================================

DIMENSION_PROMPTS = {
    "timing": {
        "question": "这个市场目前处于技术采用生命周期的哪个阶段？",
        "options": ["pre_chasm", "early_chasm", "early_majority", "late_majority", "peak", "decline"],
        "guide": (
            "pre_chasm: 创新者阶段，<2% 采用率，无主流买家\n"
            "early_chasm: 早期采用者阶段，有远见者在试用，尚未跨越鸿沟\n"
            "early_majority: 务实派开始购买，8-35% 采用率，最佳入场点\n"
            "late_majority: 35-85% 采用率，增长放缓\n"
            "peak: 接近饱和，主要靠替换需求\n"
            "decline: 市场萎缩"
        ),
    },
    "competition": {
        "question": "这个市场的竞争格局是什么？",
        "options": ["nascent", "fragmented", "consolidating", "concentrated", "monopoly"],
        "guide": (
            "nascent: 1-3 个玩家，类别尚未被分析师命名\n"
            "fragmented: 多个玩家，无人主导，市场份额分散\n"
            "consolidating: M&A 活跃，大玩家开始吞并小玩家\n"
            "concentrated: 2-3 家主导 >60% 市场\n"
            "monopoly: 单一玩家主导 >80%"
        ),
    },
    "market_size": {
        "question": "这个具体产品类别（不是被颠覆的大行业）的当前年收入规模是？",
        "options": ["micro", "small", "medium", "large", "massive"],
        "guide": (
            "micro: <$100M 年收入\n"
            "small: $100M-$1B\n"
            "medium: $1B-$10B\n"
            "large: $10B-$100B\n"
            "massive: >$100B"
        ),
    },
    "customer_readiness": {
        "question": "目标客户群体（最先进的那批）处于购买旅程的哪个阶段？",
        "options": ["unaware", "aware", "interested", "ready", "adopting", "adopted"],
        "guide": (
            "unaware: 不知道有这个问题或解决方案\n"
            "aware: 知道问题，不知道新解决方案\n"
            "interested: 在评估，还没有预算\n"
            "ready: 有预算，在做决策\n"
            "adopting: 已有人在购买和使用\n"
            "adopted: 主流已采用，增长来自替换"
        ),
    },
    "regulatory": {
        "question": "监管环境对新入者的影响程度？",
        "options": ["unregulated", "light_touch", "moderate", "heavy", "prohibited"],
        "guide": (
            "unregulated: 无专门监管，适用一般法律\n"
            "light_touch: 有相邻行业规则松散适用，无专项执法\n"
            "moderate: 有明确合规要求，成本可管理\n"
            "heavy: 严格监管，需要牌照或大量合规投入\n"
            "prohibited: 法律明确禁止或面临即时关停风险"
        ),
    },
    "infrastructure": {
        "question": "支撑这个市场的关键基础设施（技术、分发、支付等）成熟度？",
        "options": ["undefined", "emerging", "developing", "mature", "commoditized"],
        "guide": (
            "undefined: 关键使能技术尚不存在\n"
            "emerging: 技术刚出现，不稳定，成本高\n"
            "developing: 功能可用，成本曲线下降中\n"
            "mature: 可靠、可负担，多个供应商\n"
            "commoditized: 商品化，成本接近零，完全可替换"
        ),
    },
    "market_structure": {
        "question": "这个市场的价值链结构清晰度？",
        "options": ["undefined", "emerging", "forming", "defined", "mature"],
        "guide": (
            "undefined: 没有公认的类别名称，价值链角色不清\n"
            "emerging: 先驱者在定义规则，1-2 个玩家\n"
            "forming: 价值链开始清晰，有分析师关注\n"
            "defined: Gartner/Forrester 有覆盖，价值链成熟\n"
            "mature: 完整生态系统，认证体系，行业协会"
        ),
    },
}

STRUCTURE_TYPE_PROMPT = {
    "question": "这个市场最符合哪种竞争动态类型？",
    "options": MARKET_STRUCTURE_TYPES,
    "guide": (
        "winner_take_most: 网络效应或规模经济驱动 1-2 家主导\n"
        "platform_two_sided: 连接两类用户群体的平台\n"
        "technology_enablement: 新技术创造或解锁的市场\n"
        "fragmented_niche: 地域/垂直/偏好差异支持多个玩家\n"
        "regulated_infrastructure: 监管门槛、高切换成本或基础设施地位"
    ),
}


# ============================================================
# 提取函数
# ============================================================

def _build_extraction_prompt(market_text: str) -> str:
    """构建完整的结构化提取 prompt。"""
    
    dim_sections = []
    for dim in DIMENSIONS:
        cfg = DIMENSION_PROMPTS[dim]
        options_str = " | ".join(cfg["options"])
        dim_sections.append(
            f'"{dim}": {{\n'
            f'  // 问题：{cfg["question"]}\n'
            f'  // 选项：{options_str}\n'
            f'  // 参考：\n'
            + "\n".join(f"  //   {line}" for line in cfg["guide"].split("\n")) + "\n"
            f'  "classification": "<从选项中选一个>",\n'
            f'  "confidence": "<high | medium | low>",\n'
            f'  "rationale": "<一句话理由，引用市场描述中的具体事实>"\n'
            f'}}'
        )

    st = STRUCTURE_TYPE_PROMPT
    st_options = " | ".join(st["options"])

    return f"""你是一个市场结构分类专家。
请根据以下市场描述，为每个维度选择最准确的分类词。

重要规则：
1. 只能从给定选项中选择，不得使用其他词
2. 不得输出分数或数字评估
3. rationale 必须引用市场描述中的具体事实
4. 严格输出 JSON，不加任何解释或 markdown

市场描述：
{market_text}

请输出以下 JSON 格式（将尖括号内容替换为实际答案）：
{{
  {chr(10) + "  ,".join(dim_sections)},
  "market_structure_type": {{
    // 问题：{st["question"]}
    // 选项：{st_options}
    // 参考：
{chr(10).join("    // " + line for line in st["guide"].split(chr(10)))}
    "classification": "<从选项中选一个>",
    "confidence": "<high | medium | low>",
    "rationale": "<一句话理由>"
  }}
}}"""


def extract_classifications(
    market_text: str,
    client: OpenAI,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> dict:
    """
    Role 1：调用 GPT-4o 提取结构化分类。
    
    返回格式：
    {
        "timing": {"classification": "pre_chasm", "confidence": "high", "rationale": "..."},
        ...
        "market_structure_type": {"classification": "platform_two_sided", ...}
    }
    """
    prompt = _build_extraction_prompt(market_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            # 清理 markdown
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            # 找 JSON 对象
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"未找到 JSON 对象：{raw[:200]}")
            
            parsed = json.loads(match.group(0))

            # 验证并清理输出
            result = {}
            for dim in DIMENSIONS:
                if dim not in parsed:
                    raise ValueError(f"缺少维度：{dim}")
                entry = parsed[dim]
                cls = str(entry.get("classification", "")).strip().lower()
                valid_options = list(DIMENSION_PROMPTS[dim]["options"])
                if cls not in valid_options:
                    # 尝试模糊匹配
                    cls = next((o for o in valid_options if o in cls), valid_options[0])
                result[dim] = {
                    "classification": cls,
                    "confidence":     str(entry.get("confidence", "medium")).lower(),
                    "rationale":      str(entry.get("rationale", "")),
                }

            # 市场结构类型
            st_entry = parsed.get("market_structure_type", {})
            st_cls   = str(st_entry.get("classification", "")).strip().lower()
            if st_cls not in MARKET_STRUCTURE_TYPES:
                st_cls = "technology_enablement"  # 默认
            result["market_structure_type"] = {
                "classification": st_cls,
                "confidence":     str(st_entry.get("confidence", "medium")).lower(),
                "rationale":      str(st_entry.get("rationale", "")),
            }

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  [Role 1 重试 {attempt+1}/{max_retries}，{wait}s 后重试] {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Role 1 提取失败：{e}")


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_market = """
    市场：AI 原生法律研究工具（美国中型律所）
    背景：2025 年，Harvey AI 和 Casetext（已被 Thomson Reuters 以 6.5 亿美元收购）
    开始向拥有 50-200 名律师的律所销售 AI 法律研究工具。
    客户：管理合伙人，助理每周花 30-40% 时间在手动检索判例法。
    参考年：2025
    """

    print("🔍 Role 1 提取中...")
    result = extract_classifications(test_market, client)
    print(json.dumps(result, ensure_ascii=False, indent=2))
