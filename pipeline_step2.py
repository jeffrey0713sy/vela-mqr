"""
Vela MQR — pipeline_step2.py
Role 2: Gemini + 网络搜索 独立验证

核心原则：
- Gemini 独立判断每个维度（不看 Role 1 的结论）
- 输出自己的分类 + 证据 + agreement（与 Role 1 的一致程度）
- HIGH agreement → 置信度高，直接用
- MEDIUM/LOW agreement → 标记为需要审查，保守处理
"""

import json
import os
import re
import time
from google import genai
from google.genai import types
from config import DIMENSIONS, MARKET_STRUCTURE_TYPES
from pipeline_step1 import DIMENSION_PROMPTS

# ============================================================
# Gemini 验证 prompt
# ============================================================

def _build_verification_prompt(dim: str, market_text: str) -> str:
    """为单个维度构建 Gemini 验证 prompt。"""
    cfg = DIMENSION_PROMPTS[dim]
    options_str = " | ".join(cfg["options"])

    return f"""你是一个市场研究分析师。请搜索网络，为以下市场的指定维度找到真实证据，并做出独立判断。

市场描述：
{market_text}

需要判断的维度：{dim}
问题：{cfg["question"]}
可选答案：{options_str}

参考指南：
{cfg["guide"]}

任务：
1. 搜索能支持你判断的具体事实（公司名称、融资金额、市场规模数据、监管文件等）
2. 基于搜索结果独立选择最准确的分类词
3. 不要猜测，只使用你搜索到的证据

严格输出 JSON（不加任何 markdown 或解释）：
{{
  "gemini_classification": "<从选项中选一个>",
  "confidence": "<high | medium | low>",
  "evidence": "<一句话：引用具体来源和事实>",
  "key_fact": "<最重要的一个数据点或事件>"
}}"""


def _build_structure_type_prompt(market_text: str) -> str:
    """验证市场结构类型的 prompt。"""
    options_str = " | ".join(MARKET_STRUCTURE_TYPES)
    return f"""你是市场结构分析师。请搜索网络，判断以下市场属于哪种竞争动态类型。

市场描述：
{market_text}

可选类型：{options_str}

类型说明：
winner_take_most: 网络效应或规模经济，趋向 1-2 家主导
platform_two_sided: 双边平台，连接两类用户群
technology_enablement: 新技术创造的市场
fragmented_niche: 天然支持多个玩家
regulated_infrastructure: 监管壁垒或基础设施地位

严格输出 JSON：
{{
  "gemini_classification": "<从选项中选一个>",
  "confidence": "<high | medium | low>",
  "evidence": "<一句话理由>"
}}"""


# ============================================================
# 验证函数
# ============================================================

def _safe_parse_gemini(raw: str, valid_options: list, fallback: str) -> dict:
    """安全解析 Gemini 输出。"""
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "gemini_classification": fallback,
            "confidence": "low",
            "evidence": f"解析失败: {raw[:100]}",
            "key_fact": "",
        }

    try:
        parsed = json.loads(match.group(0))
        cls = str(parsed.get("gemini_classification", "")).strip().lower()
        if cls not in valid_options:
            cls = next((o for o in valid_options if o in cls), fallback)
        return {
            "gemini_classification": cls,
            "confidence":            str(parsed.get("confidence", "medium")).lower(),
            "evidence":              str(parsed.get("evidence", "")),
            "key_fact":              str(parsed.get("key_fact", "")),
        }
    except json.JSONDecodeError:
        return {
            "gemini_classification": fallback,
            "confidence": "low",
            "evidence": f"JSON 解析错误: {raw[:100]}",
            "key_fact": "",
        }


def _compute_agreement(role1_cls: str, gemini_cls: str) -> str:
    """计算 Role 1 和 Role 2 的一致程度。"""
    if role1_cls == gemini_cls:
        return "HIGH"

    # 定义相邻分类（差一档认为 MEDIUM）
    adjacency = {
        "timing":           ["pre_chasm", "early_chasm", "early_majority", "late_majority", "peak", "decline"],
        "competition":      ["nascent", "fragmented", "consolidating", "concentrated", "monopoly"],
        "market_size":      ["micro", "small", "medium", "large", "massive"],
        "customer_readiness": ["unaware", "aware", "interested", "ready", "adopting", "adopted"],
        "regulatory":       ["prohibited", "heavy", "moderate", "light_touch", "unregulated"],
        "infrastructure":   ["undefined", "emerging", "developing", "mature", "commoditized"],
        "market_structure": ["undefined", "emerging", "forming", "defined", "mature"],
    }

    for dim_order in adjacency.values():
        if role1_cls in dim_order and gemini_cls in dim_order:
            idx1 = dim_order.index(role1_cls)
            idx2 = dim_order.index(gemini_cls)
            if abs(idx1 - idx2) == 1:
                return "MEDIUM"
            else:
                return "LOW"

    return "LOW"


def verify_classifications(
    market_text: str,
    step1_result: dict,
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    delay: float = 1.5,
) -> dict:
    """
    Role 2：Gemini 独立验证每个维度。

    返回格式：
    {
        "timing": {
            "role1_classification": "pre_chasm",
            "gemini_classification": "early_chasm",
            "agreement": "MEDIUM",
            "confidence": "high",
            "evidence": "...",
            "key_fact": "..."
        },
        ...
        "market_structure_type": {...}
    }
    """
    result = {}

    # 验证七个维度
    for dim in DIMENSIONS:
        print(f"  [Role 2] 验证 {dim}...", end=" ", flush=True)

        role1_cls = step1_result.get(dim, {}).get("classification", "unknown")
        valid_options = list(DIMENSION_PROMPTS[dim]["options"])

        prompt = _build_verification_prompt(dim, market_text)

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0,
                ),
            )
            raw = response.text or ""
        except Exception as e:
            print(f"❌ {e}")
            raw = ""

        parsed = _safe_parse_gemini(raw, valid_options, role1_cls)
        agreement = _compute_agreement(role1_cls, parsed["gemini_classification"])

        result[dim] = {
            "role1_classification":   role1_cls,
            "gemini_classification":  parsed["gemini_classification"],
            "agreement":              agreement,
            "confidence":             parsed["confidence"],
            "evidence":               parsed["evidence"],
            "key_fact":               parsed["key_fact"],
        }

        print(f"r1={role1_cls} | g={parsed['gemini_classification']} | {agreement}")
        time.sleep(delay)

    # 验证市场结构类型
    print(f"  [Role 2] 验证 market_structure_type...", end=" ", flush=True)
    role1_st = step1_result.get("market_structure_type", {}).get("classification", "technology_enablement")

    try:
        response = client.models.generate_content(
            model=model,
            contents=_build_structure_type_prompt(market_text),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0,
            ),
        )
        raw_st = response.text or ""
    except Exception as e:
        print(f"❌ {e}")
        raw_st = ""

    parsed_st = _safe_parse_gemini(raw_st, MARKET_STRUCTURE_TYPES, role1_st)
    agreement_st = _compute_agreement(role1_st, parsed_st["gemini_classification"])

    result["market_structure_type"] = {
        "role1_classification":  role1_st,
        "gemini_classification": parsed_st["gemini_classification"],
        "agreement":             agreement_st,
        "confidence":            parsed_st["confidence"],
        "evidence":              parsed_st["evidence"],
        "key_fact":              parsed_st.get("key_fact", ""),
    }
    print(f"r1={role1_st} | g={parsed_st['gemini_classification']} | {agreement_st}")

    # 汇总 agreement
    agreements = [result[d]["agreement"] for d in DIMENSIONS]
    high_count   = agreements.count("HIGH")
    medium_count = agreements.count("MEDIUM")
    low_count    = agreements.count("LOW")
    overall      = "HIGH" if high_count >= 5 else ("MEDIUM" if low_count <= 1 else "LOW")

    result["_summary"] = {
        "overall_agreement": overall,
        "high":   high_count,
        "medium": medium_count,
        "low":    low_count,
        "total":  len(DIMENSIONS),
    }

    return result


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import pipeline_step1 as step1

    openai_client_module = __import__("openai")
    openai_client = openai_client_module.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    test_market = """
    市场：AI 原生法律研究工具（美国中型律所）
    背景：2025 年，Harvey AI 和 Casetext（已被 Thomson Reuters 以 6.5 亿美元收购）
    开始向拥有 50-200 名律师的律所销售 AI 法律研究工具。参考年：2025
    """

    print("Step 1: 提取分类...")
    s1 = step1.extract_classifications(test_market, openai_client)

    print("\nStep 2: Gemini 验证...")
    s2 = verify_classifications(test_market, s1, gemini_client)
    print(json.dumps(s2, ensure_ascii=False, indent=2))
