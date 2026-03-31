"""
Optional zero-shot LLM scores for idea rows (OpenAI-compatible).

Use only on small slices: full 35k rows is costly. Set OPENAI_API_KEY.
"""

from __future__ import annotations

import os
from typing import Iterable


def score_ideas_llm(
    texts: list[str],
    *,
    model: str = "gpt-4o-mini",
    max_chars: int = 1200,
) -> list[float]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    client = OpenAI(api_key=key)
    out: list[float] = []
    for i, raw in enumerate(texts):
        text = (raw or "")[:max_chars].replace("\n", " ")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You score startup ideas for likelihood of becoming a very large "
                        "venture outcome (US VC context). Reply with exactly one number from 0 to 100, "
                        "no words."
                    ),
                },
                {"role": "user", "content": f"Idea:\n{text}"},
            ],
            temperature=0.2,
            max_tokens=8,
        )
        s = (resp.choices[0].message.content or "50").strip()
        try:
            v = float("".join(c for c in s if (c.isdigit() or c == "."))[:8] or "50")
        except ValueError:
            v = 50.0
        v = max(0.0, min(100.0, v))
        out.append(v / 100.0)
        if (i + 1) % 10 == 0:
            print(f"  LLM scored {i + 1}/{len(texts)}")
    return out


def score_ideas_llm_batched(texts: Iterable[str], **kwargs) -> list[float]:
    return score_ideas_llm(list(texts), **kwargs)
