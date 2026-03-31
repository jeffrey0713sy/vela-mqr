"""
Vela Search API hook for time-bounded web context (Summer 2025 brief).

Set:
  VELA_SEARCH_API_KEY
  VELA_SEARCH_BASE_URL   (e.g. https://csearch.vela.partners)

If unset or request fails, returns empty string so pipelines still run offline.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any


def _founded_to_date(v: Any) -> date | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def fetch_idea_context(
    *,
    name: str,
    short_description: str,
    founded_on: Any,
    timeout_sec: float = 30.0,
) -> str:
    """
    Returns a short text blob to append to the idea description for modeling.
    """
    key = os.environ.get("VELA_SEARCH_API_KEY", "").strip()
    base = os.environ.get("VELA_SEARCH_BASE_URL", "").strip().rstrip("/")
    if not key or not base:
        return ""

    fd = _founded_to_date(founded_on)
    if fd is None:
        return ""

    # Placeholder path — adjust to match live API contract from Vela docs.
    url = f"{base}/v1/search"
    payload = {
        "query": f"{name} startup {short_description[:200]}",
        "after": fd.isoformat(),
        "before": (date(fd.year + 1, 12, 31)).isoformat(),
        "max_results": 8,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    try:
        import requests

        r = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
        if r.status_code >= 400:
            return ""
        data = r.json()
        if isinstance(data, dict) and "summary" in data:
            return str(data["summary"])[:8000]
        if isinstance(data, list):
            parts = []
            for item in data[:8]:
                if isinstance(item, dict):
                    parts.append(str(item.get("snippet") or item.get("text") or ""))
            return "\n".join(p for p in parts if p)[:8000]
        return str(data)[:4000]
    except Exception:
        return ""
