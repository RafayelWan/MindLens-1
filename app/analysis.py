"""多轮分析的核心业务逻辑。"""

from __future__ import annotations

import json
import re
import logging
from .llm import chat_sync
from .session import SessionData

logger = logging.getLogger(__name__)


def extract_json(text: str | None) -> str:
    """Extract JSON from LLM output, handling code blocks and surrounding text."""
    if not text:
        return "{}"
    text = text.strip()

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]

    return text


def parse_reply(reply: str) -> dict:
    """解析 LLM 返回的 JSON。解析失败时返回带 error 字段的 dict。"""
    extracted = extract_json(reply)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM reply as JSON. Raw: %s", reply[:500])
        return {
            "cards": {},
            "follow_up": None,
            "ready_for_suggestion": False,
            "round": 0,
            "error": "LLM 返回了非法 JSON，请重试",
            "raw": reply[:500] if reply else "",
        }


def _count_user_rounds(session: SessionData) -> int:
    return len([m for m in session.get_memory() if m["role"] == "user"])


def start_analysis(session: SessionData, question: str) -> dict:
    """开始新的分析会话：清空记忆，发送第一条消息，返回解析后的结果。"""
    session.clear_memory()
    reply = chat_sync(session, question, "mind_lens")
    data = parse_reply(reply)
    data["round"] = _count_user_rounds(session)
    return data


def continue_analysis(session: SessionData, user_message: str) -> dict:
    """继续分析会话：发送用户回复，返回解析后的结果。由 LLM 自主判断何时给出建议。"""
    reply = chat_sync(session, user_message, "mind_lens")
    data = parse_reply(reply)
    rounds = _count_user_rounds(session)
    data["round"] = rounds
    return data
