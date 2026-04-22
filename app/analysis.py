"""多轮分析的核心业务逻辑。server.py 和终端测试共用此模块。"""

import json
import re
import logging
from .llm import chat_sync
from .memory import save_memory, load_memory

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5


def extract_json(text: str | None) -> str:
    """Extract JSON from LLM output, handling code blocks and surrounding text."""
    if not text:
        return "{}"
    text = text.strip()

    # Try to find JSON inside a code block: ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try to find the outermost { ... } block
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


def count_rounds() -> int:
    memory = load_memory()
    return len([m for m in memory if m["role"] == "user"])


def start_analysis(client, question: str, model: str = None) -> dict:
    """开始新的分析会话：清空记忆，发送第一条消息，返回解析后的结果。"""
    save_memory([])
    reply = chat_sync(client, question, "mind_lens", model=model)
    data = parse_reply(reply)
    data["round"] = count_rounds()
    return data


def continue_analysis(client, user_message: str, model: str = None) -> dict:
    """继续分析会话：发送用户回复，返回解析后的结果。超过最大轮次强制输出建议。"""
    reply = chat_sync(client, user_message, "mind_lens", model=model)
    data = parse_reply(reply)
    rounds = count_rounds()
    data["round"] = rounds
    if rounds >= MAX_ROUNDS and not data.get("ready_for_suggestion"):
        data["ready_for_suggestion"] = True
        data["follow_up"] = None
    return data
