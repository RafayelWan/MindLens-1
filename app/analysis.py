"""多轮分析的核心业务逻辑。server.py 和终端测试共用此模块。"""

import json
from .llm import chat_sync
from .memory import save_memory, load_memory

MAX_ROUNDS = 5


def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_reply(reply: str) -> dict:
    """解析 LLM 返回的 JSON。解析失败时返回带 error 字段的 dict。"""
    try:
        return json.loads(clean_json(reply))
    except json.JSONDecodeError:
        return {
            "cards": {},
            "follow_up": None,
            "ready_for_suggestion": False,
            "round": 0,
            "error": "LLM 返回了非法 JSON",
            "raw": reply[:500],
        }


def count_rounds() -> int:
    memory = load_memory()
    return len([m for m in memory if m["role"] == "user"])


def start_analysis(client, question: str) -> dict:
    """开始新的分析会话：清空记忆，发送第一条消息，返回解析后的结果。"""
    save_memory([])
    reply = chat_sync(client, question, "mind_lens")
    data = parse_reply(reply)
    data["round"] = count_rounds()
    return data


def continue_analysis(client, user_message: str) -> dict:
    """继续分析会话：发送用户回复，返回解析后的结果。超过最大轮次强制输出建议。"""
    reply = chat_sync(client, user_message, "mind_lens")
    data = parse_reply(reply)
    rounds = count_rounds()
    data["round"] = rounds
    if rounds >= MAX_ROUNDS and not data.get("ready_for_suggestion"):
        data["ready_for_suggestion"] = True
        data["follow_up"] = None
    return data
