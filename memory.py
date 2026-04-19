import json
import os
from config import MEMORY_FILE, MAX_MEMORY_ROUNDS


def load_memory() -> list[dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_memory(messages: list[dict]) -> None:
    trimmed = [{"role": m["role"], "content": m["content"]} for m in messages]
    if len(trimmed) > MAX_MEMORY_ROUNDS * 2:
        trimmed = trimmed[-(MAX_MEMORY_ROUNDS * 2):]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def append_to_memory(role: str, content: str) -> list[dict]:
    messages = load_memory()
    messages.append({"role": role, "content": content})
    save_memory(messages)
    return messages
