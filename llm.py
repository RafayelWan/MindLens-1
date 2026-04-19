"""LLM 核心层：构建消息、调用 API、管理记忆。chat.py 和 server.py 共用此模块。"""

from typing import Generator
from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL, TEMPERATURE, TOP_P, load_system_prompt
from memory import load_memory, append_to_memory


def create_client() -> OpenAI:
    if not API_KEY or API_KEY == "your-api-key-here":
        raise ValueError("请先在 .env 文件中设置 LLM_API_KEY")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def build_messages(user_input: str, prompt_name: str = "default") -> list[dict]:
    system_prompt = load_system_prompt(prompt_name)
    history = load_memory()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


def chat_sync(client: OpenAI, user_input: str, prompt_name: str = "default") -> str:
    messages = build_messages(user_input, prompt_name)

    response = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=TEMPERATURE, top_p=TOP_P,
    )
    reply = response.choices[0].message.content

    append_to_memory("user", user_input)
    append_to_memory("assistant", reply)
    return reply


def chat_stream(client: OpenAI, user_input: str, prompt_name: str = "default") -> Generator[str, None, str]:
    """流式生成回复，yield 每个文本片段，最终 return 完整回复。"""
    messages = build_messages(user_input, prompt_name)

    stream = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=TEMPERATURE, top_p=TOP_P,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_reply = []
    usage = None

    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_reply.append(text)
            yield text

    reply = "".join(full_reply)
    append_to_memory("user", user_input)
    append_to_memory("assistant", reply)
    return reply, usage
