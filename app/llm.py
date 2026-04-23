"""LLM 核心层：构建消息、调用 API。"""

from __future__ import annotations

from openai import OpenAI
from .config import MODEL, TEMPERATURE, TOP_P, load_system_prompt
from .session import SessionData


def build_messages(user_input: str, prompt_name: str, history: list[dict]) -> list[dict]:
    system_prompt = load_system_prompt(prompt_name)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


def chat_sync(
    session: SessionData,
    user_input: str,
    prompt_name: str = "default",
) -> str:
    """Send a message via the session's client, persist to session memory."""
    messages = build_messages(user_input, prompt_name, session.get_memory())
    use_model = session.model or MODEL

    response = session.client.chat.completions.create(
        model=use_model, messages=messages,
        temperature=TEMPERATURE, top_p=TOP_P,
    )
    reply = response.choices[0].message.content

    session.append_memory("user", user_input)
    session.append_memory("assistant", reply)
    return reply


def chat_stream(
    session: SessionData,
    user_input: str,
    prompt_name: str = "default",
):
    """Yield tokens from LLM via streaming. Persists to memory after completion."""
    messages = build_messages(user_input, prompt_name, session.get_memory())
    use_model = session.model or MODEL

    stream = session.client.chat.completions.create(
        model=use_model, messages=messages,
        temperature=TEMPERATURE, top_p=TOP_P,
        stream=True,
    )

    chunks = []
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            chunks.append(token)
            yield token

    reply = "".join(chunks)
    session.append_memory("user", user_input)
    session.append_memory("assistant", reply)


async def chat_stream_async(
    session: SessionData,
    user_input: str,
    prompt_name: str = "default",
):
    """Async version: yield tokens directly on the event loop, no threadpool."""
    messages = build_messages(user_input, prompt_name, session.get_memory())
    use_model = session.model or MODEL

    stream = await session.async_client.chat.completions.create(
        model=use_model, messages=messages,
        temperature=TEMPERATURE, top_p=TOP_P,
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            chunks.append(token)
            yield token

    reply = "".join(chunks)
    session.append_memory("user", user_input)
    session.append_memory("assistant", reply)
