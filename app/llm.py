"""LLM 核心层：构建消息、调用 API、管理记忆。"""

from openai import OpenAI
from .config import API_KEY, BASE_URL, MODEL, TEMPERATURE, TOP_P, load_system_prompt
from .memory import load_memory, append_to_memory


def create_client(api_key: str = None, base_url: str = None) -> OpenAI:
    key = api_key or API_KEY
    url = base_url or BASE_URL
    if not key or key == "your-api-key-here":
        raise ValueError("请先配置 API Key（通过前端设置或 .env 文件）")
    return OpenAI(api_key=key, base_url=url)


def build_messages(user_input: str, prompt_name: str = "default") -> list[dict]:
    system_prompt = load_system_prompt(prompt_name)
    history = load_memory()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


def chat_sync(client: OpenAI, user_input: str, prompt_name: str = "default", model: str = None) -> str:
    messages = build_messages(user_input, prompt_name)
    use_model = model or MODEL

    response = client.chat.completions.create(
        model=use_model, messages=messages,
        temperature=TEMPERATURE, top_p=TOP_P,
    )
    reply = response.choices[0].message.content

    append_to_memory("user", user_input)
    append_to_memory("assistant", reply)
    return reply
