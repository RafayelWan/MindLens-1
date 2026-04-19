import sys
from openai import OpenAI
from config import (
    API_KEY, BASE_URL, MODEL, TEMPERATURE, TOP_P,
    load_system_prompt, list_prompts,
)
from memory import load_memory, append_to_memory, save_memory


def create_client() -> OpenAI:
    if not API_KEY or API_KEY == "your-api-key-here":
        raise ValueError("请先在 .env 文件中设置 LLM_API_KEY")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(client: OpenAI, user_input: str, system_prompt: str) -> str:
    history = load_memory()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
        stream_options={"include_usage": True},
    )

    print("AI: ", end="", flush=True)
    full_reply = []
    usage = None

    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            full_reply.append(text)
    print()

    if usage:
        print(
            f"  [tokens] 输入: {usage.prompt_tokens}  "
            f"输出: {usage.completion_tokens}  "
            f"总计: {usage.total_tokens}"
        )
    print()

    reply = "".join(full_reply)
    append_to_memory("user", user_input)
    append_to_memory("assistant", reply)
    return reply


def show_help():
    print("  命令列表:")
    print("    quit / exit   退出程序")
    print("    clear          清空对话记忆")
    print("    prompt         查看当前 prompt 和可用列表")
    print("    prompt <名称>  切换 system prompt")
    print("    help           显示此帮助\n")


def main():
    print("=== LLM Chat ===")
    print(f"模型: {MODEL}  |  端点: {BASE_URL}")
    print("输入 'help' 查看所有命令\n")

    client = create_client()

    current_prompt_name = "default"
    system_prompt = load_system_prompt(current_prompt_name)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("quit", "exit"):
            print("再见!")
            break

        if lower == "help":
            show_help()
            continue

        if lower == "clear":
            save_memory([])
            print("[记忆已清空]\n")
            continue

        if lower == "prompt":
            available = list_prompts()
            print(f"  当前 prompt: {current_prompt_name}")
            print(f"  可用列表: {', '.join(available)}\n")
            continue

        if lower.startswith("prompt "):
            name = user_input[7:].strip()
            new_prompt = load_system_prompt(name)
            if new_prompt == "You are a helpful assistant." and name != "default":
                print(f"  [未找到 prompts/{name}.txt]\n")
            else:
                current_prompt_name = name
                system_prompt = new_prompt
                save_memory([])
                print(f"  [已切换到 {name}，记忆已清空]\n")
            continue

        try:
            chat(client, user_input, system_prompt)
        except Exception as e:
            print(f"[错误] {e}\n")


if __name__ == "__main__":
    main()
