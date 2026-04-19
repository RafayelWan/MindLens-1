from config import MODEL, BASE_URL, list_prompts
from memory import save_memory
from llm import create_client, chat_stream


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
            print(f"  当前 prompt: {current_prompt_name}")
            print(f"  可用列表: {', '.join(list_prompts())}\n")
            continue

        if lower.startswith("prompt "):
            from config import load_system_prompt
            name = user_input[7:].strip()
            test = load_system_prompt(name)
            if test == "You are a helpful assistant." and name != "default":
                print(f"  [未找到 prompts/{name}.txt]\n")
            else:
                current_prompt_name = name
                save_memory([])
                print(f"  [已切换到 {name}，记忆已清空]\n")
            continue

        try:
            print("AI: ", end="", flush=True)
            for text in chat_stream(client, user_input, current_prompt_name):
                print(text, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"[错误] {e}\n")


if __name__ == "__main__":
    main()
