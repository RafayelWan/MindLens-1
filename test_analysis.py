"""终端测试多轮分析流程，无需启动 server。"""

from app.llm import create_client
from app.analysis import start_analysis, continue_analysis, MAX_ROUNDS


def print_cards(cards: dict):
    labels = {
        "surface": "表层问题",
        "assumption": "隐含假设",
        "info_gap": "信息缺失",
        "blind_spot": "认知盲点",
        "suggestion": "建议",
    }
    print("\n" + "=" * 50)
    for key, label in labels.items():
        value = cards.get(key)
        if value:
            print(f"  [{label}] {value}")
        elif key == "suggestion":
            print(f"  [{label}] 🔒 等待解锁")
        else:
            print(f"  [{label}] ...")
    print("=" * 50)


def handle_response(data: dict) -> bool:
    """处理响应数据，返回 True 表示分析结束。"""
    if data.get("error"):
        print(f"\n[错误] {data['error']}")
        if data.get("raw"):
            print(f"原始内容：{data['raw']}")
        return True

    cards = data.get("cards", {})
    follow_up = data.get("follow_up")
    ready = data.get("ready_for_suggestion", False)
    round_num = data.get("round", 0)

    print_cards(cards)

    if ready and cards.get("suggestion"):
        print(f"\n✅ 分析完成（第 {round_num} 轮）")
        return True

    if follow_up:
        print(f"\n💬 AI 追问（第 {round_num} 轮 / 共 {MAX_ROUNDS} 轮）：")
        print(f"   {follow_up}")
        return False

    print(f"\n分析完成（第 {round_num} 轮）")
    return True


def main():
    client = create_client()

    print("\n🔍 Mind Lens 多轮分析测试")
    print("-" * 40)
    question = input("请输入你的问题：\n> ").strip()
    if not question:
        print("未输入问题，退出。")
        return

    data = start_analysis(client, question)
    if handle_response(data):
        return

    while True:
        user_input = input("\n你的回答：\n> ").strip()
        if not user_input:
            print("未输入内容，结束对话。")
            break
        data = continue_analysis(client, user_input)
        if handle_response(data):
            break

    print("\n--- 测试结束 ---")


if __name__ == "__main__":
    main()
