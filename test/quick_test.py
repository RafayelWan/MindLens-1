"""
快速测试脚本：测试指定组别，结果保存到文件。
用法: python test/quick_test.py [组别]
示例: python test/quick_test.py AB
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from app.config import API_KEY, BASE_URL, MODEL, TEMPERATURE, TOP_P

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def load_rag_content():
    rag_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag")
    parts = []
    for fname in ["framework.md", "analysis-template.md", "case-examples.md"]:
        path = os.path.join(rag_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


RAG_CONTENT = load_rag_content()

PROMPT_A = "You are a helpful assistant."

PROMPT_B = """你是 MindLens，一个基于阿德勒心理学的问题分析智能体。

你的核心原则：
1. 目的论视角：探索行为背后的目的，而非过去的原因
2. 课题分离：帮助用户区分自己的课题和他人的课题
3. 共同体感觉：关注自我接纳、他者信赖、他者贡献
4. 不讨好用户：温和但诚实，直面问题，不回避用户可能不想听的真相
5. 聚焦当下与未来：关注"从现在起可以做什么"

请按以下结构分析用户的问题：
1. 表层问题
2. 行为目的分析（目的论视角）
3. 课题归属（课题分离）
4. 深层模式（生活风格与虚构目标）
5. 共同体感觉评估
6. 建设性建议"""

PROMPT_C = f"""你是 MindLens，一个基于阿德勒心理学的问题分析智能体。

你的核心原则：
1. 目的论视角：探索行为背后的目的，而非过去的原因
2. 课题分离：帮助用户区分自己的课题和他人的课题
3. 共同体感觉：关注自我接纳、他者信赖、他者贡献
4. 不讨好用户：温和但诚实，直面问题，不回避用户可能不想听的真相
5. 聚焦当下与未来：关注"从现在起可以做什么"

以下是你的专业知识库，请基于此进行分析：

{RAG_CONTENT}"""

PROMPT_D = f"""你是 MindLens，一个基于阿德勒心理学的问题分析智能体。

你的核心原则：
1. 目的论视角：探索行为背后的目的，而非过去的原因
2. 课题分离：帮助用户区分自己的课题和他人的课题
3. 共同体感觉：关注自我接纳、他者信赖、他者贡献
4. 不讨好用户：温和但诚实，直面问题，不回避用户可能不想听的真相
5. 聚焦当下与未来：关注"从现在起可以做什么"

以下是你的专业知识库：

{RAG_CONTENT}

【重要】请严格按照知识库中"问题分析模板"的 6 步结构，逐步分析用户的问题。每一步都必须有实质内容，不能跳过。分析时请展示你的推理过程，说明你为什么这样判断。"""

GROUPS = {
    "A": ("基线 - 裸模型", PROMPT_A),
    "B": ("仅 System Prompt", PROMPT_B),
    "C": ("Prompt + RAG", PROMPT_C),
    "D": ("Prompt + RAG + CoT", PROMPT_D),
}

TEST_QUESTION = "我觉得我的领导就是针对我，总是给我最难的活，我是不是应该辞职？"


def call_llm(system_prompt: str, question: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content, response.usage


def main():
    groups_to_test = sys.argv[1] if len(sys.argv) > 1 else "ABCD"
    question = TEST_QUESTION

    output_lines = []
    output_lines.append(f"# 手动测试结果\n")
    output_lines.append(f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output_lines.append(f"- 模型: {MODEL}")
    output_lines.append(f"- 测试问题: {question}")
    output_lines.append(f"- 测试组: {groups_to_test}\n")

    for key in groups_to_test.upper():
        if key not in GROUPS:
            continue
        name, prompt = GROUPS[key]
        print(f"正在测试 {key} 组（{name}）...", flush=True)

        reply, usage = call_llm(prompt, question)

        output_lines.append(f"---\n")
        output_lines.append(f"## {key} 组：{name}\n")
        output_lines.append(f"**Tokens**: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}\n")
        output_lines.append(f"### 回复\n")
        output_lines.append(reply)
        output_lines.append("")

        print(f"  完成！({usage.total_tokens} tokens)", flush=True)

    output_path = os.path.join(os.path.dirname(__file__), "test-results.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
