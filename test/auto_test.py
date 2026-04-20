"""
自动化 A/B 测试 + 评分脚本

功能：
1. 对 4 组配置（A/B/C/D）× 10 个测试问题，自动调用 LLM 获取回复
2. 使用评估模型对每个回复进行 6 维度评分
3. 汇总结果，生成对比报告

用法:
    python test/auto_test.py                # 运行全部测试 + 评分
    python test/auto_test.py --generate     # 只生成回复，不评分
    python test/auto_test.py --evaluate     # 只评分（基于已有回复）
    python test/auto_test.py --groups AB    # 只测试 A、B 两组
    python test/auto_test.py --questions 1 3 5  # 只测试 Q1、Q3、Q5
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from app.config import API_KEY, BASE_URL, MODEL, TEMPERATURE, TOP_P

# ── 配置 ──────────────────────────────────────────────────

EVAL_MODEL = os.getenv("MINDLENS_EVAL_MODEL", MODEL)
EVAL_BASE_URL = os.getenv("MINDLENS_EVAL_BASE_URL", BASE_URL)
EVAL_API_KEY = os.getenv("MINDLENS_EVAL_API_KEY", API_KEY)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
REPLIES_FILE = os.path.join(OUTPUT_DIR, "replies.json")
SCORES_FILE = os.path.join(OUTPUT_DIR, "scores.json")
REPORT_FILE = os.path.join(OUTPUT_DIR, "report.md")

# ── 测试问题 ──────────────────────────────────────────────

QUESTIONS = [
    {"id": "Q1", "text": "我觉得我的领导就是针对我，总是给我最难的活，我是不是应该辞职？",
     "category": "强情感倾向"},
    {"id": "Q2", "text": "我伴侣每次吵架都不理我，ta是不是根本不爱我？",
     "category": "强情感倾向"},
    {"id": "Q3", "text": "我已经30岁了还没结婚，父母天天催，我是不是有问题？",
     "category": "强情感倾向"},
    {"id": "Q4", "text": "为什么我怎么努力都比不过同事？",
     "category": "隐含假设"},
    {"id": "Q5", "text": "孩子不听话是不是因为我教育失败了？",
     "category": "隐含假设"},
    {"id": "Q6", "text": "我总是拖延怎么办？",
     "category": "需深层分析"},
    {"id": "Q7", "text": "我朋友借钱不还，我该怎么要回来？",
     "category": "需深层分析"},
    {"id": "Q8", "text": "我想辞职创业，但又怕失败养不起自己，怎么办？",
     "category": "矛盾冲突"},
    {"id": "Q9", "text": "我不想加班，但不加班就升不了职，该怎么选？",
     "category": "矛盾冲突"},
    {"id": "Q10", "text": "如何培养每天阅读的习惯？",
     "category": "中性对照"},
]

# ── RAG 知识库 ────────────────────────────────────────────

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

# ── 4 组 System Prompt ───────────────────────────────────

MINDLENS_BASE = """你是 MindLens，一个基于阿德勒心理学的问题分析智能体。

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

COT_INSTRUCTION = """

【重要】请严格按照知识库中"问题分析模板"的 6 步结构，逐步分析用户的问题。每一步都必须有实质内容，不能跳过。分析时请展示你的推理过程，说明你为什么这样判断。"""

GROUP_PROMPTS = {
    "A": "You are a helpful assistant.",
    "B": MINDLENS_BASE,
    "C": f"{MINDLENS_BASE}\n\n以下是你的专业知识库，请基于此进行分析：\n\n{RAG_CONTENT}",
    "D": f"{MINDLENS_BASE}\n\n以下是你的专业知识库：\n\n{RAG_CONTENT}{COT_INSTRUCTION}",
}

GROUP_NAMES = {
    "A": "基线 - 裸模型",
    "B": "仅 System Prompt",
    "C": "Prompt + RAG",
    "D": "Prompt + RAG + CoT",
}

# ── 评分 Prompt ───────────────────────────────────────────

def build_eval_prompt(question: str, response: str) -> str:
    """用 f-string 构建评分 prompt，避免 .format() 的花括号冲突。"""
    return f"""你是一位专业的 AI 输出质量评估专家。请根据以下维度对 AI 回复进行评分（1-5分），并给出简短的评分理由。

【用户问题】
{question}

【AI 回复】
{response}

【评分维度】
1. 客观性（objectivity, 1-5）：是否避免了讨好用户，能否直面问题？1=完全讨好，5=温和但直面问题
2. 分析深度（depth, 1-5）：是否揭示了隐含假设和认知盲点？1=只回应表面，5=揭示深层动机
3. 结构性（structure, 1-5）：回答是否有清晰的分析框架？1=无结构，5=严格框架
4. 逻辑连贯性（coherence, 1-5）：推理链是否完整、不跳步？1=跳跃矛盾，5=推理链清晰
5. 实用性（practicality, 1-5）：建议是否具体可执行？1=空洞鸡汤，5=具体可执行
6. 专业性（professionalism, 1-5）：是否体现了心理学专业视角？1=无专业视角，5=自然运用心理学框架

请严格输出以下 JSON 格式，不要输出其他内容：
{{
  "objectivity": {{"score": 0, "reason": "..."}},
  "depth": {{"score": 0, "reason": "..."}},
  "structure": {{"score": 0, "reason": "..."}},
  "coherence": {{"score": 0, "reason": "..."}},
  "practicality": {{"score": 0, "reason": "..."}},
  "professionalism": {{"score": 0, "reason": "..."}},
  "total": 0,
  "summary": "一句话总结"
}}"""

# ── 工具函数 ──────────────────────────────────────────────

def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_eval_client():
    return OpenAI(api_key=EVAL_API_KEY, base_url=EVAL_BASE_URL)

def call_llm(client, system_prompt, user_message, model=None, temperature=None):
    resp = client.chat.completions.create(
        model=model or MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature if temperature is not None else TEMPERATURE,
        top_p=TOP_P,
    )
    return resp.choices[0].message.content, {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── 第一阶段：生成回复 ───────────────────────────────────

def generate_replies(groups, question_ids):
    client = get_client()
    replies = load_json(REPLIES_FILE)

    questions = [q for q in QUESTIONS if q["id"] in question_ids]
    total = len(groups) * len(questions)
    done = 0

    for group in groups:
        group_name = GROUP_NAMES[group]
        prompt = GROUP_PROMPTS[group]

        for q in questions:
            key = f"{group}_{q['id']}"
            if key in replies:
                done += 1
                print(f"  [{done}/{total}] {group}组 × {q['id']} — 已有缓存，跳过")
                continue

            done += 1
            print(f"  [{done}/{total}] {group}组（{group_name}）× {q['id']}...", end="", flush=True)

            try:
                reply, usage = call_llm(client, prompt, q["text"])
                replies[key] = {
                    "group": group,
                    "group_name": group_name,
                    "question_id": q["id"],
                    "question": q["text"],
                    "category": q["category"],
                    "reply": reply,
                    "usage": usage,
                    "timestamp": datetime.now().isoformat(),
                }
                save_json(replies, REPLIES_FILE)
                print(f" 完成（{usage['total_tokens']} tokens）")
                time.sleep(0.5)
            except Exception as e:
                print(f" 错误: {e}")

    return replies

# ── 第二阶段：自动评分 ───────────────────────────────────

def evaluate_replies(replies, groups, question_ids):
    eval_client = get_eval_client()
    scores = load_json(SCORES_FILE)

    keys = [f"{g}_{q}" for g in groups for q in question_ids if f"{g}_{q}" in replies]
    total = len(keys)
    done = 0

    for key in keys:
        if key in scores:
            done += 1
            print(f"  [{done}/{total}] {key} — 已有评分，跳过")
            continue

        done += 1
        entry = replies[key]
        print(f"  [{done}/{total}] 评分 {key}...", end="", flush=True)

        eval_input = build_eval_prompt(
            question=entry["question"],
            response=entry["reply"],
        )

        try:
            raw_score, _ = call_llm(
                eval_client,
                "你是一个严格的评分专家。只输出 JSON，不要输出其他内容。",
                eval_input,
                model=EVAL_MODEL,
                temperature=0.1,
            )

            raw_score = raw_score.strip()
            if raw_score.startswith("```"):
                raw_score = raw_score.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            score_data = json.loads(raw_score)

            if "total" not in score_data:
                dims = ["objectivity", "depth", "structure", "coherence", "practicality", "professionalism"]
                score_data["total"] = sum(score_data[d]["score"] for d in dims if d in score_data)

            scores[key] = {
                "group": entry["group"],
                "group_name": entry["group_name"],
                "question_id": entry["question_id"],
                "category": entry["category"],
                "scores": score_data,
                "timestamp": datetime.now().isoformat(),
            }
            save_json(scores, SCORES_FILE)
            print(f" 完成（总分: {score_data['total']}）")
            time.sleep(0.5)

        except json.JSONDecodeError as e:
            print(f" JSON 解析失败: {e}")
            print(f"    原始返回: {raw_score[:200]}...")
        except Exception as e:
            print(f" 错误: {e}")

    return scores

# ── 第三阶段：生成报告 ───────────────────────────────────

def generate_report(replies, scores):
    lines = []
    lines.append("# MindLens 技术选型 A/B 测试报告\n")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 测试模型: {MODEL}")
    lines.append(f"- 评估模型: {EVAL_MODEL}")
    lines.append(f"- 测试问题: {len(QUESTIONS)} 个")
    lines.append(f"- 测试组: {len(GROUP_NAMES)} 组\n")

    dims = ["objectivity", "depth", "structure", "coherence", "practicality", "professionalism"]
    dim_names = {"objectivity": "客观性", "depth": "分析深度", "structure": "结构性",
                 "coherence": "逻辑连贯", "practicality": "实用性", "professionalism": "专业性"}

    group_scores = {}
    for group in GROUP_NAMES:
        group_scores[group] = {d: [] for d in dims}
        group_scores[group]["total"] = []

    for key, entry in scores.items():
        group = entry["group"]
        s = entry["scores"]
        for d in dims:
            if d in s and isinstance(s[d], dict) and "score" in s[d]:
                group_scores[group][d].append(s[d]["score"])
        if "total" in s:
            group_scores[group]["total"].append(s["total"])

    # 各组平均分总览
    lines.append("## 一、各组平均分总览\n")
    header = "| 维度 |"
    sep = "|------|"
    for group in GROUP_NAMES:
        header += f" {group}（{GROUP_NAMES[group]}）|"
        sep += "---:|"
    lines.append(header)
    lines.append(sep)

    for d in dims:
        row = f"| {dim_names[d]} |"
        for group in GROUP_NAMES:
            vals = group_scores[group][d]
            avg = sum(vals) / len(vals) if vals else 0
            row += f" {avg:.1f} |"
        lines.append(row)

    row = "| **总分** |"
    for group in GROUP_NAMES:
        vals = group_scores[group]["total"]
        avg = sum(vals) / len(vals) if vals else 0
        row += f" **{avg:.1f}** |"
    lines.append(row)

    # Token 消耗
    lines.append("\n## 二、Token 消耗对比\n")
    lines.append("| 组别 | 平均 Prompt Tokens | 平均 Completion Tokens | 平均总 Tokens |")
    lines.append("|------|---:|---:|---:|")

    for group in GROUP_NAMES:
        prompt_tokens = []
        comp_tokens = []
        total_tokens = []
        for key, entry in replies.items():
            if entry["group"] == group:
                prompt_tokens.append(entry["usage"]["prompt_tokens"])
                comp_tokens.append(entry["usage"]["completion_tokens"])
                total_tokens.append(entry["usage"]["total_tokens"])
        if total_tokens:
            lines.append(f"| {group}（{GROUP_NAMES[group]}）| "
                        f"{sum(prompt_tokens)/len(prompt_tokens):.0f} | "
                        f"{sum(comp_tokens)/len(comp_tokens):.0f} | "
                        f"{sum(total_tokens)/len(total_tokens):.0f} |")

    # 技术增量分析
    lines.append("\n## 三、技术增量分析\n")
    for pair, desc in [("A→B", "System Prompt 效果"), ("B→C", "RAG 增量效果"), ("C→D", "CoT 增量效果")]:
        g1, g2 = pair[0], pair[-1]
        t1 = group_scores[g1]["total"]
        t2 = group_scores[g2]["total"]
        if t1 and t2:
            avg1 = sum(t1) / len(t1)
            avg2 = sum(t2) / len(t2)
            delta = avg2 - avg1
            lines.append(f"- **{pair}（{desc}）**: {avg1:.1f} → {avg2:.1f}（{'+' if delta >= 0 else ''}{delta:.1f}）")

    # 按问题类别分析
    lines.append("\n## 四、按问题类别的表现\n")
    categories = list(dict.fromkeys(q["category"] for q in QUESTIONS))
    for cat in categories:
        lines.append(f"\n### {cat}\n")
        cat_questions = [q for q in QUESTIONS if q["category"] == cat]
        lines.append("| 问题 |" + "".join(f" {g} |" for g in GROUP_NAMES))
        lines.append("|------|" + "".join("---:|" for _ in GROUP_NAMES))
        for q in cat_questions:
            row = f"| {q['id']} |"
            for group in GROUP_NAMES:
                key = f"{group}_{q['id']}"
                if key in scores and "total" in scores[key]["scores"]:
                    row += f" {scores[key]['scores']['total']} |"
                else:
                    row += " - |"
            lines.append(row)

    # 详细评分
    lines.append("\n## 五、详细评分（每题每组）\n")
    for q in QUESTIONS:
        lines.append(f"\n### {q['id']}: {q['text']}\n")
        for group in GROUP_NAMES:
            key = f"{group}_{q['id']}"
            if key in scores:
                s = scores[key]["scores"]
                lines.append(f"**{group}组（{GROUP_NAMES[group]}）** — 总分: {s.get('total', '-')}")
                for d in dims:
                    if d in s and isinstance(s[d], dict):
                        lines.append(f"- {dim_names[d]}: {s[d].get('score', '-')} — {s[d].get('reason', '')}")
                if "summary" in s:
                    lines.append(f"- 总评: {s['summary']}")
                lines.append("")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存到: {REPORT_FILE}")
    return report_text

# ── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MindLens A/B 测试自动化脚本")
    parser.add_argument("--generate", action="store_true", help="只生成回复")
    parser.add_argument("--evaluate", action="store_true", help="只评分")
    parser.add_argument("--report", action="store_true", help="只生成报告")
    parser.add_argument("--groups", type=str, default="ABCD", help="测试组（如 AB、ABCD）")
    parser.add_argument("--questions", nargs="+", type=int, help="测试问题编号（如 1 3 5）")
    args = parser.parse_args()

    groups = list(args.groups.upper())
    question_ids = [f"Q{i}" for i in args.questions] if args.questions else [q["id"] for q in QUESTIONS]

    only_generate = args.generate
    only_evaluate = args.evaluate
    only_report = args.report
    run_all = not only_generate and not only_evaluate and not only_report

    print(f"MindLens A/B 测试")
    print(f"  模型: {MODEL}")
    print(f"  评估模型: {EVAL_MODEL}")
    print(f"  测试组: {', '.join(groups)}")
    print(f"  测试问题: {', '.join(question_ids)}")
    print(f"  总计: {len(groups)} 组 × {len(question_ids)} 题 = {len(groups) * len(question_ids)} 次调用\n")

    if run_all or only_generate:
        print("=" * 50)
        print("阶段 1/3：生成回复")
        print("=" * 50)
        replies = generate_replies(groups, question_ids)
    else:
        replies = load_json(REPLIES_FILE)

    if run_all or only_evaluate:
        print("\n" + "=" * 50)
        print("阶段 2/3：自动评分")
        print("=" * 50)
        scores = evaluate_replies(replies, groups, question_ids)
    else:
        scores = load_json(SCORES_FILE)

    if run_all or only_report:
        print("\n" + "=" * 50)
        print("阶段 3/3：生成报告")
        print("=" * 50)
        generate_report(replies, scores)

    print("\n完成！")


if __name__ == "__main__":
    main()
