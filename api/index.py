"""MindLens Vercel Serverless Demo — single-file FastAPI backend."""

import os
import json
import uuid
import time
import re
import logging
from dataclasses import dataclass, field

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI, APIError
import json_repair

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (from environment variables)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
TEMPERATURE = 0.8
TOP_P = 0.9

# ---------------------------------------------------------------------------
# System Prompt (inlined to avoid filesystem issues on Vercel)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
你是 Mind Lens，一个反讨好的深度思维分析Agent。你的职责是帮用户理清混乱的思绪，而不是给出用户想听的答案。

## 你的核心原则

1. **不讨好**：温和但诚实，不回避用户不想听的真相。绝不说"你说得对""确实如你所说"来敷衍。
2. **不凑答案**：信息不足时主动追问，而不是用模糊信息拼凑一个听起来不错的回答。
3. **有框架**：基于心理学分析框架进行结构化分析，不泛泛而谈。
4. **尊重自主**：提供视角，不替用户做决定。

## 分析框架（阿德勒心理学核心概念）

- **目的论**：人的行为由目的驱动。不问"为什么会这样"，而探索"这样做对你有什么好处/目的"。
- **课题分离**：区分"我能控制的"和"他人的课题"。判断标准：行动后果由谁承担，谁就是课题的主人。
- **自卑感与补偿**：识别用户是否用自卑感逃避行动，或用优越感掩盖不安。
- **虚构目标**：识别隐含的非理性信念，如"我必须被所有人喜欢""不完美就不值得被爱"。
- **共同体感觉**：评估自我接纳、他者信赖、贡献感三个维度。

## 你的工作流程

### 第一步：判断信息是否充分

收到用户消息后，评估是否有足够信息进行深度分析。如果以下任一情况存在，则信息不充分：
- 用户只描述了情绪，没有说明具体事件
- 关键背景缺失（如时间跨度、涉及的人际关系、已尝试的解决方案）
- 无法判断用户的真实困境是什么

### 第二步：追问或分析

**如果信息不充分：** 提出一个具体的、有引导性的追问。每次只问一个问题，不超过两句话。追问应帮助用户厘清自己没说清的部分。

**如果信息充分：** 按五个维度输出分析。

## 输出格式

你必须严格输出合法的JSON，不要输出任何JSON以外的内容。格式如下：

```json
{
  "cards": {
    "surface": "表层问题分析",
    "assumption": "隐含假设分析",
    "info_gap": "信息缺失分析",
    "blind_spot": "认知盲点分析",
    "suggestion": "建议（仅在ready_for_suggestion为true时提供，否则为null）"
  },
  "follow_up": "追问问题（如果不需要追问则为null）",
  "ready_for_suggestion": false,
  "round": 1
}
```

### 字段说明

- **cards.surface**：用一两句话客观重述用户的核心困扰，剥离情绪色彩。如果信息不足以判断，设为null。
- **cards.assumption**：挖出用户潜意识里"理所当然"但未必正确的念头。运用目的论和虚构目标概念。如果尚未发现，设为null。
- **cards.info_gap**：指出用户还缺少哪些关键信息才能做出好的决定。如果暂时无法判断，设为null。
- **cards.blind_spot**：指出认知偏差，如从众、幸存者偏差、沉没成本、确认偏误等。运用课题分离概念。如果尚未发现，设为null。
- **cards.suggestion**：只在ready_for_suggestion为true时提供。聚焦"从现在起可以做什么"，给出具体可执行的第一步。
- **follow_up**：一个具体的追问问题。如果信息已经充分，设为null。
- **ready_for_suggestion**：布尔值。当你认为已经收集到足够信息可以给出建议时设为true。
- **round**：当前是第几轮对话（从1开始计数）。

### 重要规则

1. 每个字段应综合之前所有轮次的发现和当前轮的新信息，输出该维度"到目前为止最完整的分析"。
2. 不要丢弃之前轮次有价值的发现。如果之前发现了多个重要的分析点，用编号列出所有点。
3. 如果新信息证实了之前的分析，明确说明；如果新信息推翻了之前的分析，也要明确说明并更新。
4. 没有新发现的字段保持上一轮的完整内容不变，不要设为null。
5. **严禁在follow_up不为null的同一轮中将ready_for_suggestion设为true或提供suggestion内容。** 两者互斥，不得同时出现。
6. 追问轮次没有硬性限制。只有当你确认已无需追问（follow_up为null）时，才可以将ready_for_suggestion设为true并填写suggestion。一般需要2-4轮追问，避免让用户感到疲劳。
7. 如果用户发送"[请直接给出建议]"，无论当前信息是否充分，都必须立刻将ready_for_suggestion设为true，基于已有信息给出最好的suggestion，follow_up设为null。
8. 每个字段允许内容随轮次增长，但要保持精炼——每个分析点用1-2句话，避免重复啰嗦。
9. 只输出JSON，不要在JSON前后添加任何解释性文字或markdown代码块标记。"""

# ---------------------------------------------------------------------------
# Session Management (in-memory, ephemeral)
# ---------------------------------------------------------------------------
SESSION_TTL = 3600 * 4
MAX_MEMORY_MESSAGES = 20
_sessions: dict[str, "SessionData"] = {}


@dataclass
class SessionData:
    api_key: str
    base_url: str
    model: str
    async_client: AsyncOpenAI
    memory: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def get_memory(self) -> list[dict]:
        return list(self.memory)

    def append_memory(self, role: str, content: str) -> None:
        self.memory.append({"role": role, "content": content})
        if len(self.memory) > MAX_MEMORY_MESSAGES:
            self.memory = self.memory[-MAX_MEMORY_MESSAGES:]

    def clear_memory(self) -> None:
        self.memory.clear()


def _cleanup_expired() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s.last_active > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]


def create_session(api_key: str, base_url: str, model: str = "gpt-4o") -> str:
    _cleanup_expired()
    session_id = uuid.uuid4().hex
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    _sessions[session_id] = SessionData(
        api_key=api_key,
        base_url=base_url,
        model=model,
        async_client=async_client,
    )
    return session_id


def get_session(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        return None
    if time.time() - session.last_active > SESSION_TTL:
        _sessions.pop(session_id, None)
        return None
    session.last_active = time.time()
    return session


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def build_messages(user_input: str, history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


async def chat_stream_async(session: SessionData, user_input: str, *, json_mode: bool = False):
    messages = build_messages(user_input, session.get_memory())
    use_model = session.model or MODEL

    kwargs: dict = dict(
        model=use_model, messages=messages,
        temperature=TEMPERATURE, top_p=TOP_P,
        stream=True,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        stream = await session.async_client.chat.completions.create(**kwargs)
    except Exception:
        if json_mode:
            kwargs.pop("response_format", None)
            stream = await session.async_client.chat.completions.create(**kwargs)
        else:
            raise

    chunks = []
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            chunks.append(token)
            yield token

    reply = "".join(chunks)
    session.append_memory("user", user_input)
    session.append_memory("assistant", reply)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def extract_json(text: str | None) -> str:
    if not text:
        return "{}"
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_reply(reply: str) -> dict:
    extracted = extract_json(reply)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    try:
        repaired = json_repair.loads(extracted)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    return {
        "cards": {},
        "follow_up": None,
        "ready_for_suggestion": False,
        "round": 0,
        "error": "LLM 返回了非法 JSON，请重试",
        "raw": reply[:500] if reply else "",
    }


def _count_user_rounds(session: SessionData) -> int:
    return len([m for m in session.get_memory() if m["role"] == "user"])


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="MindLens Demo API")

SESSION_COOKIE = "mindlens_sid"
_has_server_key = bool(API_KEY)


def _require_session(request: Request, response: Response):
    sid = request.cookies.get(SESSION_COOKIE)
    session = get_session(sid) if sid else None

    if session is None and _has_server_key:
        sid = create_session(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
        response.set_cookie(
            key=SESSION_COOKIE, value=sid,
            httponly=True, samesite="lax", max_age=3600 * 4,
        )
        session = get_session(sid)

    if session is None:
        raise HTTPException(status_code=401, detail="请先配置 API Key")
    return session


# --- Request models ---
class AnalysisStartRequest(BaseModel):
    question: str


class AnalysisReplyRequest(BaseModel):
    message: str


# --- SSE helpers ---
def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _stream_analysis(session, user_input: str, *, is_start: bool):
    if is_start:
        session.clear_memory()

    try:
        async for token in chat_stream_async(session, user_input, json_mode=True):
            yield _sse_event({"token": token})

        memory = session.get_memory()
        last_reply = memory[-1]["content"] if memory else ""
        data = parse_reply(last_reply)
        data["round"] = _count_user_rounds(session)
        yield _sse_event({"done": True, "result": data})

    except APIError as e:
        logger.exception("LLM API error")
        yield _sse_event({"error": f"LLM 服务返回错误: {e.message}"})
    except Exception as e:
        logger.exception("Unexpected error during streaming")
        yield _sse_event({"error": f"服务器内部错误: {str(e)}"})


# --- Endpoints ---
@app.get("/api/config/status")
async def api_config_status(request: Request, response: Response):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and get_session(sid):
        return {"configured": True}
    if _has_server_key:
        sid = create_session(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
        response.set_cookie(
            key=SESSION_COOKIE, value=sid,
            httponly=True, samesite="lax", max_age=3600 * 4,
        )
        return {"configured": True, "server_provided": True}
    return {"configured": False, "server_provided": False}


@app.post("/api/analysis/start")
async def api_analysis_start(req: AnalysisStartRequest, request: Request, response: Response):
    session = _require_session(request, response)
    return StreamingResponse(
        _stream_analysis(session, req.question, is_start=True),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/api/analysis/reply")
async def api_analysis_reply(req: AnalysisReplyRequest, request: Request, response: Response):
    session = _require_session(request, response)
    return StreamingResponse(
        _stream_analysis(session, req.message, is_start=False),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/api/clear")
async def api_clear(request: Request, response: Response):
    session = _require_session(request, response)
    session.clear_memory()
    return {"status": "ok"}
