import json
import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import APIError

from .analysis import parse_reply, _count_user_rounds, MAX_ROUNDS
from .config import API_KEY, BASE_URL, MODEL
from .llm import chat_stream_async
from .session import create_session, get_session

logger = logging.getLogger(__name__)

app = FastAPI(title="DAWN Demo API")

SESSION_COOKIE = "mindlens_sid"

_has_server_key = bool(API_KEY)


def _require_session(request: Request, response: Response):
    """Extract session from cookie, auto-provision from env vars if possible."""
    sid = request.cookies.get(SESSION_COOKIE)
    session = get_session(sid) if sid else None

    if session is None and _has_server_key:
        sid = create_session(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=sid,
            httponly=True,
            samesite="lax",
            max_age=3600 * 4,
        )
        session = get_session(sid)

    if session is None:
        raise HTTPException(status_code=401, detail="请先配置 API Key")
    return session


class ApiConfigRequest(BaseModel):
    api_key: str
    api_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"


class AnalysisStartRequest(BaseModel):
    question: str


class AnalysisReplyRequest(BaseModel):
    message: str


@app.post("/api/config")
async def api_config(req: ApiConfigRequest, response: Response):
    """Receive API Key, URL and model, create a session, set session cookie."""
    if not req.api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    sid = create_session(api_key=req.api_key, base_url=req.api_url, model=req.model)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="strict",
        max_age=3600 * 4,
    )
    return {"status": "ok"}


@app.get("/api/config/status")
async def api_config_status(request: Request, response: Response):
    """Check whether the current session has a valid API config.

    If no session exists but server-level env vars are set, auto-provision one.
    """
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and get_session(sid):
        return {"configured": True}

    if _has_server_key:
        sid = create_session(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=sid,
            httponly=True,
            samesite="lax",
            max_age=3600 * 4,
        )
        return {"configured": True, "server_provided": True}

    return {"configured": False, "server_provided": False}


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_analysis(session, user_input: str, *, is_start: bool):
    """Async generator: yields SSE events directly on the event loop for low latency."""
    if is_start:
        session.clear_memory()

    prompt_name = "mind_lens"
    try:
        async for token in chat_stream_async(session, user_input, prompt_name):
            yield _sse_event({"token": token})

        memory = session.get_memory()
        last_reply = memory[-1]["content"] if memory else ""
        data = parse_reply(last_reply)
        rounds = _count_user_rounds(session)
        data["round"] = rounds

        if rounds >= MAX_ROUNDS and not data.get("ready_for_suggestion"):
            data["ready_for_suggestion"] = True
            data["follow_up"] = None

        yield _sse_event({"done": True, "result": data})

    except APIError as e:
        logger.exception("LLM API error")
        yield _sse_event({"error": f"LLM 服务返回错误: {e.message}"})
    except Exception as e:
        logger.exception("Unexpected error during streaming")
        yield _sse_event({"error": f"服务器内部错误: {str(e)}"})


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.post("/api/analysis/start")
async def api_analysis_start(req: AnalysisStartRequest, request: Request, response: Response):
    """Start a new analysis session via SSE streaming."""
    session = _require_session(request, response)
    return StreamingResponse(
        _stream_analysis(session, req.question, is_start=True),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/api/analysis/reply")
async def api_analysis_reply(req: AnalysisReplyRequest, request: Request, response: Response):
    """Continue an ongoing analysis session via SSE streaming."""
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


@app.get("/api/memory")
async def api_memory(request: Request, response: Response):
    session = _require_session(request, response)
    return session.get_memory()


app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
