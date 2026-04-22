import json
import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI, APIError

from .llm import chat_sync, chat_stream
from .memory import load_memory, save_memory
from .analysis import start_analysis, continue_analysis
from .session import create_session, get_client, get_session

logger = logging.getLogger(__name__)

app = FastAPI(title="DAWN Demo API")

SESSION_COOKIE = "mindlens_sid"


def _require_session(request: Request):
    """Extract the full session from the cookie, or raise 401."""
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        raise HTTPException(status_code=401, detail="请先配置 API Key")
    session = get_session(sid)
    if session is None:
        raise HTTPException(status_code=401, detail="会话已过期，请重新配置 API Key")
    return session


class ApiConfigRequest(BaseModel):
    api_key: str
    api_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"


class ChatRequest(BaseModel):
    message: str
    prompt_name: str = "default"
    stream: bool = True


class AnalysisStartRequest(BaseModel):
    question: str


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
async def api_config_status(request: Request):
    """Check whether the current session has a valid API config."""
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and get_session(sid):
        return {"configured": True}
    return {"configured": False}


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    session = _require_session(request)
    try:
        if req.stream:
            return StreamingResponse(
                _sse_wrapper(session.client, req.message, req.prompt_name, session.model),
                media_type="text/event-stream",
            )
        reply = chat_sync(session.client, req.message, req.prompt_name, model=session.model)
        return {"reply": reply}
    except APIError as e:
        logger.exception("LLM API error")
        return JSONResponse(status_code=502, content={"error": f"LLM 服务返回错误: {e.message}"})
    except Exception as e:
        logger.exception("Unexpected error in chat")
        return JSONResponse(status_code=500, content={"error": f"服务器内部错误: {str(e)}"})


@app.post("/api/analysis/start")
async def api_analysis_start(req: AnalysisStartRequest, request: Request):
    """Start a new analysis session: clear memory, send first message, return parsed JSON."""
    session = _require_session(request)
    try:
        return start_analysis(session.client, req.question, model=session.model)
    except APIError as e:
        logger.exception("LLM API error")
        return JSONResponse(status_code=502, content={"error": f"LLM 服务返回错误: {e.message}"})
    except Exception as e:
        logger.exception("Unexpected error in analysis start")
        return JSONResponse(status_code=500, content={"error": f"服务器内部错误: {str(e)}"})


@app.post("/api/analysis/reply")
async def api_analysis_reply(req: ChatRequest, request: Request):
    """Continue an ongoing analysis session, return parsed JSON."""
    session = _require_session(request)
    try:
        return continue_analysis(session.client, req.message, model=session.model)
    except APIError as e:
        logger.exception("LLM API error")
        return JSONResponse(status_code=502, content={"error": f"LLM 服务返回错误: {e.message}"})
    except Exception as e:
        logger.exception("Unexpected error in analysis reply")
        return JSONResponse(status_code=500, content={"error": f"服务器内部错误: {str(e)}"})


async def _sse_wrapper(client: OpenAI, user_input: str, prompt_name: str, model: str):
    for text in chat_stream(client, user_input, prompt_name, model=model):
        yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/api/clear")
async def api_clear():
    save_memory([])
    return {"status": "ok"}


@app.get("/api/memory")
async def api_memory():
    return load_memory()


app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
