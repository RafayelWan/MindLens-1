import logging
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import APIError

from .analysis import start_analysis, continue_analysis
from .config import API_KEY, BASE_URL, MODEL
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


@app.post("/api/analysis/start")
async def api_analysis_start(req: AnalysisStartRequest, request: Request, response: Response):
    """Start a new analysis session: clear memory, send first message, return parsed JSON."""
    session = _require_session(request, response)
    try:
        return start_analysis(session, req.question)
    except APIError as e:
        logger.exception("LLM API error")
        return JSONResponse(status_code=502, content={"error": f"LLM 服务返回错误: {e.message}"})
    except Exception as e:
        logger.exception("Unexpected error in analysis start")
        return JSONResponse(status_code=500, content={"error": f"服务器内部错误: {str(e)}"})


@app.post("/api/analysis/reply")
async def api_analysis_reply(req: AnalysisReplyRequest, request: Request, response: Response):
    """Continue an ongoing analysis session, return parsed JSON."""
    session = _require_session(request, response)
    try:
        return continue_analysis(session, req.message)
    except APIError as e:
        logger.exception("LLM API error")
        return JSONResponse(status_code=502, content={"error": f"LLM 服务返回错误: {e.message}"})
    except Exception as e:
        logger.exception("Unexpected error in analysis reply")
        return JSONResponse(status_code=500, content={"error": f"服务器内部错误: {str(e)}"})


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
