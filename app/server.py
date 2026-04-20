import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .llm import create_client, chat_sync, chat_stream
from .memory import load_memory, save_memory

app = FastAPI(title="DAWN Demo API")

client = create_client()


class ChatRequest(BaseModel):
    message: str
    prompt_name: str = "default"
    stream: bool = True


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if req.stream:
        return StreamingResponse(
            _sse_wrapper(req.message, req.prompt_name),
            media_type="text/event-stream",
        )
    reply = chat_sync(client, req.message, req.prompt_name)
    return {"reply": reply}


async def _sse_wrapper(user_input: str, prompt_name: str):
    for text in chat_stream(client, user_input, prompt_name):
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
