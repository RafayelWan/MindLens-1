"""Session-based API configuration management.

Stores per-session API keys and OpenAI client instances in memory.
All data is lost on server restart — by design.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


SESSION_TTL = 3600 * 4  # 4 hours

_sessions: dict[str, "SessionData"] = {}


MAX_MEMORY_MESSAGES = 20


@dataclass
class SessionData:
    api_key: str
    base_url: str
    model: str
    client: OpenAI
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


def create_session(api_key: str, base_url: str, model: str = "gpt-4o") -> str:
    """Create a new session, return session ID."""
    _cleanup_expired()
    session_id = uuid.uuid4().hex
    client = OpenAI(api_key=api_key, base_url=base_url)
    _sessions[session_id] = SessionData(
        api_key=api_key,
        base_url=base_url,
        model=model,
        client=client,
    )
    return session_id


def get_session(session_id: str) -> Optional[SessionData]:
    """Retrieve session data. Returns None if expired or missing."""
    session = _sessions.get(session_id)
    if session is None:
        return None
    if time.time() - session.last_active > SESSION_TTL:
        _sessions.pop(session_id, None)
        return None
    session.last_active = time.time()
    return session


def get_client(session_id: str) -> Optional[OpenAI]:
    """Convenience: get the OpenAI client for a session."""
    session = get_session(session_id)
    return session.client if session else None


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def _cleanup_expired() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s.last_active > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
