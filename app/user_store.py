"""File-based user storage.

Directory layout:
  users/<email>/
    profile.json   – email + password_hash
    notes.json     – list of note objects
    sessions.json  – list of session objects
"""

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

USERS_DIR = Path(__file__).parent.parent / "users"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _email_to_dirname(email: str) -> str:
    """Sanitize email so it is safe to use as a directory name."""
    return re.sub(r'[<>:"/\\|?*]', "_", email.lower())


def _user_dir(email: str) -> Path:
    return USERS_DIR / _email_to_dirname(email)


def _read_json(path: Path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def register(email: str, name: str, password: str) -> dict:
    """Create a new user. Raises ValueError if email already exists."""
    d = _user_dir(email)
    profile_path = d / "profile.json"
    if profile_path.exists():
        raise ValueError("该邮箱已注册")

    profile = {
        "user_id": str(uuid.uuid4()),
        "email": email,
        "name": name,
        "password_hash": _hash_password(password),
        "created_at": time.time(),
    }
    _write_json(profile_path, profile)
    _write_json(d / "notes.json", [])
    _write_json(d / "sessions.json", [])
    return {"user_id": profile["user_id"], "name": name}


def login(email: str, password: str) -> dict:
    """Verify credentials. Raises ValueError on bad email or wrong password."""
    profile_path = _user_dir(email) / "profile.json"
    if not profile_path.exists():
        raise ValueError("该邮箱未注册")

    profile = _read_json(profile_path, {})
    if profile.get("password_hash") != _hash_password(password):
        raise ValueError("密码错误")

    return {"user_id": profile["user_id"], "name": profile["name"]}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def get_notes(email: str) -> list:
    return _read_json(_user_dir(email) / "notes.json", [])


def add_note(email: str, content: str) -> dict:
    notes = get_notes(email)
    note = {
        "id": str(uuid.uuid4()),
        "content": content,
        "created_at": time.time(),
    }
    notes.insert(0, note)
    _write_json(_user_dir(email) / "notes.json", notes)
    return note


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def get_sessions(email: str) -> list:
    return _read_json(_user_dir(email) / "sessions.json", [])


def save_session(email: str, question: str, messages: list, cards: dict) -> dict:
    """Upsert a session identified by its question (first match)."""
    sessions = get_sessions(email)

    # Try to find existing session for this question
    existing = next((s for s in sessions if s.get("question") == question), None)
    if existing:
        existing["messages"] = messages
        existing["cards"] = cards
        existing["updated_at"] = time.time()
        session = existing
    else:
        session = {
            "id": str(uuid.uuid4()),
            "question": question,
            "messages": messages,
            "cards": cards,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        sessions.insert(0, session)

    # Keep at most 50 sessions per user
    sessions = sessions[:50]
    _write_json(_user_dir(email) / "sessions.json", sessions)
    return session
