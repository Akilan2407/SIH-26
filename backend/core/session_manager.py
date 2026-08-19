"""
Conversation State Manager.

Responsible for:
- Loading / creating session JSON files on disk
- Merging newly extracted structured/dynamic data into the session
- Persisting conversation history
- Determining which core clinical fields are still missing
- Saving session state safely (atomic write, corruption-tolerant read)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from backend.models.schemas import CORE_FIELDS, REQUIRED_FIELDS

DATA_DIR = os.environ.get("DATA_DIR", "data")


def _session_path(session_id: str) -> str:
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
    if not safe_id:
        safe_id = "default"
    return os.path.join(DATA_DIR, f"session_{safe_id}.json")


def _empty_session(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "structured_data": {field: None for field in CORE_FIELDS},
        "dynamic_data": {},
        "conversation_history": [],
        "completed": False,
        "pending_confirmation": None,
    }


def load_session(session_id: str) -> Dict[str, Any]:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _session_path(session_id)

    if not os.path.exists(path):
        session = _empty_session(session_id)
        save_session(session)
        return session

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                raise ValueError("empty file")
            session = json.loads(content)
    except (json.JSONDecodeError, ValueError, OSError):
        # Corrupted session file: back it up and start fresh so the
        # conversation can continue instead of crashing.
        try:
            backup_path = f"{path}.corrupt.{int(time.time())}"
            shutil.move(path, backup_path)
        except OSError:
            pass
        session = _empty_session(session_id)
        save_session(session)
        return session

    # Fill in any keys that might be missing from an older session format.
    defaults = _empty_session(session_id)
    for key, value in defaults.items():
        session.setdefault(key, value)
    for field in CORE_FIELDS:
        session["structured_data"].setdefault(field, None)

    return session


def save_session(session: Dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _session_path(session["session_id"])

    # Atomic write to avoid corrupting the file if the process dies mid-write.
    fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False, default=str)
        shutil.move(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def append_turn(session: Dict[str, Any], role: str, message: str) -> None:
    session["conversation_history"].append(
        {
            "role": role,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def merge_structured_data(
    session: Dict[str, Any], new_data: Dict[str, Any]
) -> None:
    """Only overwrite a field when a real (non-null) new value arrives."""
    for field in CORE_FIELDS:
        value = new_data.get(field)
        if value is not None and value != "":
            session["structured_data"][field] = value


def merge_dynamic_data(session: Dict[str, Any], new_data: Dict[str, Any]) -> None:
    """
    Dynamic data is free-form. Never discard information: merge new keys
    in, and only replace an existing key when a non-empty value is given.
    """
    for key, value in (new_data or {}).items():
        if value is None or value == "":
            continue
        session["dynamic_data"][key] = value


def get_missing_required_fields(session: Dict[str, Any]) -> List[str]:
    structured = session["structured_data"]
    return [f for f in REQUIRED_FIELDS if structured.get(f) in (None, "")]


def is_complete(session: Dict[str, Any]) -> bool:
    return len(get_missing_required_fields(session)) == 0
