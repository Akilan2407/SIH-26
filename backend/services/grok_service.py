"""
Grok Extraction Engine.

Wraps the xAI Grok API (OpenAI-compatible) to turn a free-text patient
message into structured clinical data, free-form dynamic context, a list
of missing core fields, and a natural next follow-up question.

This module does NOT do image analysis, DR grading, or PDF generation.
It is purely a text-understanding service for the chatbot.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, APIConnectionError, AuthenticationError

from backend.models.schemas import CORE_FIELDS, REQUIRED_FIELDS

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-2-latest")
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")

SYSTEM_PROMPT = f"""You are a clinical intake assistant for a Diabetic
Retinopathy screening chatbot. You NEVER diagnose or grade disease
severity. Your only job is to read the patient's latest message (plus
conversation context) and return structured JSON.

Core clinical fields you should try to extract when mentioned:
{", ".join(CORE_FIELDS)}

Rules:
- age: integer years.
- diabetes_duration: number of years the patient has had diabetes.
- hba1c: percentage number (e.g. 9.2).
- blood_pressure: string like "120/80".
- ldl, hdl, triglycerides: numbers (mg/dL).
- hemoglobin: number (g/dL).
- smoking, blurred_vision, floaters, vision_loss, eye_pain: true/false
  booleans if the patient states or clearly implies them, otherwise omit.
- Anything medically relevant that does NOT fit a core field (family
  history, occupation, previous treatments, surgeries, medication,
  glasses use, lifestyle, etc.) goes into "dynamic_data" as free-form
  key/value pairs. NEVER discard relevant information.
- Only include fields you are reasonably confident about. Omit fields
  you don't know instead of guessing or inserting null placeholders for
  everything.
- missing_fields must be computed from this required list, using ONLY
  fields that are STILL unknown after considering both the new message
  and the fields already known so far: {", ".join(REQUIRED_FIELDS)}.
- next_question must be ONE short, warm, conversational follow-up
  question (not a form label) that asks about the single most useful
  missing piece of information. If nothing required is missing, return
  an empty string for next_question.

Respond ONLY with strict JSON in exactly this shape, no markdown fences,
no commentary:

{{
  "structured_data": {{}},
  "dynamic_data": {{}},
  "missing_fields": [],
  "next_question": ""
}}
"""


class GrokServiceError(Exception):
    """Raised when the Grok API cannot be reached or returns bad data."""


def _get_client() -> OpenAI:
    if not GROK_API_KEY:
        raise GrokServiceError(
            "GROK_API_KEY is not set. Add it to your backend/.env file."
        )
    return OpenAI(api_key=GROK_API_KEY, base_url=GROK_BASE_URL)


def _build_context_messages(
    conversation_history: List[Dict[str, Any]],
    known_structured: Dict[str, Any],
    known_dynamic: Dict[str, Any],
    latest_message: str,
) -> List[Dict[str, str]]:
    history_snippet = conversation_history[-10:]  # keep prompt small
    history_text = "\n".join(
        f'{turn.get("role", "user")}: {turn.get("message", "")}'
        for turn in history_snippet
    )

    context_block = (
        f"Known structured data so far:\n{json.dumps(known_structured)}\n\n"
        f"Known dynamic data so far:\n{json.dumps(known_dynamic)}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Latest patient message:\n{latest_message}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_block},
    ]


def _safe_json_parse(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    # Strip accidental markdown fences if the model adds them anyway.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage the first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise GrokServiceError(
                    f"Grok returned invalid JSON: {exc}"
                ) from exc
        raise GrokServiceError("Grok returned a non-JSON response.")


def extract_information(
    latest_message: str,
    conversation_history: List[Dict[str, Any]],
    known_structured: Dict[str, Any],
    known_dynamic: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calls Grok to extract structured + dynamic clinical data from the
    latest patient message, and to suggest the next follow-up question.

    Returns a dict shaped like:
    {
        "structured_data": {...},
        "dynamic_data": {...},
        "missing_fields": [...],
        "next_question": "..."
    }
    """
    client = _get_client()
    messages = _build_context_messages(
        conversation_history, known_structured, known_dynamic, latest_message
    )

    try:
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except AuthenticationError as exc:
        raise GrokServiceError(
            "Grok authentication failed. Check GROK_API_KEY."
        ) from exc
    except APIConnectionError as exc:
        raise GrokServiceError(
            "Could not connect to the Grok API. Check your network / GROK_BASE_URL."
        ) from exc
    except APIError as exc:
        raise GrokServiceError(f"Grok API error: {exc}") from exc

    if not response.choices:
        raise GrokServiceError("Grok returned no choices in the response.")

    raw_text = response.choices[0].message.content or ""
    parsed = _safe_json_parse(raw_text)

    parsed.setdefault("structured_data", {})
    parsed.setdefault("dynamic_data", {})
    parsed.setdefault("missing_fields", [])
    parsed.setdefault("next_question", "")

    if not isinstance(parsed["structured_data"], dict):
        parsed["structured_data"] = {}
    if not isinstance(parsed["dynamic_data"], dict):
        parsed["dynamic_data"] = {}
    if not isinstance(parsed["missing_fields"], list):
        parsed["missing_fields"] = []
    if not isinstance(parsed["next_question"], str):
        parsed["next_question"] = ""

    return parsed
