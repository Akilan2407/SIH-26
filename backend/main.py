"""
Unified FastAPI backend for chatbot intake and retinal screening.

Endpoints:
    POST /chat                -> converse with the patient
    GET  /session/{session_id} -> fetch stored session JSON
    GET  /health               -> simple health check

The chatbot remains an intake module; screening routes delegate prediction
and Grad-CAM to the existing Keras model wrapper in ``ml/dr_model.py``.

NLP Engine: fully rule-based (no external API calls required).
"""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.core import session_manager, validation
from backend.models.schemas import ChatRequest, ChatResponse, REQUIRED_FIELDS
from backend.services.nlp_service import extract_information
from backend.routes.screening import router as screening_router

app = FastAPI(
    title="DR Screening Integration API",
    description="Unified chatbot and diabetic retinopathy screening API.",
    version="2.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(screening_router)


# ---------------------------------------------------------------------------
# Greeting — sent once when a brand-new session opens.
# The NLP engine handles all follow-up questions dynamically.
# ---------------------------------------------------------------------------
GREETING = (
    "Hello! 👋 I'm Dr. Rita, your virtual screening assistant. "
    "I'll ask you a few quick questions about your health before your "
    "diabetic retinopathy screening. Let's start — what's your name?"
)

COMPLETION_MESSAGE = (
    "Thank you so much! 🎉 All the required information has been collected "
    "successfully. The screening team will review your details shortly. "
    "Take care!"
)

CONFIRMATION_REQUEST_TEMPLATE = (
    "Just to confirm — you mentioned {field} of '{value}'. "
    "That looks a little unusual — could you double-check and confirm it's correct?"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_question(missing_fields: list, name: str | None = None) -> str:
    """
    Deterministic fallback question when the NLP engine returns no next_question.
    Used only as a safety net.
    """
    field_questions = {
        "age": "Could you tell me your age?",
        "diabetes_duration": "How many years have you had diabetes?",
        "hba1c": "Do you know your most recent HbA1c value?",
        "blood_pressure": "Could you share your latest blood pressure reading (e.g. 120/80)?",
    }
    for field in REQUIRED_FIELDS:
        if field in missing_fields:
            q = field_questions.get(field, f"Could you tell me your {field.replace('_', ' ')}?")
            if name:
                return f"{name}, {q[0].lower()}{q[1:]}"
            return q
    return "Is there anything else about your eyes or health you'd like to add?"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "engine": "rule-based-nlp"}


@app.get("/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")
    session = session_manager.load_session(session_id)
    return session


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session_id = (payload.session_id or "").strip()
    message = (payload.message or "").strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    session = session_manager.load_session(session_id)

    # ------------------------------------------------------------------
    # Empty message: brand-new session gets the greeting; existing
    # sessions reject empty messages.
    # ------------------------------------------------------------------
    if not message:
        if not session["conversation_history"]:
            session_manager.append_turn(session, "assistant", GREETING)
            session_manager.save_session(session)
            return ChatResponse(
                reply=GREETING,
                patient_state=session,
                completed=session["completed"],
            )
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    session_manager.append_turn(session, "user", message)

    # ------------------------------------------------------------------
    # Pending confirmation: patient is confirming (or denying) a
    # previously flagged suspicious value.
    # ------------------------------------------------------------------
    if session.get("pending_confirmation"):
        pending = session["pending_confirmation"]
        lowered = message.lower()
        if any(w in lowered for w in ["yes", "correct", "right", "confirm", "yeah", "yep", "yup"]):
            session["structured_data"][pending["field"]] = pending["value"]
        session["pending_confirmation"] = None

    # ------------------------------------------------------------------
    # Run NLP extraction
    # ------------------------------------------------------------------
    extraction = extract_information(
        latest_message=message,
        conversation_history=session["conversation_history"],
        known_structured=session["structured_data"],
        known_dynamic=session["dynamic_data"],
    )

    new_structured = extraction["structured_data"]
    new_dynamic = extraction["dynamic_data"]
    intent = extraction.get("intent", "data")

    # ------------------------------------------------------------------
    # If the intent produced a full reply override (greeting, confusion,
    # farewell, etc.), use it directly without merging any data.
    # ------------------------------------------------------------------
    if "reply_override" in extraction:
        reply = extraction["reply_override"]
        session_manager.append_turn(session, "assistant", reply)
        session_manager.save_session(session)
        missing = session_manager.get_missing_required_fields(session)
        return ChatResponse(
            reply=reply,
            patient_state=session,
            completed=len(missing) == 0 and session.get("completed", False),
        )

    # ------------------------------------------------------------------
    # Validate before merging — flag suspicious clinical values.
    # ------------------------------------------------------------------
    suspicious_fields = validation.validate_structured_data(new_structured)

    if suspicious_fields:
        field = suspicious_fields[0]
        value = new_structured.pop(field)
        session_manager.merge_structured_data(session, new_structured)
        session_manager.merge_dynamic_data(session, new_dynamic)
        session["pending_confirmation"] = {"field": field, "value": value}
        reply = CONFIRMATION_REQUEST_TEMPLATE.format(
            field=field.replace("_", " "), value=value
        )
        session_manager.append_turn(session, "assistant", reply)
        session_manager.save_session(session)
        return ChatResponse(reply=reply, patient_state=session, completed=False)

    session_manager.merge_structured_data(session, new_structured)
    session_manager.merge_dynamic_data(session, new_dynamic)

    # ------------------------------------------------------------------
    # Check completion
    # ------------------------------------------------------------------
    missing = session_manager.get_missing_required_fields(session)
    completed = len(missing) == 0

    if completed:
        session["completed"] = True
        reply = COMPLETION_MESSAGE
    else:
        # Use the NLP-generated contextual question, fall back to static
        nlp_question = extraction.get("next_question", "").strip()
        if nlp_question:
            reply = nlp_question
        else:
            known_name = session["structured_data"].get("name")
            reply = _fallback_question(missing, known_name)

    session_manager.append_turn(session, "assistant", reply)
    session_manager.save_session(session)

    return ChatResponse(reply=reply, patient_state=session, completed=completed)