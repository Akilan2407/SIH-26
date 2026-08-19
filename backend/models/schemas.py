"""
Pydantic models for the Diabetic Retinopathy Screening Chatbot module.

This module ONLY defines the conversational data model. It does not
touch image analysis, DR grading, PDF generation, or authentication.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Core clinical fields tracked explicitly.
# ----------------------------------------------------------------------
CORE_FIELDS = [
    "name",
    "age",
    "gender",
    "diabetes_duration",
    "hba1c",
    "blood_pressure",
    "ldl",
    "hdl",
    "triglycerides",
    "hemoglobin",
    "smoking",
    "blurred_vision",
    "floaters",
    "vision_loss",
    "eye_pain",
]

# Fields that MUST be present before a conversation is considered complete.
REQUIRED_FIELDS = ["age", "diabetes_duration", "hba1c", "blood_pressure"]


class StructuredData(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    diabetes_duration: Optional[float] = None
    hba1c: Optional[float] = None
    blood_pressure: Optional[str] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    hemoglobin: Optional[float] = None
    smoking: Optional[bool] = None
    blurred_vision: Optional[bool] = None
    floaters: Optional[bool] = None
    vision_loss: Optional[bool] = None
    eye_pain: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant" | "system"
    message: str
    timestamp: Optional[str] = None


class PatientSession(BaseModel):
    session_id: str
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    dynamic_data: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    completed: bool = False
    pending_confirmation: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    patient_state: Dict[str, Any]
    completed: bool
