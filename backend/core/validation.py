"""
Validation rules for clinical values extracted by the Grok extraction
engine. Values outside plausible medical ranges are flagged rather than
silently accepted, so the conversation manager can ask the patient to
confirm them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# (field, min, max, unit) - only numeric fields need range checks
NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (0, 120),
    "diabetes_duration": (0, 80),
    "hba1c": (3, 20),
    "ldl": (0, 400),
    "hdl": (0, 150),
    "triglycerides": (0, 2000),
    "hemoglobin": (0, 25),
}


def validate_blood_pressure(value: str) -> bool:
    """Accept strings like '120/80'. Reject nonsensical readings."""
    if not value or not isinstance(value, str):
        return False
    parts = value.replace(" ", "").split("/")
    if len(parts) != 2:
        return False
    try:
        systolic, diastolic = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    if not (60 <= systolic <= 260):
        return False
    if not (30 <= diastolic <= 180):
        return False
    if systolic <= diastolic:
        return False
    return True


def validate_structured_data(data: Dict[str, Any]) -> List[str]:
    """
    Returns a list of field names whose values are outside plausible
    medical ranges (i.e. suspicious / likely wrong) and should be
    confirmed with the patient before being trusted.
    """
    suspicious: List[str] = []

    for field, (lo, hi) in NUMERIC_RANGES.items():
        value = data.get(field)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            suspicious.append(field)
            continue
        if not (lo <= numeric_value <= hi):
            suspicious.append(field)

    bp = data.get("blood_pressure")
    if bp is not None and not validate_blood_pressure(str(bp)):
        suspicious.append("blood_pressure")

    return suspicious
