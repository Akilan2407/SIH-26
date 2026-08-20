"""
SHAP Explainability Service — Tabular DR Risk Model.

Architecture
------------
1. A LightGBM 5-class classifier is trained once (lazily, on first request)
   on a 2 000-row synthetic cohort whose risk correlations mirror published
   diabetic-retinopathy literature.
2. SHAP TreeExplainer computes exact Shapley values for the patient's feature
   vector in milliseconds.
3. The endpoint returns per-feature SHAP contributions ranked by |shap|,
   alongside the model's class probabilities and a plain-English summary.

No external data file is needed — the synthetic cohort is generated
deterministically from a fixed seed, so results are reproducible.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast and avoid hard failures when these
# packages are absent (they are listed in requirements.txt).
# ---------------------------------------------------------------------------
_lgbm = None
_shap = None


def _get_lgbm():
    global _lgbm
    if _lgbm is None:
        import lightgbm as lgbm  # noqa: PLC0415
        _lgbm = lgbm
    return _lgbm


def _get_shap():
    global _shap
    if _shap is None:
        import shap  # noqa: PLC0415
        _shap = shap
    return _shap


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "age",
    "diabetes_duration",
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
    "ldl",
    "hdl",
    "triglycerides",
    "hemoglobin",
    "smoking",
    "symptom_score",   # derived: sum of 4 binary eye symptoms (0-4)
]

# Human-readable labels shown in the SHAP response
FEATURE_LABELS: Dict[str, str] = {
    "age":               "Age (years)",
    "diabetes_duration": "Diabetes duration (years)",
    "hba1c":             "HbA1c (%)",
    "systolic_bp":       "Systolic BP (mmHg)",
    "diastolic_bp":      "Diastolic BP (mmHg)",
    "ldl":               "LDL cholesterol (mg/dL)",
    "hdl":               "HDL cholesterol (mg/dL)",
    "triglycerides":     "Triglycerides (mg/dL)",
    "hemoglobin":        "Haemoglobin (g/dL)",
    "smoking":           "Smoking",
    "symptom_score":     "Ocular symptom burden (0-4)",
}

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
N_CLASSES = len(CLASS_NAMES)

# Population-level means used to impute missing patient fields
_COHORT_MEANS: Dict[str, float] = {
    "age":               55.0,
    "diabetes_duration":  8.0,
    "hba1c":              7.8,
    "systolic_bp":       130.0,
    "diastolic_bp":       82.0,
    "ldl":               115.0,
    "hdl":                48.0,
    "triglycerides":     165.0,
    "hemoglobin":         13.5,
    "smoking":             0.25,
    "symptom_score":       0.5,
}


# ---------------------------------------------------------------------------
# Synthetic cohort generation + model training
# ---------------------------------------------------------------------------

_cached_model = None
_cached_explainer = None
_cached_background = None   # small background dataset for SHAP


def _generate_synthetic_cohort(n: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic DR-risk cohort with clinically plausible correlations.

    Risk formula (inspired by UKPDS / Wisconsin Epidemiologic Study findings):
        risk_score = 0.05*age + 0.12*diabetes_duration + 0.18*hba1c
                   + 0.04*systolic_bp + 0.02*ldl - 0.03*hdl
                   + 0.06*smoking + 0.05*symptom_score
                   + gaussian noise
    The continuous score is cut into 5 DR severity classes via population
    quintiles so each class has roughly equal representation.
    """
    rng = np.random.default_rng(seed)

    age               = rng.normal(55, 12, n).clip(18, 90)
    diabetes_duration = rng.exponential(8, n).clip(0.1, 40)
    hba1c             = rng.normal(7.8, 1.8, n).clip(4.5, 16.0)
    systolic_bp       = rng.normal(130, 18, n).clip(80, 200)
    diastolic_bp      = rng.normal(82, 11, n).clip(50, 120)
    ldl               = rng.normal(115, 35, n).clip(40, 280)
    hdl               = rng.normal(48, 13, n).clip(20, 100)
    triglycerides     = rng.normal(165, 75, n).clip(40, 600)
    hemoglobin        = rng.normal(13.5, 1.8, n).clip(7, 18)
    smoking           = rng.binomial(1, 0.25, n).astype(float)
    symptom_score     = rng.choice([0, 1, 2, 3, 4], n, p=[0.55, 0.22, 0.12, 0.07, 0.04]).astype(float)

    X = np.column_stack([
        age, diabetes_duration, hba1c, systolic_bp, diastolic_bp,
        ldl, hdl, triglycerides, hemoglobin, smoking, symptom_score,
    ])

    # Continuous risk score
    risk = (
        0.05  * (age - 55) / 12
        + 0.12  * (diabetes_duration - 8) / 8
        + 0.18  * (hba1c - 7.8) / 1.8
        + 0.04  * (systolic_bp - 130) / 18
        + 0.02  * (ldl - 115) / 35
        - 0.03  * (hdl - 48) / 13
        + 0.06  * smoking
        + 0.05  * symptom_score
        + rng.normal(0, 0.15, n)
    )

    # Map to 5 classes via quintiles
    thresholds = np.percentile(risk, [20, 40, 60, 80])
    y = np.digitize(risk, thresholds).astype(int)  # 0..4

    return X, y


def _build_model_and_explainer():
    global _cached_model, _cached_explainer, _cached_background

    if _cached_model is not None:
        return _cached_model, _cached_explainer, _cached_background

    lgbm = _get_lgbm()
    shap = _get_shap()

    print("[shap_service] Training LightGBM DR-risk model on synthetic cohort...")
    X, y = _generate_synthetic_cohort()

    model = lgbm.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X, y)
    print("[shap_service] LightGBM model trained. Building SHAP TreeExplainer...")

    # Use a small background dataset (100 rows) for marginal SHAP
    background = X[:100]
    explainer = shap.TreeExplainer(model, data=background, feature_names=FEATURE_COLUMNS)
    print("[shap_service] SHAP explainer ready.")

    _cached_model = model
    _cached_explainer = explainer
    _cached_background = background
    return model, explainer, background


# ---------------------------------------------------------------------------
# Blood-pressure parser
# ---------------------------------------------------------------------------

def _parse_blood_pressure(bp_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Parse '120/80', '120 / 80', '120-80', etc. → (systolic, diastolic)."""
    if not bp_str:
        return None, None
    m = re.search(r"(\d{2,3})\s*[/\-]\s*(\d{2,3})", str(bp_str))
    if m:
        return float(m.group(1)), float(m.group(2))
    # Single number — assume systolic only
    m2 = re.search(r"(\d{2,3})", str(bp_str))
    if m2:
        return float(m2.group(1)), None
    return None, None


# ---------------------------------------------------------------------------
# Session → feature vector
# ---------------------------------------------------------------------------

def _session_to_feature_vector(structured_data: Dict[str, Any]) -> np.ndarray:
    """
    Convert a patient session's structured_data into the model's feature vector.
    Missing fields are imputed with cohort-level means.
    """
    sd = structured_data or {}

    def _get(key: str) -> Optional[float]:
        v = sd.get(key)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    # Parse blood pressure
    sys_bp, dia_bp = _parse_blood_pressure(sd.get("blood_pressure"))

    # Symptom score: count of True symptoms
    symptoms = ["blurred_vision", "floaters", "vision_loss", "eye_pain"]
    known_symptoms = [sd.get(s) for s in symptoms if sd.get(s) is not None]
    symptom_score = float(sum(1 for s in known_symptoms if s is True)) if known_symptoms else None

    raw: Dict[str, Optional[float]] = {
        "age":               _get("age"),
        "diabetes_duration": _get("diabetes_duration"),
        "hba1c":             _get("hba1c"),
        "systolic_bp":       sys_bp,
        "diastolic_bp":      dia_bp,
        "ldl":               _get("ldl"),
        "hdl":               _get("hdl"),
        "triglycerides":     _get("triglycerides"),
        "hemoglobin":        _get("hemoglobin"),
        "smoking":           float(sd["smoking"]) if sd.get("smoking") is not None else None,
        "symptom_score":     symptom_score,
    }

    # Impute missing with cohort means
    vector = np.array(
        [raw[f] if raw[f] is not None else _COHORT_MEANS[f] for f in FEATURE_COLUMNS],
        dtype=np.float32,
    )
    return vector, raw  # also return raw for "was this imputed?" tracking


# ---------------------------------------------------------------------------
# Summary sentence builder
# ---------------------------------------------------------------------------

def _build_summary(
    shap_entries: List[Dict[str, Any]],
    predicted_label: str,
    top_n: int = 3,
) -> str:
    """
    Produce a plain-English one-sentence summary of the top risk drivers.
    """
    risk_drivers = [e for e in shap_entries if e["direction"] == "increases_risk"][:top_n]
    protective    = [e for e in shap_entries if e["direction"] == "decreases_risk"][:1]

    if not risk_drivers:
        return (
            f"The model predicts {predicted_label}. "
            "No single feature strongly dominates the prediction."
        )

    def _fmt(entry: Dict[str, Any]) -> str:
        label = FEATURE_LABELS.get(entry["feature"], entry["feature"])
        val = entry["value"]
        if entry["feature"] == "smoking":
            return f"{'active smoking' if val else 'non-smoking'}"
        if entry["feature"] == "symptom_score":
            return f"ocular symptom burden ({int(val)}/4)"
        return f"{label} ({val})"

    drivers_text = ", ".join(_fmt(e) for e in risk_drivers)
    summary = (
        f"The top risk factor{'s' if len(risk_drivers) > 1 else ''} "
        f"for this patient {'are' if len(risk_drivers) > 1 else 'is'}: {drivers_text}."
    )
    if protective:
        summary += f" {FEATURE_LABELS.get(protective[0]['feature'], protective[0]['feature'])} is a protective factor."
    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SHAPServiceError(Exception):
    """Raised when SHAP computation fails."""


def compute_shap(session_id: str, structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute SHAP values for a patient identified by their session data.

    Parameters
    ----------
    session_id      : The patient session identifier (for reference in response).
    structured_data : The ``structured_data`` dict from the session JSON.

    Returns
    -------
    A dict suitable for JSON serialisation containing:
        session_id, predicted_grade, predicted_label, confidence,
        probabilities, shap_values (ranked list), base_value, summary,
        imputed_fields (list of fields that used cohort means).
    """
    try:
        model, explainer, _ = _build_model_and_explainer()
    except ImportError as exc:
        raise SHAPServiceError(
            "lightgbm or shap is not installed. "
            "Run: pip install lightgbm shap"
        ) from exc

    feature_vector, raw_values = _session_to_feature_vector(structured_data)
    X_row = feature_vector.reshape(1, -1)

    # --- Prediction ---
    probabilities_array = model.predict_proba(X_row)[0]
    predicted_grade     = int(np.argmax(probabilities_array))
    predicted_label     = CLASS_NAMES[predicted_grade]
    confidence          = float(probabilities_array[predicted_grade])
    probabilities       = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probabilities_array)}

    # --- SHAP values ---
    # shap_values shape for multiclass: (n_samples, n_features, n_classes)
    shap_explanation = explainer(X_row)

    # Extract SHAP for the predicted class
    # shap_explanation.values: (1, n_features, n_classes)
    shap_vals_for_class = shap_explanation.values[0, :, predicted_grade]   # (n_features,)
    base_value          = float(shap_explanation.base_values[0, predicted_grade])

    # Build ranked entries
    imputed_fields: List[str] = [
        f for f in FEATURE_COLUMNS if raw_values.get(f) is None
    ]

    shap_entries: List[Dict[str, Any]] = []
    for feat, sv in zip(FEATURE_COLUMNS, shap_vals_for_class):
        raw_val = raw_values.get(feat)
        display_val = float(feature_vector[FEATURE_COLUMNS.index(feat)])
        shap_entries.append({
            "feature":   feat,
            "label":     FEATURE_LABELS.get(feat, feat),
            "value":     round(display_val, 3),
            "shap":      round(float(sv), 4),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
            "imputed":   raw_val is None,
        })

    # Sort by absolute SHAP descending
    shap_entries.sort(key=lambda e: abs(e["shap"]), reverse=True)

    summary = _build_summary(shap_entries, predicted_label)

    return {
        "session_id":      session_id,
        "predicted_grade": predicted_grade,
        "predicted_label": predicted_label,
        "confidence":      round(confidence, 4),
        "probabilities":   probabilities,
        "shap_values":     shap_entries,
        "base_value":      round(base_value, 4),
        "imputed_fields":  imputed_fields,
        "summary":         summary,
    }
