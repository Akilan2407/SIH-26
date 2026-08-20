from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.model_service import explain, predict, read_fundus_image
from backend.core import session_manager
from backend.models.schemas import SHAPRequest, SHAPResponse
from backend.services.shap_service import SHAPServiceError, compute_shap

router = APIRouter(prefix="/screening", tags=["screening"])


@router.post("/predict")
async def predict_screening(image: UploadFile = File(...)):
    result = predict(await read_fundus_image(image))
    return {
        "predicted_grade": result["grade"],
        "severity": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    }


@router.post("/explain")
async def explain_screening(image: UploadFile = File(...)):
    result = explain(await read_fundus_image(image))
    return {
        "predicted_grade": result["grade"],
        "severity": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "gradcam_layer": result["gradcam_layer"],
        "original_image": result["original_image"],
        "heatmap_image": result["heatmap_image"],
        "overlay_image": result["overlay_image"],
    }


# ---------------------------------------------------------------------------
# POST /screening/shap — accepts structured_data inline or loads from session
# ---------------------------------------------------------------------------

@router.post("/shap", response_model=SHAPResponse, summary="Compute SHAP tabular explanation (POST)")
async def shap_post(payload: SHAPRequest):
    """
    Compute SHAP feature-importance values from a JSON request body.

    Preferred over the GET variant for frontend calls — the client can pass
    ``structured_data`` directly (no need for a fully persisted session file).

    If ``structured_data`` is omitted the session is loaded from disk by
    ``session_id`` and its stored ``structured_data`` is used instead.

    Returns per-feature Shapley values ranked by |shap|, LightGBM class
    probabilities across all 5 DR severity grades, a plain-English summary
    of the top risk drivers, and a list of fields imputed from cohort means.
    """
    sid = (payload.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required.")

    # Prefer inline structured_data; fall back to persisted session on disk.
    if payload.structured_data is not None:
        structured_data = payload.structured_data
    else:
        session = session_manager.load_session(sid)
        structured_data = session.get("structured_data", {})

    try:
        result = compute_shap(sid, structured_data)
    except SHAPServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP computation failed: {exc}",
        ) from exc

    return result


# ---------------------------------------------------------------------------
# GET /screening/{session_id}/shap — session-based lookup (kept for backwards compat)
# ---------------------------------------------------------------------------

@router.get("/{session_id}/shap", response_model=SHAPResponse, summary="Compute SHAP tabular explanation (GET)")
async def shap_get(session_id: str):
    """
    Compute SHAP feature-importance values for a patient's tabular risk profile.

    Loads the session identified by ``session_id``, extracts the clinical
    structured data collected during the chatbot intake, and returns:

    - Per-feature SHAP values ranked by absolute contribution
    - LightGBM class probabilities across all 5 DR severity grades
    - A plain-English summary of the top risk drivers
    - A list of fields that were imputed from cohort means (if the patient
      did not provide them during intake)

    The LightGBM model is trained once (lazily) on a 2 000-row synthetic
    cohort whose risk correlations mirror published DR epidemiology.
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    session = session_manager.load_session(session_id)
    structured_data = session.get("structured_data", {})

    try:
        result = compute_shap(session_id, structured_data)
    except SHAPServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP computation failed: {exc}",
        ) from exc

    return result