from fastapi import APIRouter, File, UploadFile

from backend.services.model_service import explain, predict, read_fundus_image

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