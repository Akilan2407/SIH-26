from typing import Any

from fastapi import HTTPException, UploadFile



async def read_fundus_image(upload: UploadFile) -> bytes:
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="Please upload a fundus image.")
    if upload.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Please upload a JPG, JPEG, or PNG image.")
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    return content


def predict(image_bytes: bytes) -> dict[str, Any]:
    try:
        from ml import dr_model
        return dr_model.predict_dr_severity_bytes(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="We couldn't process this retinal image. Please upload a valid fundus image.") from exc


def explain(image_bytes: bytes) -> dict[str, Any]:
    try:
        from ml import dr_model
        return dr_model.gradcam_images(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="We couldn't generate the AI explanation for this image.") from exc