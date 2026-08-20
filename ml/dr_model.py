from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    tf = None
    TENSORFLOW_IMPORT_ERROR = exc
else:
    TENSORFLOW_IMPORT_ERROR = None


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = 224

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# ============================================================
# Model path
# ============================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "prototype_model.keras"),
)


# ============================================================
# Load model ONCE
# ============================================================

model = None


def _get_model():
    global model
    if model is not None:
        return model
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Use Python 3.11-3.13 and install backend requirements for screening inference.") from TENSORFLOW_IMPORT_ERROR
    print("Loading diabetic retinopathy model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("DR model loaded successfully.")
    return model


def _image_tensor(image_bytes: bytes) -> tf.Tensor:
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return tf.cast(image, tf.float32)


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    """Use the existing wrapper's resize/cast preprocessing unchanged."""
    return tf.expand_dims(_image_tensor(image_bytes), axis=0)


def _prediction_from_tensor(image: tf.Tensor) -> dict[str, Any]:
    loaded_model = _get_model()
    probabilities = np.asarray(loaded_model(image, training=False)[0], dtype=np.float32)
    if probabilities.shape != (len(CLASS_NAMES),):
        raise ValueError("The loaded model did not return five class probabilities.")
    predicted_grade = int(np.argmax(probabilities))
    return {
        "grade": predicted_grade,
        "label": CLASS_NAMES[predicted_grade],
        "confidence": float(probabilities[predicted_grade]),
        "probabilities": {
            CLASS_NAMES[index]: float(probabilities[index])
            for index in range(len(CLASS_NAMES))
        },
    }


# ============================================================
# Prediction
# ============================================================

def predict_dr_severity(image_path):
    _get_model()
    return _prediction_from_tensor(preprocess_image(tf.io.read_file(image_path)))


def predict_dr_severity_bytes(image_bytes: bytes) -> dict[str, Any]:
    _get_model()
    return _prediction_from_tensor(preprocess_image(image_bytes))


def _find_gradcam_layer() -> tf.keras.layers.Layer:
    loaded_model = _get_model()
    candidates = []
    for layer in loaded_model.submodules:
        if layer is loaded_model or not hasattr(layer, "output"):
            continue
        try:
            shape = tuple(layer.output.shape)
        except (AttributeError, TypeError):
            continue
        if len(shape) == 4 and layer.__class__.__name__.lower().endswith("conv2d"):
            candidates.append(layer)
    if not candidates:
        raise RuntimeError("No convolutional layer is available for Grad-CAM.")
    return candidates[-1]


def _as_png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def gradcam_images(image_bytes: bytes) -> dict[str, Any]:
    loaded_model = _get_model()
    image_tensor = preprocess_image(image_bytes)
    base_model = loaded_model.get_layer("efficientnetb0")
    target_layer = _find_gradcam_layer()
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output],
    )
    with tf.GradientTape() as tape:
        activation, features = grad_model(image_tensor, training=False)
        pooled = loaded_model.get_layer("global_average_pooling2d")(features)
        dropped = loaded_model.get_layer("dropout")(pooled, training=False)
        predictions = loaded_model.get_layer("severity_output")(dropped)
        predicted_grade = tf.argmax(predictions[0])
        score = predictions[:, predicted_grade]
    gradients = tape.gradient(score, activation)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * activation, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], [IMG_SIZE, IMG_SIZE])[..., 0]
    heatmap_array = np.asarray(heatmap) * 255

    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = original.resize((IMG_SIZE, IMG_SIZE))
    color = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    color[..., 0] = heatmap_array.astype(np.uint8)
    color[..., 1] = (heatmap_array * 0.35).astype(np.uint8)
    heatmap_image = Image.fromarray(color, mode="RGB")
    overlay = Image.blend(original, heatmap_image, alpha=0.42)
    result = _prediction_from_tensor(image_tensor)
    return {
        **result,
        "gradcam_layer": target_layer.name,
        "original_image": _as_png_data_uri(original),
        "heatmap_image": _as_png_data_uri(heatmap_image),
        "overlay_image": _as_png_data_uri(overlay),
    }