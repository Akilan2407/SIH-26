"""
gradcam.py
----------
Task 1 — Gradient-weighted Class Activation Mapping (Grad-CAM) for the
Diabetic Retinopathy detection CNN.

Supports both TensorFlow/Keras and PyTorch backends. The correct backend
is selected automatically based on what is installed and what type of model
object is passed in.

Produces a heatmap overlay on a fundus image showing which retinal regions
most influenced the predicted DR stage, identifies the dominant retinal
quadrant, and maps (stage, quadrant) to a plain-language associated finding.

This module identifies model attention regions only. It is a decision-support
aid -- the outputs are NOT a clinical diagnosis.

Public API
----------
generate_gradcam(model, image_path, layer_name, class_idx, output_path)
    -> (heatmap_np, overlay_bgr_np)

get_quadrant_dominance(heatmap, image_size=(224, 224))
    -> {"quadrant": str, "mean_activations": dict}

get_lesion_finding(dr_stage, dominant_quadrant)
    -> str  (plain-language finding sentence)

gradcam_pipeline(model, image_path, dr_stage, layer_name, output_dir)
    -> {"overlay_image_path": str, "finding": str, "quadrant_info": dict, ...}

list_conv_layers(model)
    -> list of {"name", "type", "output_shape"}

save_gradcam_figure(original_image_path, overlay, heatmap, finding,
                    quadrant_info, output_path)
    -> str (path to saved 3-panel figure)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Optional framework imports
# ---------------------------------------------------------------------------

_HAS_TF = False
_HAS_TORCH = False

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants & Lookup Tables
# ---------------------------------------------------------------------------

#: DR stages: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
_STAGE_LESION_MAP: dict = {
    0: "no significant retinal changes",
    1: "early microaneurysms",
    2: "dot-blot haemorrhages and hard exudates",
    3: (
        "flame haemorrhages, venous beading, and intraretinal "
        "microvascular abnormalities (IRMAs)"
    ),
    4: "neovascularisation and fibrovascular proliferation",
}

_STAGE_LABEL_MAP: dict = {
    0: "No DR (Stage 0)",
    1: "Mild DR (Stage 1)",
    2: "Moderate DR (Stage 2)",
    3: "Severe DR (Stage 3)",
    4: "Proliferative DR (Stage 4)",
}

_QUADRANT_LABEL: dict = {
    "superior_temporal": "superior-temporal (upper-outer)",
    "superior_nasal":    "superior-nasal (upper-inner)",
    "inferior_temporal": "inferior-temporal (lower-outer)",
    "inferior_nasal":    "inferior-nasal (lower-inner)",
}


# ---------------------------------------------------------------------------
# Backend Detection
# ---------------------------------------------------------------------------

def _detect_backend(model) -> str:
    """
    Auto-detect whether model is a Keras or PyTorch model.

    Returns
    -------
    "keras" | "torch"
    """
    if _HAS_TF:
        try:
            if isinstance(model, tf.keras.Model):
                return "keras"
        except Exception:
            pass
    if _HAS_TORCH:
        try:
            if isinstance(model, nn.Module):
                return "torch"
        except Exception:
            pass
    raise RuntimeError(
        "Could not determine model framework. "
        "Model must be a tf.keras.Model or a torch.nn.Module. "
        "Ensure tensorflow or torch is installed."
    )


# ---------------------------------------------------------------------------
# Image Utilities (shared)
# ---------------------------------------------------------------------------

def _load_image(image_path: str, image_size: tuple) -> tuple:
    """
    Load, resize, and normalise a fundus image.

    Returns
    -------
    img_resized : np.ndarray (H, W, 3) uint8  -- original scale for overlay
    img_norm    : np.ndarray (H, W, 3) float32 in [0, 1]
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(
            f"Could not read image at '{image_path}'. "
            "Check the file exists and is a valid image format."
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size[1], image_size[0]))
    img_norm = img_resized.astype("float32") / 255.0
    return img_resized, img_norm


def _make_overlay(img_resized_rgb: np.ndarray, heatmap_norm: np.ndarray) -> np.ndarray:
    """
    Blend the normalised heatmap with the original fundus image (BGR output).
    """
    heatmap_coloured = cm.jet(heatmap_norm)[:, :, :3]
    heatmap_uint8 = (heatmap_coloured * 255).astype("uint8")
    heatmap_bgr = cv2.cvtColor(heatmap_uint8, cv2.COLOR_RGB2BGR)
    img_bgr = img_resized_rgb[:, :, ::-1]
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_bgr, 0.5, 0)
    return overlay


def _normalise_heatmap(raw: np.ndarray) -> np.ndarray:
    """Clip to non-negative and normalise to [0, 1]."""
    raw = np.maximum(raw, 0)
    vmin, vmax = raw.min(), raw.max()
    if vmax - vmin > 1e-8:
        return (raw - vmin) / (vmax - vmin)
    return np.zeros_like(raw)


# ---------------------------------------------------------------------------
# Keras Grad-CAM Backend
# ---------------------------------------------------------------------------

def _keras_gradcam(
    model,
    img_norm: np.ndarray,
    layer_name: Optional[str],
    class_idx: Optional[int],
    image_size: tuple,
) -> tuple:
    """
    Compute Grad-CAM using tf.GradientTape (Keras backend).

    Returns
    -------
    heatmap_upscaled : np.ndarray (H, W) float32 in [0, 1]
    predicted_class  : int
    """
    target_layer = _keras_resolve_layer(model, layer_name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    img_tensor = tf.expand_dims(img_norm, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_outputs)        # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)

    weighted = conv_outputs[0] * pooled_grads               # (h, w, C)
    heatmap_raw = tf.reduce_mean(weighted, axis=-1).numpy() # (h, w)

    heatmap_norm = _normalise_heatmap(heatmap_raw)
    heatmap_up = cv2.resize(
        heatmap_norm, (image_size[1], image_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return heatmap_up, class_idx


def _keras_resolve_layer(model, layer_name: Optional[str]):
    if layer_name is not None:
        try:
            return model.get_layer(layer_name)
        except ValueError as exc:
            available = [l.name for l in model.layers]
            raise ValueError(
                f"Layer '{layer_name}' not found. Available: {available}"
            ) from exc
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer
    raise ValueError("No Conv2D layer found. Provide an explicit layer_name.")


# ---------------------------------------------------------------------------
# PyTorch Grad-CAM Backend
# ---------------------------------------------------------------------------

def _torch_gradcam(
    model,
    img_norm: np.ndarray,
    layer_name: Optional[str],
    class_idx: Optional[int],
    image_size: tuple,
) -> tuple:
    """
    Compute Grad-CAM using PyTorch hooks (torch backend).

    Works with any nn.Module. For ResNet-style models, pass
    layer_name="layer4" or layer_name="layer4.1.conv2", etc.

    Returns
    -------
    heatmap_upscaled : np.ndarray (H, W) float32 in [0, 1]
    predicted_class  : int
    """
    target_layer = _torch_resolve_layer(model, layer_name)

    # Storage for forward activations and backward gradients
    activations: list = []
    gradients:   list = []

    def _fwd_hook(module, inp, out):
        activations.clear()
        activations.append(out.detach())

    def _bwd_hook(module, grad_in, grad_out):
        gradients.clear()
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    # (C, H, W) tensor
    img_t = torch.tensor(img_norm.transpose(2, 0, 1), dtype=torch.float32)
    img_t = img_t.unsqueeze(0)  # (1, C, H, W)

    model.eval()
    with torch.enable_grad():
        output = model(img_t)                                # (1, num_classes)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        score = output[0, class_idx]
        model.zero_grad()
        score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    act  = activations[0].squeeze(0)   # (C, h, w)
    grad = gradients[0].squeeze(0)     # (C, h, w)

    # Global-average-pooled importance weights
    weights = grad.mean(dim=(1, 2))    # (C,)
    cam = (act * weights[:, None, None]).sum(dim=0).numpy()  # (h, w)

    heatmap_norm = _normalise_heatmap(cam)
    heatmap_up = cv2.resize(
        heatmap_norm, (image_size[1], image_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return heatmap_up, class_idx


def _torch_resolve_layer(model, layer_name: Optional[str]):
    """
    Return a named submodule, or auto-detect the last Conv2d in the model.
    """
    if layer_name is not None:
        # Support dot-notation: "layer4.1.conv2"
        try:
            parts = layer_name.split(".")
            sub = model
            for p in parts:
                if p.isdigit():
                    sub = list(sub.children())[int(p)]
                else:
                    sub = getattr(sub, p)
            return sub
        except (AttributeError, IndexError) as exc:
            raise ValueError(
                f"Layer '{layer_name}' not found in model."
            ) from exc

    # Auto-detect: find the last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d found in model. Provide an explicit layer_name.")
    return last_conv


# ---------------------------------------------------------------------------
# Public: Core Grad-CAM Function
# ---------------------------------------------------------------------------

def generate_gradcam(
    model,
    image_path: str,
    layer_name: Optional[str] = None,
    class_idx: Optional[int] = None,
    output_path: Optional[str] = None,
    image_size: tuple = (224, 224),
) -> tuple:
    """
    Compute Grad-CAM for a single fundus image and optionally save the overlay.

    Automatically selects Keras or PyTorch backend based on model type.

    Parameters
    ----------
    model : tf.keras.Model | torch.nn.Module
        Trained DR detection model.
    image_path : str
        Path to the input fundus image.
    layer_name : str, optional
        Target convolutional layer name.
        - Keras : layer name string (e.g. "conv2d_2", "conv3_last")
        - PyTorch: attribute path (e.g. "layer4", "layer4.1.conv2")
        Auto-detected (last Conv layer) if None.
    class_idx : int, optional
        Class index to explain. Defaults to the predicted (argmax) class.
    output_path : str, optional
        If given, the heatmap overlay is saved as a PNG here.
    image_size : tuple
        (height, width) for image preprocessing. Must match model input size.

    Returns
    -------
    heatmap : np.ndarray  (H, W) float32 in [0, 1]
        Normalised Grad-CAM heatmap.
    overlay : np.ndarray  (H, W, 3) uint8 BGR
        Original fundus image with heatmap blended as colour overlay.
    """
    backend = _detect_backend(model)
    img_resized, img_norm = _load_image(image_path, image_size)

    if backend == "keras":
        heatmap, class_idx = _keras_gradcam(
            model, img_norm, layer_name, class_idx, image_size
        )
    else:  # "torch"
        heatmap, class_idx = _torch_gradcam(
            model, img_norm, layer_name, class_idx, image_size
        )

    overlay = _make_overlay(img_resized, heatmap)

    if output_path is not None:
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)

    return heatmap, overlay


# ---------------------------------------------------------------------------
# Public: Quadrant Dominance Analysis
# ---------------------------------------------------------------------------

def get_quadrant_dominance(
    heatmap: np.ndarray,
    image_size: tuple = (224, 224),
) -> dict:
    """
    Divide the Grad-CAM heatmap into four standard retinal quadrants and
    identify which quadrant has the highest mean activation.

    Quadrant layout (left-eye anatomical convention):
    +-------------------+--------------------+
    |  superior_nasal   |  superior_temporal |
    +-------------------+--------------------+
    |  inferior_nasal   |  inferior_temporal |
    +-------------------+--------------------+
    Nasal = towards optic disc / nose side (image left by default).
    Temporal = towards macula / ear side (image right by default).

    Parameters
    ----------
    heatmap : np.ndarray  (H, W) float32 in [0, 1]
        Normalised Grad-CAM heatmap.
    image_size : tuple
        (H, W) -- used only for documentation; heatmap.shape is used directly.

    Returns
    -------
    dict:
        "quadrant"         : str   -- name of the dominant quadrant
        "mean_activations" : dict  -- mean activation per quadrant
    """
    h, w = heatmap.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    quadrants = {
        "superior_nasal":    heatmap[:mid_h, :mid_w],
        "superior_temporal": heatmap[:mid_h, mid_w:],
        "inferior_nasal":    heatmap[mid_h:, :mid_w],
        "inferior_temporal": heatmap[mid_h:, mid_w:],
    }

    mean_activations = {k: float(np.mean(v)) for k, v in quadrants.items()}
    dominant_quadrant = max(mean_activations, key=mean_activations.get)

    return {
        "quadrant": dominant_quadrant,
        "mean_activations": mean_activations,
    }


# ---------------------------------------------------------------------------
# Public: Plain-Language Finding
# ---------------------------------------------------------------------------

def get_lesion_finding(dr_stage: int, dominant_quadrant: str) -> str:
    """
    Map a DR stage and dominant retinal quadrant to a plain-language finding.

    The sentence describes statistically associated model attention regions.
    It does NOT constitute a clinical diagnosis.

    Parameters
    ----------
    dr_stage : int
        Predicted DR stage (0-4).
    dominant_quadrant : str
        Dominant retinal quadrant key from get_quadrant_dominance().

    Returns
    -------
    str
        One-sentence associated finding suitable for the patient report.
    """
    stage_desc = _STAGE_LESION_MAP.get(
        dr_stage, "retinal changes of an unspecified stage"
    )
    stage_label = _STAGE_LABEL_MAP.get(dr_stage, f"Stage {dr_stage}")
    quadrant_label = _QUADRANT_LABEL.get(
        dominant_quadrant, dominant_quadrant.replace("_", "-")
    )

    return (
        f"Model attention was concentrated in the {quadrant_label} retinal region, "
        f"associated with {stage_desc} consistent with {stage_label}; "
        f"clinical correlation is recommended."
    )


# ---------------------------------------------------------------------------
# Public: High-Level Pipeline Function
# ---------------------------------------------------------------------------

def gradcam_pipeline(
    model,
    image_path: str,
    dr_stage: int,
    layer_name: Optional[str] = None,
    output_dir: str = "gradcam_outputs",
    image_size: tuple = (224, 224),
) -> dict:
    """
    Full Grad-CAM pipeline: generate heatmap overlay, identify dominant retinal
    quadrant, and produce a plain-language associated finding.

    Parameters
    ----------
    model : tf.keras.Model | torch.nn.Module
        Trained DR detection model.
    image_path : str
        Path to the input fundus image.
    dr_stage : int
        DR stage predicted by the model's argmax output (0-4).
    layer_name : str, optional
        Target convolutional layer. Auto-detected if None.
    output_dir : str
        Directory where the overlay PNG is saved.
    image_size : tuple
        (H, W) for resizing. Must match model input size.

    Returns
    -------
    dict:
        "overlay_image_path" : str   -- absolute path to the saved overlay PNG
        "finding"            : str   -- plain-language associated finding
        "quadrant_info"      : dict  -- dominant quadrant + mean activations
        "predicted_class"    : int   -- class index used for gradient computation
        "dr_stage_label"     : str   -- human-readable stage label
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{stem}_gradcam.png")

    heatmap, overlay = generate_gradcam(
        model=model,
        image_path=image_path,
        layer_name=layer_name,
        class_idx=dr_stage,
        output_path=output_path,
        image_size=image_size,
    )

    quadrant_info = get_quadrant_dominance(heatmap, image_size=image_size)
    finding = get_lesion_finding(dr_stage, quadrant_info["quadrant"])

    return {
        "overlay_image_path": os.path.abspath(output_path),
        "finding": finding,
        "quadrant_info": quadrant_info,
        "predicted_class": dr_stage,
        "dr_stage_label": _STAGE_LABEL_MAP.get(dr_stage, f"Stage {dr_stage}"),
    }


# ---------------------------------------------------------------------------
# Public: Utilities
# ---------------------------------------------------------------------------

def list_conv_layers(model) -> list:
    """
    List all convolutional layers in a model with their names and output shapes.
    Useful for choosing the right Grad-CAM target layer.

    Parameters
    ----------
    model : tf.keras.Model | torch.nn.Module

    Returns
    -------
    list of dicts with keys "name", "type", "output_shape" (Keras) or
    "name", "type", "in_channels", "out_channels" (PyTorch).
    """
    backend = _detect_backend(model)
    result = []

    if backend == "keras":
        conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
        for layer in model.layers:
            if isinstance(layer, conv_types):
                result.append({
                    "name": layer.name,
                    "type": type(layer).__name__,
                    "output_shape": layer.output_shape,
                })
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                result.append({
                    "name": name,
                    "type": "Conv2d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                })
    return result


def save_gradcam_figure(
    original_image_path: str,
    overlay: np.ndarray,
    heatmap: np.ndarray,
    finding: str,
    quadrant_info: dict,
    output_path: str,
) -> str:
    """
    Save a 3-panel matplotlib figure:
        Panel 1 -- original fundus image
        Panel 2 -- Grad-CAM attention overlay
        Panel 3 -- raw heatmap with quadrant grid and mean activation labels

    Parameters
    ----------
    original_image_path : str
    overlay : np.ndarray  (H, W, 3) BGR from generate_gradcam
    heatmap : np.ndarray  (H, W) float in [0, 1]
    finding : str
    quadrant_info : dict
    output_path : str

    Returns
    -------
    str : absolute path to the saved figure PNG
    """
    img_bgr = cv2.imread(str(original_image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (overlay.shape[1], overlay.shape[0]))
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    h, w = heatmap.shape
    mid_h, mid_w = h // 2, w // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")

    # Panel 1: original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Fundus Image", color="white", fontsize=11, pad=8)
    axes[0].axis("off")

    # Panel 2: Grad-CAM overlay
    axes[1].imshow(overlay_rgb)
    axes[1].set_title("Grad-CAM Attention Overlay", color="white", fontsize=11, pad=8)
    axes[1].axis("off")

    # Panel 3: heatmap + quadrant grid
    im = axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].axhline(mid_h, color="white", linewidth=1.2, linestyle="--", alpha=0.7)
    axes[2].axvline(mid_w, color="white", linewidth=1.2, linestyle="--", alpha=0.7)

    ma = quadrant_info["mean_activations"]
    dom = quadrant_info["quadrant"]
    quad_pos = {
        "superior_nasal":    (mid_h // 2,             mid_w // 2,             "SN"),
        "superior_temporal": (mid_h // 2,             mid_w + mid_w // 2,     "ST"),
        "inferior_nasal":    (mid_h + mid_h // 2,     mid_w // 2,             "IN"),
        "inferior_temporal": (mid_h + mid_h // 2,     mid_w + mid_w // 2,     "IT"),
    }
    for qname, (row, col, abbr) in quad_pos.items():
        colour = "#FFD700" if qname == dom else "white"
        axes[2].text(
            col, row, f"{abbr}\n{ma[qname]:.3f}",
            ha="center", va="center", color=colour,
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Heatmap + Retinal Quadrants", color="white", fontsize=11, pad=8)
    axes[2].axis("off")

    fig.suptitle(finding, color="#a8dadc", fontsize=9, y=0.02, ha="center")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Self-contained Demo (no trained weights or real images required)
# ---------------------------------------------------------------------------

def _demo_torch():
    """Smoke-test with a minimal PyTorch CNN."""
    import tempfile
    print("\n[PyTorch backend demo]")

    # Minimal ResNet-like model
    class TinyResNet(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.relu  = nn.ReLU()
            self.pool  = nn.AdaptiveAvgPool2d((7, 7))
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.layer4 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
            )
            self.gap  = nn.AdaptiveAvgPool2d(1)
            self.fc   = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.layer4(x)
            x = self.gap(x)
            x = x.flatten(1)
            return self.fc(x)

    model = TinyResNet(num_classes=5)

    print("Conv layers in TinyResNet:")
    for info in list_conv_layers(model):
        print(f"  {info['name']:20s} | in={info['in_channels']:2d} out={info['out_channels']:2d}")

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "synthetic_fundus.png")
        fake_img = np.zeros((224, 224, 3), dtype=np.uint8)
        fake_img[:, :, 1] = 120
        cv2.imwrite(img_path, fake_img)

        result = gradcam_pipeline(
            model=model,
            image_path=img_path,
            dr_stage=4,
            layer_name="layer4.0",   # first Conv2d inside layer4 Sequential
            output_dir=os.path.join(tmpdir, "outputs"),
        )

    print(f"  DR Stage      : {result['dr_stage_label']}")
    print(f"  Dominant quad : {result['quadrant_info']['quadrant']}")
    print(f"  Finding:\n    {result['finding']}")
    print("  [PyTorch backend] OK")


def _demo_keras():
    """Smoke-test with a minimal Keras CNN."""
    import tempfile
    print("\n[Keras backend demo]")

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = tf.keras.layers.Conv2D(8,  3, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 4, padding="same", activation="relu", name="conv3_last")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, x)

    print("Conv layers in Keras model:")
    for info in list_conv_layers(model):
        print(f"  {info['name']:20s} | output: {info['output_shape']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "synthetic_fundus.png")
        fake_img = np.zeros((224, 224, 3), dtype=np.uint8)
        fake_img[:, :, 1] = 120
        cv2.imwrite(img_path, fake_img)

        result = gradcam_pipeline(
            model=model,
            image_path=img_path,
            dr_stage=2,
            layer_name="conv3_last",
            output_dir=os.path.join(tmpdir, "outputs"),
        )

    print(f"  DR Stage      : {result['dr_stage_label']}")
    print(f"  Dominant quad : {result['quadrant_info']['quadrant']}")
    print(f"  Finding:\n    {result['finding']}")
    print("  [Keras backend] OK")


def _demo():
    print("=" * 60)
    print("gradcam.py — self-contained demo")
    print("=" * 60)
    print(f"TensorFlow available : {_HAS_TF}")
    print(f"PyTorch available    : {_HAS_TORCH}")

    if not (_HAS_TF or _HAS_TORCH):
        print("\nERROR: Neither tensorflow nor torch is installed.")
        print("Install at least one: pip install torch  OR  pip install tensorflow")
        return

    if _HAS_TORCH:
        _demo_torch()
    if _HAS_TF:
        _demo_keras()

    print("\nDemo complete.")


if __name__ == "__main__":
    _demo()
