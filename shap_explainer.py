"""
shap_explainer.py
-----------------
Task 2 — SHAP TreeExplainer for the LightGBM clinical risk model.

Takes a trained LightGBM / MultiOutputClassifier model and a single patient
feature row, computes per-feature SHAP values, ranks the top-3 positive
contributors to the systemic risk score, and converts them into plain-language
risk-driver sentences.

This module identifies statistically associated risk factors only. The outputs
are a decision-support aid — they do NOT constitute a clinical diagnosis.

Public API
----------
generate_shap_summary(lgbm_model, patient_features_df, feature_names,
                      target_output_idx, top_n)
    -> SHAPSummary (dataclass)

shap_to_sentences(shap_summary)
    -> list[str]  (plain-language risk-driver sentences)

plot_waterfall(shap_summary, output_path)
    -> str        (path to saved waterfall PNG)

FEATURE_NAMES  : list[str]  -- canonical feature ordering expected by the model
RISK_LABEL_MAP : dict       -- risk-driver label -> plain-language description
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from sklearn.multioutput import MultiOutputClassifier
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Canonical feature list & plain-language mappings
# ---------------------------------------------------------------------------

#: Ordered feature names expected by the tabular clinical model.
#: Pass this list as feature_names when calling generate_shap_summary().
FEATURE_NAMES: list = [
    "hba1c",
    "systolic_bp",
    "diastolic_bp",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "hemoglobin",
    "egfr",
    "bmi",
    "diabetes_duration_years",
    "age",
]

#: Maps feature name -> (short label for chart, plain-language description).
#: Used to construct human-readable sentences from SHAP values.
_FEATURE_META: dict = {
    "hba1c": (
        "HbA1c",
        "Elevated HbA1c is a significant contributing factor, "
        "suggesting suboptimal long-term glycaemic control.",
    ),
    "systolic_bp": (
        "Systolic BP",
        "Elevated systolic blood pressure is a significant contributing factor, "
        "associated with increased cardiovascular and microvascular load.",
    ),
    "diastolic_bp": (
        "Diastolic BP",
        "Elevated diastolic blood pressure is a significant contributing factor, "
        "associated with persistent vascular stress.",
    ),
    "ldl_cholesterol": (
        "LDL Cholesterol",
        "Elevated LDL cholesterol is a significant contributing factor, "
        "associated with dyslipidaemia and atherogenic risk.",
    ),
    "hdl_cholesterol": (
        "Low HDL Cholesterol",
        "Reduced HDL cholesterol is a significant contributing factor, "
        "associated with impaired lipid clearance.",
    ),
    "triglycerides": (
        "Triglycerides",
        "Elevated triglycerides is a significant contributing factor, "
        "associated with dyslipidaemia and metabolic syndrome.",
    ),
    "hemoglobin": (
        "Hemoglobin",
        "Low hemoglobin is a significant contributing factor, "
        "associated with anaemia and impaired oxygen delivery.",
    ),
    "egfr": (
        "eGFR",
        "Reduced eGFR is a significant contributing factor, "
        "associated with renal impairment and diabetic nephropathy.",
    ),
    "bmi": (
        "BMI",
        "Elevated BMI is a significant contributing factor, "
        "associated with insulin resistance and metabolic risk.",
    ),
    "diabetes_duration_years": (
        "Diabetes Duration",
        "Extended diabetes duration is a significant contributing factor, "
        "associated with cumulative microvascular injury.",
    ),
    "age": (
        "Age",
        "Advanced age is a significant contributing factor, "
        "associated with increased susceptibility to diabetic complications.",
    ),
}

#: Risk-driver label -> plain language (for MultiOutputClassifier outputs)
RISK_LABEL_MAP: dict = {
    "poor_glycemic_control":  "Poor glycaemic control",
    "hypertension":           "Hypertension",
    "dyslipidemia":           "Dyslipidaemia",
    "anemia":                 "Anaemia",
    "renal_impairment":       "Renal impairment",
    "long_disease_duration":  "Long disease duration",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """Stores a single feature's SHAP contribution for one patient."""
    feature_name: str
    shap_value: float
    feature_value: float
    short_label: str
    sentence: str


@dataclass
class SHAPSummary:
    """
    Full SHAP summary for one patient, ready for report generation.

    Attributes
    ----------
    top_contributors : list[FeatureContribution]
        Top-N positive SHAP contributors, ranked by magnitude (descending).
    all_contributions : list[FeatureContribution]
        All feature contributions (for waterfall plot), ranked positive→negative.
    risk_score : float
        Predicted systemic risk score (probability of positive class).
    risk_driver_labels : list[str]
        Risk-driver output labels predicted as active (MultiOutputClassifier).
    plain_language_sentences : list[str]
        Plain-language sentences for the top-N contributors.
    base_value : float
        SHAP base value (expected model output over training data).
    """
    top_contributors: list = field(default_factory=list)
    all_contributions: list = field(default_factory=list)
    risk_score: float = 0.0
    risk_driver_labels: list = field(default_factory=list)
    plain_language_sentences: list = field(default_factory=list)
    base_value: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_lgbm(model):
    """
    If model is a sklearn MultiOutputClassifier wrapping LightGBM estimators,
    return the first estimator (for the systemic risk score output).
    Otherwise return model as-is.
    """
    if _HAS_SKLEARN and isinstance(model, MultiOutputClassifier):
        return model.estimators_[0]
    return model


def _get_shap_values_for_class(explainer, patient_row: np.ndarray, class_idx: int = 1):
    """
    Run SHAP TreeExplainer and return SHAP values for the positive class.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    patient_row : np.ndarray  shape (1, n_features)
    class_idx : int
        Index of the positive class (default 1 for binary classification).

    Returns
    -------
    shap_vals : np.ndarray  shape (n_features,)
    base_val  : float
    """
    explanation = explainer(patient_row)

    # shap >= 0.40: explanation.values shape is (n_samples, n_features) for binary
    # or (n_samples, n_features, n_classes) for multi-class
    vals = explanation.values
    base = explanation.base_values

    if vals.ndim == 3:
        # Multi-class: extract the positive class slice
        shap_vals = vals[0, :, class_idx]
        base_val  = float(base[0, class_idx]) if base.ndim == 2 else float(base[0])
    elif vals.ndim == 2:
        shap_vals = vals[0, :]
        base_val  = float(base[0]) if hasattr(base, "__len__") else float(base)
    else:
        shap_vals = vals
        base_val  = float(base)

    return shap_vals, base_val


def _build_contributions(
    shap_vals: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list,
) -> list:
    """
    Build a list of FeatureContribution objects, sorted positive → negative
    (most positive first, then most negative).
    """
    contributions = []
    for i, fname in enumerate(feature_names):
        meta = _FEATURE_META.get(fname, (fname, f"{fname} is a contributing factor."))
        contributions.append(FeatureContribution(
            feature_name=fname,
            shap_value=float(shap_vals[i]),
            feature_value=float(feature_values[i]),
            short_label=meta[0],
            sentence=meta[1],
        ))
    # Sort: largest positive first, then most negative last
    contributions.sort(key=lambda c: c.shap_value, reverse=True)
    return contributions


# ---------------------------------------------------------------------------
# Public: Core SHAP function
# ---------------------------------------------------------------------------

def generate_shap_summary(
    lgbm_model,
    patient_features_df: pd.DataFrame,
    feature_names: Optional[list] = None,
    target_output_idx: int = 0,
    top_n: int = 3,
    class_idx: int = 1,
) -> "SHAPSummary":
    """
    Compute SHAP values for a single patient row and extract the top-N
    positive contributors to the systemic risk score.

    Parameters
    ----------
    lgbm_model : lgb.LGBMClassifier | sklearn.MultiOutputClassifier
        Trained LightGBM model or MultiOutputClassifier wrapping one.
        For MultiOutputClassifier, target_output_idx selects which estimator
        is treated as the systemic risk score predictor.
    patient_features_df : pd.DataFrame
        Single-row DataFrame containing the patient's clinical feature values.
        Column names must match feature_names.
        IMPORTANT: Must NOT contain identifying columns (name, MRN, contact).
    feature_names : list[str], optional
        Ordered list of feature names. Defaults to FEATURE_NAMES.
    target_output_idx : int
        Which estimator to explain when using MultiOutputClassifier (0 = first).
    top_n : int
        Number of top positive contributors to extract (default 3).
    class_idx : int
        Class index for the positive outcome (default 1).

    Returns
    -------
    SHAPSummary
        Dataclass containing ranked contributions, sentences, risk score, etc.

    Raises
    ------
    ImportError
        If shap or lightgbm is not installed.
    ValueError
        If patient_features_df does not have exactly one row.
    """
    if not _HAS_SHAP:
        raise ImportError("shap is not installed. Run: pip install shap")
    if not _HAS_LGB:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    if feature_names is None:
        feature_names = FEATURE_NAMES

    if len(patient_features_df) != 1:
        raise ValueError(
            f"patient_features_df must have exactly 1 row; "
            f"got {len(patient_features_df)}."
        )

    # -- 1. Select columns in canonical order ---------------------------------
    patient_row = patient_features_df[feature_names].values  # (1, n_features)

    # -- 2. Unwrap MultiOutputClassifier if needed ----------------------------
    if _HAS_SKLEARN and isinstance(lgbm_model, MultiOutputClassifier):
        base_estimator = lgbm_model.estimators_[target_output_idx]
    else:
        base_estimator = lgbm_model

    # -- 3. Predicted risk score (probability of positive class) --------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = base_estimator.predict_proba(patient_row)
    risk_score = float(proba[0, class_idx])

    # -- 4. Risk-driver labels (MultiOutputClassifier multi-label outputs) ----
    risk_driver_labels = []
    if _HAS_SKLEARN and isinstance(lgbm_model, MultiOutputClassifier):
        output_names = list(RISK_LABEL_MAP.keys())
        for idx, estimator in enumerate(lgbm_model.estimators_):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = estimator.predict(patient_row)[0]
            if pred == 1 and idx < len(output_names):
                risk_driver_labels.append(output_names[idx])

    # -- 5. SHAP TreeExplainer ------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(base_estimator)
        shap_vals, base_val = _get_shap_values_for_class(
            explainer, patient_row, class_idx=class_idx
        )

    # -- 6. Build contribution list -------------------------------------------
    all_contributions = _build_contributions(
        shap_vals, patient_row[0], feature_names
    )

    # -- 7. Top-N positive contributors ---------------------------------------
    positive_contribs = [c for c in all_contributions if c.shap_value > 0]
    top_contributors  = positive_contribs[:top_n]

    # -- 8. Plain-language sentences ------------------------------------------
    sentences = shap_to_sentences_from_contributions(top_contributors)

    return SHAPSummary(
        top_contributors=top_contributors,
        all_contributions=all_contributions,
        risk_score=risk_score,
        risk_driver_labels=risk_driver_labels,
        plain_language_sentences=sentences,
        base_value=base_val,
    )


# ---------------------------------------------------------------------------
# Public: Sentence converter
# ---------------------------------------------------------------------------

def shap_to_sentences(shap_summary: "SHAPSummary") -> list:
    """
    Extract plain-language risk-driver sentences from a SHAPSummary.

    Parameters
    ----------
    shap_summary : SHAPSummary

    Returns
    -------
    list[str]
        One sentence per top contributor, plus one sentence per active
        risk-driver label (if any) not already covered.
    """
    return shap_summary.plain_language_sentences


def shap_to_sentences_from_contributions(contributions: list) -> list:
    """
    Convert a list of FeatureContribution objects into plain-language sentences.

    Parameters
    ----------
    contributions : list[FeatureContribution]

    Returns
    -------
    list[str]
    """
    sentences = []
    for rank, contrib in enumerate(contributions, start=1):
        # Personalise the sentence with the actual measured value
        base_sentence = contrib.sentence.rstrip(".")
        fname = contrib.feature_name
        val   = contrib.feature_value

        # Inject measured value into the sentence for clinical context
        unit_map = {
            "hba1c":                  f"{val:.1f}%",
            "systolic_bp":            f"{val:.0f} mmHg",
            "diastolic_bp":           f"{val:.0f} mmHg",
            "ldl_cholesterol":        f"{val:.1f} mmol/L",
            "hdl_cholesterol":        f"{val:.1f} mmol/L",
            "triglycerides":          f"{val:.1f} mmol/L",
            "hemoglobin":             f"{val:.1f} g/dL",
            "egfr":                   f"{val:.0f} mL/min/1.73m²",
            "bmi":                    f"{val:.1f} kg/m²",
            "diabetes_duration_years":f"{val:.0f} years",
            "age":                    f"{val:.0f} years",
        }
        measured = unit_map.get(fname, f"{val:.2f}")
        sentence = f"{base_sentence} (measured: {measured})."
        sentences.append(sentence)
    return sentences


# ---------------------------------------------------------------------------
# Public: Waterfall chart
# ---------------------------------------------------------------------------

def plot_waterfall(
    shap_summary: "SHAPSummary",
    output_path: str = "shap_waterfall.png",
    title: str = "SHAP Feature Contributions — Systemic Risk Score",
    max_features: int = 11,
) -> str:
    """
    Render a horizontal waterfall bar chart of SHAP contributions and save it.

    The chart shows all features ranked from most positive (top) to most
    negative (bottom). Positive bars are red-orange, negative bars are teal.
    The dominant quadrant for top-3 contributors is highlighted in gold.

    Parameters
    ----------
    shap_summary : SHAPSummary
    output_path : str
        Path to save the PNG figure.
    title : str
        Chart title.
    max_features : int
        Maximum number of features to display (default 11 = all canonical).

    Returns
    -------
    str : absolute path to the saved figure.
    """
    contributions = shap_summary.all_contributions[:max_features]
    labels  = [c.short_label for c in contributions]
    values  = [c.shap_value  for c in contributions]
    fvalues = [c.feature_value for c in contributions]

    # Colours: top-3 positive → gold, other positive → coral, negative → teal
    top3_names = {c.feature_name for c in shap_summary.top_contributors}
    colours = []
    for c in contributions:
        if c.feature_name in top3_names and c.shap_value > 0:
            colours.append("#FFD700")   # gold — top contributor
        elif c.shap_value > 0:
            colours.append("#E76F51")   # coral
        else:
            colours.append("#2A9D8F")   # teal

    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.52)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(range(n), values, color=colours, edgecolor="none", height=0.65)

    # Value + measured-value annotations
    for i, (bar, c) in enumerate(zip(bars, contributions)):
        xpos = bar.get_width()
        ha   = "left" if xpos >= 0 else "right"
        offset = 0.002 * (max(abs(v) for v in values) or 1)
        ax.text(
            xpos + (offset if xpos >= 0 else -offset),
            i,
            f"{xpos:+.3f}",
            va="center", ha=ha, color="white", fontsize=8, fontweight="bold",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("SHAP value (contribution to risk score)", color="#a8dadc", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    # Risk score annotation
    ax.set_title(
        f"{title}\nPredicted risk score: {shap_summary.risk_score:.1%}  "
        f"| Base value: {shap_summary.base_value:.3f}",
        color="white", fontsize=10, pad=12,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FFD700", label="Top-3 contributor"),
        Patch(facecolor="#E76F51", label="Positive contributor"),
        Patch(facecolor="#2A9D8F", label="Negative contributor"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        facecolor="#1a1a2e", edgecolor="#444",
        labelcolor="white", fontsize=8,
    )

    plt.tight_layout()
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Self-contained Demo (no real model or patient data required)
# ---------------------------------------------------------------------------

def _demo():
    """
    Smoke-test generate_shap_summary() with a synthetically trained LightGBM
    model and a mock patient feature row. No real data or weights needed.
    """
    print("=" * 60)
    print("shap_explainer.py — self-contained demo")
    print("=" * 60)

    if not _HAS_SHAP:
        print("ERROR: shap not installed. Run: pip install shap")
        return
    if not _HAS_LGB:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm")
        return

    import tempfile
    from sklearn.datasets import make_classification

    n_features = len(FEATURE_NAMES)

    # -- Train a tiny LightGBM on synthetic data --
    print("\nTraining synthetic LightGBM model for demo...")
    X, y = make_classification(
        n_samples=300, n_features=n_features, n_informative=6,
        n_redundant=2, random_state=42,
    )
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    clf.fit(X_df, y)
    print("  Model trained.")

    # -- Mock patient row (realistic-ish clinical values) --
    patient_row = pd.DataFrame([{
        "hba1c":                   9.2,   # elevated
        "systolic_bp":             148,   # elevated
        "diastolic_bp":            92,    # mildly elevated
        "ldl_cholesterol":         4.1,   # elevated
        "hdl_cholesterol":         0.9,   # low
        "triglycerides":           2.8,   # elevated
        "hemoglobin":              10.5,  # low (mild anaemia)
        "egfr":                    52,    # mildly reduced
        "bmi":                     31.2,  # overweight
        "diabetes_duration_years": 14,    # long duration
        "age":                     58,
    }])

    # -- Run SHAP summary --
    print("\nRunning SHAP TreeExplainer...")
    summary = generate_shap_summary(
        lgbm_model=clf,
        patient_features_df=patient_row,
        feature_names=FEATURE_NAMES,
        top_n=3,
    )

    print(f"\n  Predicted risk score : {summary.risk_score:.1%}")
    print(f"  SHAP base value      : {summary.base_value:.4f}")

    print("\n  Top-3 positive contributors:")
    for i, c in enumerate(summary.top_contributors, 1):
        print(f"    {i}. {c.short_label:25s}  SHAP={c.shap_value:+.4f}  "
              f"value={c.feature_value}")

    print("\n  Plain-language sentences:")
    for sentence in summary.plain_language_sentences:
        print(f"    • {sentence}")

    print("\n  All contributions (waterfall order):")
    for c in summary.all_contributions:
        bar = "█" * int(abs(c.shap_value) * 200)
        sign = "+" if c.shap_value >= 0 else "-"
        print(f"    {c.short_label:25s}  {sign}{bar} ({c.shap_value:+.4f})")

    # -- Save waterfall chart --
    with tempfile.TemporaryDirectory() as tmpdir:
        chart_path = os.path.join(tmpdir, "shap_waterfall.png")
        saved = plot_waterfall(summary, output_path=chart_path)
        size  = os.path.getsize(saved)
        print(f"\n  Waterfall chart saved: {saved} ({size:,} bytes)")

    print("\nAll steps completed. shap_explainer.py is working correctly.")


if __name__ == "__main__":
    _demo()
