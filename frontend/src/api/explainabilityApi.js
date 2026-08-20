import { mockExplainability } from "../mock/explainabilityMock.js";

const mock = import.meta.env.VITE_USE_MOCK_API === "true";
const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Normalise the raw response from either endpoint into a unified shape
// ---------------------------------------------------------------------------
export function normalizeExplainability(raw) {
  return {
    ...raw,
    gradCam: raw.gradCam || raw.grad_cam || raw.heatmap_image || null,
    overlay: raw.overlay || raw.overlay_url || raw.overlay_image || null,
    originalImage: raw.originalImage || raw.original_image || null,
    // SHAP fields
    shap: raw.shap || raw.shap_values || null,
    predicted_grade: raw.predicted_grade ?? null,
    predicted_label: raw.predicted_label ?? null,
    confidence: raw.confidence ?? null,
    probabilities: raw.probabilities ?? null,
    base_value: raw.base_value ?? null,
    imputed_fields: raw.imputed_fields ?? [],
    summary: raw.summary ?? null,
  };
}

// ---------------------------------------------------------------------------
// Grad-CAM — POST /screening/explain (multipart image upload)
// ---------------------------------------------------------------------------
export async function getGradCAM(file) {
  if (mock) return normalizeExplainability(await mockExplainability(file));

  const form = new FormData();
  form.append("image", file);

  const response = await fetch(`${BASE}/screening/explain`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "The AI explanation is not available yet.");
  }

  return normalizeExplainability(await response.json());
}

// ---------------------------------------------------------------------------
// SHAP — POST /screening/shap (JSON body with session_id + optional structured_data)
//
// sessionId   : chatbot session ID from localStorage["dr_chatbot_session_id"]
// structuredData : optional — if provided, skips disk lookup on the backend
// ---------------------------------------------------------------------------
export async function getSHAP(sessionId, structuredData = null) {
  if (mock) {
    // Return a plausible mock structure so the UI doesn't break in mock mode
    return normalizeExplainability({
      session_id: sessionId,
      predicted_grade: 1,
      predicted_label: "Mild",
      confidence: 0.61,
      probabilities: {
        "No DR": 0.22,
        Mild: 0.61,
        Moderate: 0.11,
        Severe: 0.04,
        "Proliferative DR": 0.02,
      },
      shap_values: [
        { feature: "hba1c", label: "HbA1c (%)", value: 7.8, shap: 0.42, direction: "increases_risk", imputed: false },
        { feature: "diabetes_duration", label: "Diabetes duration (years)", value: 8, shap: 0.31, direction: "increases_risk", imputed: false },
        { feature: "age", label: "Age (years)", value: 55, shap: 0.18, direction: "increases_risk", imputed: false },
        { feature: "systolic_bp", label: "Systolic BP (mmHg)", value: 130, shap: 0.12, direction: "increases_risk", imputed: true },
        { feature: "hdl", label: "HDL cholesterol (mg/dL)", value: 48, shap: -0.09, direction: "decreases_risk", imputed: true },
        { feature: "ldl", label: "LDL cholesterol (mg/dL)", value: 115, shap: 0.07, direction: "increases_risk", imputed: true },
        { feature: "smoking", label: "Smoking", value: 0, shap: -0.05, direction: "decreases_risk", imputed: false },
        { feature: "symptom_score", label: "Ocular symptom burden (0-4)", value: 1, shap: 0.03, direction: "increases_risk", imputed: false },
      ],
      base_value: -0.18,
      imputed_fields: ["systolic_bp", "diastolic_bp", "ldl", "hdl", "triglycerides", "hemoglobin"],
      summary: "The top risk factors for this patient are: HbA1c (%) (7.8), Diabetes duration (years) (8), Age (years) (55). HDL cholesterol (mg/dL) is a protective factor.",
    });
  }

  const body = { session_id: sessionId };
  if (structuredData !== null) body.structured_data = structuredData;

  const response = await fetch(`${BASE}/screening/shap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || "The tabular SHAP explanation is not available yet.");
  }

  return normalizeExplainability(await response.json());
}